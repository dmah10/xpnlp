"""
Metrics for evaluating transformer model explanations.

This module provides metrics for:
1. Soft evaluation of explanations
2. Sparsity measures (Gini index, threshold sparsity)
3. Sensitivity evaluation using adversarial perturbations
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel
from sklearn.preprocessing import MinMaxScaler

from ferret.explainers.explanation import Explanation
from ferret.evaluators import BaseEvaluator
from ferret.evaluators.evaluation import Evaluation
from src.metrics_util import *


@torch.no_grad()
def soft_eval(
    explanation: Explanation,
    model,
    tokenizer,
    device: torch.device,
    remove: bool = False
) -> float:
    """Evaluate explanation by masking tokens based on importance scores.
    
    Args:
        explanation: Model explanation to evaluate
        model: Model to evaluate against
        tokenizer: Tokenizer for processing text
        device: Device to run computation on
        remove: Whether to remove (True) or keep (False) important tokens
        
    Returns:
        float: Evaluation score (0-1)
    """
    model.config.output_hidden_states = True
    scaler = MinMaxScaler()
    scores, tokens, target = explanation.scores, explanation.tokens, explanation.target

    # Normalize scores to 0-1 range
    normalized_scores = torch.tensor(
        scaler.fit_transform(scores.reshape(-1, 1)).flatten()
    )

    # Get embeddings from model
    x = tokenizer(explanation.text)
    inputs = {
        k: torch.tensor(v).unsqueeze(0).to(device)
        for k, v in x.items()
        if k in tokenizer.model_input_names
    }
    outputs = model(**inputs)
    logit = outputs.logits[0][target]
    embeddings = outputs.hidden_states[0]

    # Create and apply mask
    mask_prob = normalized_scores.unsqueeze(dim=0).T.repeat(1, embeddings.size(-1))
    if remove:
        mask_prob = 1 - mask_prob
    mask_prob = torch.clamp(mask_prob, 0, 1)
    mask = torch.bernoulli(mask_prob).int()

    # Apply mask to embeddings
    size = embeddings.size()
    embeddings = embeddings.flatten()[mask.flatten()].view(size)

    # Get new prediction
    new_logit = model(inputs_embeds=embeddings).logits[0][target]
    s = max(torch.tensor(0), logit - new_logit).item()
    
    return s if remove else 1 - s


def gini_index(explanation: Explanation, eps: float = 1e-12) -> float:
    """Calculate Gini index for sparseness of attribution vector.
    
    Implementation follows:
    https://proceedings.mlr.press/v119/chalasani20a/chalasani20a.pdf
    
    Args:
        explanation: Model explanation to evaluate
        eps: Small value to avoid division by zero
        
    Returns:
        float: Gini index (0-1, higher is more sparse)
    """
    # Get absolute values and sort
    v = np.sort(np.abs(explanation.scores) + eps)
    k = np.arange(1, v.shape[0] + 1)  # sorted indices
    d = v.shape[0]  # size
    
    # Calculate Gini index
    return 1 - 2 * np.sum((v / np.linalg.norm(v, ord=1)) * (1 - k / d + 0.5 / d))


def threshold_sparsity(explanation: Explanation, t: float = 0.01) -> float:
    """Calculate proportion of scores above threshold.
    
    Args:
        explanation: Model explanation to evaluate
        t: Threshold value
        
    Returns:
        float: Proportion of scores above threshold (0-1)
    """
    return np.sum(np.abs(explanation.scores) >= t) / explanation.scores.shape[0]


class EvaluationMetricOutput:
    """Container for metric evaluation results."""
    metric: BaseEvaluator
    value: float


class Sensitivity_Evaluation(BaseEvaluator):
    """Evaluator for measuring explanation sensitivity using adversarial perturbations."""
    
    NAME = "auc_sensitivity"
    SHORT_NAME = "sens"
    LOWER_IS_BETTER = True
    MIN_VALUE = 0.0
    MAX_VALUE = 1.0
    BEST_VALUE = 0.0
    METRIC_FAMILY = "faithfulness"
    BEST_SORTING_ASCENDING = False
    TYPE_METRIC = "faithfulness"

    def compute_evaluation(
        self,
        explanation: Explanation,
        target: int = 1,
        **evaluation_args
    ) -> Optional[Evaluation]:
        """Evaluate explanation sensitivity using adversarial perturbations.
        
        Args:
            explanation: Model explanation to evaluate
            target: Target class label
            **evaluation_args: Additional arguments including:
                - remove_first_last: Whether to remove [CLS]/[SEP] tokens
                - name_model: Model name/path
                
        Returns:
            Evaluation result or None if evaluation fails
        """
        if isinstance(explanation, list):
            return None
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Parsing additional evaluation arguments
        remove_first_last, _, _, top_k, name_model = parse_evaluator_args_model(
            evaluation_args
        )

        removal_args = {"remove_tokens": True, "based_on": "k"}
        if isinstance(explanation, list):
            return None

        text = explanation.text
        tokenizer = self.helper.tokenizer
        score_explanation = explanation.scores

        removal_args["remove_tokens"] = True

        # Tokenize the input text
        item = self.helper._tokenize(text)
        input_len = item["attention_mask"].sum().item()
        input_ids = item["input_ids"][0][:input_len].tolist()

        # Remove first and last tokens if specified
        if remove_first_last == True:
            input_ids = input_ids[1:-1]
            if self.tokenizer.cls_token == explanation.tokens[0]:
                score_explanation = score_explanation[1:-1]

        # Define thresholds and epsilon values for evaluation
        n = input_len - 1
        n = min(n, top_k)
        thresholds = [i for i in range(1, n)]
        epsilons = []

        # Load classification model and base model
        name = name_model
        # classification_model = AutoModelForSequenceClassification.from_pretrained(name)
        classification_model = self.helper.model.to(device)  # for fine-tuned weights
        base_model = AutoModel.from_pretrained(name, force_download=False).to(device)

        # Tokenize text for the model
        inputs = tokenizer(text, return_tensors="pt").to(device)
        if remove_first_last == True:
            inputs = {
                "input_ids": inputs["input_ids"][:, 1:-1],
                "attention_mask": inputs["attention_mask"][:, 1:-1],
            }

        # Get token embeddings
        with torch.no_grad():
            outputs = base_model(**inputs)
            token_embeddings = outputs.last_hidden_state

        def pgd_attack(
            embeddings: torch.Tensor,
            epsilon: float,
            alpha: float,
            num_steps: int,
            perturbation_vector: List[int],
            y: torch.Tensor
        ) -> torch.Tensor:
            """Perform PGD attack on embeddings.
            
            Args:
                embeddings: Token embeddings to perturb
                epsilon: Maximum perturbation size
                alpha: Step size
                num_steps: Number of optimization steps
                perturbation_vector: Binary mask for which tokens to perturb
                y: Target labels
                
            Returns:
                Perturbed embeddings
            """
            perturbed_embeddings = embeddings.clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([perturbed_embeddings], lr=alpha)

            for _ in range(num_steps):
                optimizer.zero_grad()
                
                # Get model prediction
                classificator_inputs = {
                    "attention_mask": inputs["attention_mask"],
                    "inputs_embeds": perturbed_embeddings,
                }
                logits = classification_model(**classificator_inputs).logits
                loss = F.cross_entropy(logits, y)
                loss.backward()

                with torch.no_grad():
                    grad = perturbed_embeddings.grad.clone().to(device)
                    mask = (
                        torch.tensor(perturbation_vector, dtype=torch.float32)
                        .unsqueeze(0)
                        .unsqueeze(-1)
                        .to(device)
                    )
                    mask = mask.expand_as(grad)
                    grad = grad * mask
                    perturbed_embeddings += alpha * grad.sign()
                    perturbation = torch.clamp(
                        perturbed_embeddings - embeddings, -epsilon, epsilon
                    )
                    perturbed_embeddings = (
                        (embeddings + perturbation).detach().requires_grad_(True)
                    )

                classification_model.zero_grad()
                if perturbed_embeddings.grad is not None:
                    perturbed_embeddings.grad.zero_()

                return perturbed_embeddings

            # Binary search for optimal epsilon value
            def binary_search_epsilon(
                token_embeddings,
                perturbation_vector,
                alpha,
                num_steps,
                y,
                tol=1e-3,
                max_iter=15,
            ):
                low = 0.0
                high = 1.0
                best_epsilon = low

                for i in range(max_iter):
                    mid = (low + high) / 2.0
                    perturbed_embeddings = pgd_attack(
                        token_embeddings, mid, alpha, num_steps, perturbation_vector, y
                    )
                    classificator_inputs = {
                        "attention_mask": inputs["attention_mask"],
                        "inputs_embeds": perturbed_embeddings,
                    }

                    with torch.no_grad():
                        classification_outputs = classification_model(
                            **classificator_inputs
                        )
                        logits = classification_outputs.logits
                        predictions = torch.argmax(logits, dim=-1)

                    if predictions != y:
                        best_epsilon = mid
                        high = mid
                    else:
                        low = mid

                    if high - low < tol:
                        break

                return best_epsilon

            # Compute epsilon for current threshold
            epsilon = binary_search_epsilon(
                token_embeddings, perturbation_vector, alpha, num_steps, y
            )
            epsilons.append(epsilon)

        # Compute AUC for sensitivity
        sens_auc = np.trapz(epsilons, thresholds)
        # print(f"Epsilons: {epsilons}, Sensitivity AUC: {sens_auc}")

        evaluation_output = Evaluation(self.SHORT_NAME, sens_auc)
        return evaluation_output
