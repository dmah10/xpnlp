#!/usr/bin/env python3
"""
Main script for training and evaluating transformer models with explanations.
This script handles:
1. Model fine-tuning on text classification tasks
2. Generation of explanations using various methods
3. Evaluation of explanation quality and fairness
4. Result logging and storage
"""

import os
import sys
from datetime import datetime
from itertools import product
import pickle
from pathlib import Path

# Configure PyTorch memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Add parent directory to path for imports
sys.path.insert(1, os.path.join(sys.path[0], ".."))

# External imports
from tqdm import tqdm
import wandb
import torch
import gc

# Transformer-related imports
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# Ferret framework imports
from ferret import Benchmark
from ferret.evaluators.evaluation import Evaluation
from ferret.explainers.gradient import GradientExplainer, IntegratedGradientExplainer
from ferret.explainers.lime import LIMEExplainer
from ferret.explainers.shap import SHAPExplainer

# Local imports
from src.data import load_sentiment_dataset, load_crows
from src.train import compute_metrics, preprocess_function
from src.util import df_from_ferret_scores
from src.parser import parse_args
from src.metrics import (
    gini_index,
    threshold_sparsity,
    Sensitivity_Evaluation,
    soft_eval,
)

# Parse command line arguments
args = parse_args()

# Set up device (GPU if available, else CPU)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("running on ", device)
print(args)

# Main loop over models, datasets, and repetitions
for model_str, dataset_str, r in product(args.models, args.datasets, range(args.reps)):
    timestamp = datetime.today().strftime("%Y%m%d%H%M%S")
    seed = args.seeds[r]

    # Initialize Weights & Biases logging if specified
    if args.project is not None:
        wandb.login(relogin=False)
        wandb.init(
            project=args.project,
            config=vars(args),
            dir=Path(".").resolve(),
            reinit=True,
        )
        config = wandb.config
        config.model = model_str
        config.dataset = dataset_str

    # Load and preprocess dataset
    if dataset_str == "crows":
        trainset, testset = load_crows()
        n_labels = 2
    else:
        trainset, testset, n_labels = load_sentiment_dataset(
            dataset_str,
            shrink=args.shrink,
            max_length=args.max_length,
        )
    trainset = trainset.shuffle()

    # Initialize model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        model_str,
        num_labels=n_labels,
        trust_remote_code=True,
        force_download=False,
        ignore_mismatched_sizes=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_str,
        trust_remote_code=True,
        force_download=False,
    )

    # Tokenize datasets
    tokenized_train = trainset.map(preprocess_function(tokenizer), batched=True)
    tokenized_test = testset.map(preprocess_function(tokenizer), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    eval_set = testset

    # Set up model paths and training
    model_name = model_str.split("/")[-1]
    model_path = f"models/{model_name}_{dataset_str}.pt"

    # Fine-tune model if requested
    if args.train:
        training_args = TrainingArguments(
            output_dir="test_trainer",
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            save_steps=1e99,
            evaluation_strategy="steps",
            optim="adamw_torch",
            save_strategy="no",
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        # Train and evaluate
        trainer.train()
        pred_results = trainer.predict(tokenized_test)

        if args.project is not None:
            wandb.log(pred_results.metrics)
        
        model = model.to(device)
        if not args.test_run:
            model.save_pretrained(model_path, from_pt=True)
    else:
        # Load pre-trained model if not training
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model = model.to(device)

    # Initialize Ferret benchmark with explainers
    bench = Benchmark(model, tokenizer)
    bench.explainers = [
        SHAPExplainer(bench.model, bench.tokenizer),
        LIMEExplainer(bench.model, bench.tokenizer),
        GradientExplainer(bench.model, bench.tokenizer, multiply_by_inputs=False),
        IntegratedGradientExplainer(bench.model, bench.tokenizer, multiply_by_inputs=False),
    ]

    # Initialize sensitivity benchmark if requested
    if args.sensitivity:
        sens_bench = Benchmark(
            model, tokenizer, evaluators=[Sensitivity_Evaluation(model, tokenizer)]
        )

    # Storage for explanations and evaluations
    exp_evals = {"explanations": [], "evaluations": []}
    lengths = []
    failed_indices = []

    # Generate and evaluate explanations for each sample
    for j, sample in enumerate(tqdm(eval_set)):
        print("SENTENCE: ", sample["text"])
        print("----")
        try:
            torch.cuda.empty_cache()
            if args.test_run and j == 2:
                break

            # Skip samples that would require truncation
            tokenized_sample = tokenizer(sample["text"])
            if len(tokenized_sample["input_ids"]) > args.max_tokens:
                continue

            # Generate explanations
            explanations = bench.explain(sample["text"], target=sample["label"], show_progress=False)

            # Evaluate explanations
            if not args.sensitivity:
                evaluations = bench.evaluate_explanations(
                    explanations,
                    target=sample["label"],
                    show_progress=False,
                    remove_first_last=True,
                )
            else:
                evaluations = sens_bench.evaluate_explanations(
                    explanations,
                    target=sample["label"],
                    show_progress=True,
                    remove_first_last=True,
                    top_k=1,
                    name=model_str,
                )

            # Calculate additional metrics
            if not args.sensitivity:
                for i, eval in enumerate(evaluations):
                    if not args.soft_only:
                        gini = gini_index(eval.explanation)
                        s = threshold_sparsity(eval.explanation)
                    else:
                        gini, s = 0, 0

                    # Calculate soft comprehensiveness and sufficiency
                    soft_compr = soft_eval(
                        eval.explanation, model, tokenizer, device, remove=True
                    )
                    soft_suff = soft_eval(
                        eval.explanation, model, tokenizer, device, remove=False
                    )

                    eval.evaluation_scores += [
                        Evaluation(name="gini_index", score=gini),
                        Evaluation(name="sparsity", score=s),
                        Evaluation(name="soft_compr", score=soft_compr),
                        Evaluation(name="soft_suff", score=soft_suff),
                    ]

            lengths.append(len(sample["text"]))
            exp_evals["explanations"].append(explanations)
            exp_evals["evaluations"].append(evaluations)

        except Exception as e:
            failed_indices.append(j)
            continue

    # Save results
    results = df_from_ferret_scores(exp_evals["evaluations"], lengths)

    if not os.path.exists("results"):
        os.makedirs("results")

    save_filename = f"{model_str.split('/')[-1]}_{dataset_str}_{timestamp}{'_sens' if args.sensitivity else ''}.csv"
    results.to_csv(f"results/{save_filename}")

    # Clean up to free memory
    del model
    del tokenizer
    del data_collator
    del tokenized_train
    del tokenized_test
    del bench
    if args.sensitivity:
        del sens_bench

    torch.cuda.empty_cache()
    gc.collect()

    # save results to wandb
    if args.project is not None:
        name = wandb.run.name
        run_path = f"{args.data_dir}/{name}"
        if not os.path.exists(run_path):
            os.makedirs(run_path)

        with open(f"{run_path}/failed.csv", "w+") as f:
            # write the indices of the failed inputs to the csv file
            for item in failed_indices:
                f.write(f"{item}\n")

        if args.save:
            # save explanations
            with open(f"{run_path}/explanations.pkl", "wb+") as f:
                pickle.dump(exp_evals["explanations"], f)
        wandb.save(f"{run_path}/failed.csv")
        wandb.save(f"results/{save_filename}")
        wandb.finish()
