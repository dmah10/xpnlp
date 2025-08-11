"""
Training and evaluation utilities for transformer models.

This module provides functions for:
1. Computing evaluation metrics (accuracy, F1, TPR, TNR)
2. Text preprocessing for transformer models
3. Model evaluation on test sets
"""

import evaluate as hf_eval
import numpy as np


def compute_metrics(eval_pred):
    """Compute evaluation metrics for model predictions.
    
    Calculates:
    - Accuracy
    - Macro F1 score
    - True Positive Rate (TPR)
    - True Negative Rate (TNR)
    
    Args:
        eval_pred (tuple): (logits, labels) from model predictions
        
    Returns:
        dict: Dictionary containing computed metrics
    """
    # Load metric functions
    load_accuracy = hf_eval.load("evaluate/metrics/accuracy/accuracy.py")
    load_f1 = hf_eval.load("evaluate/metrics/f1/f1.py")
    load_conf = hf_eval.load("evaluate/metrics/confusion_matrix/confusion_matrix.py")

    # Extract predictions
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Compute basic metrics
    accuracy = load_accuracy.compute(
        predictions=predictions, 
        references=labels
    )["accuracy"]
    
    f1 = load_f1.compute(
        predictions=predictions, 
        references=labels, 
        average="macro"
    )["f1"]
    
    # Compute confusion matrix and derived metrics
    cm = load_conf.compute(
        predictions=predictions, 
        references=labels
    )["confusion_matrix"]
    
    # Calculate TPR and TNR from confusion matrix
    # cm layout: [[TN, FP], [FN, TP]]
    tpr = cm[1][1] / sum(cm[1])  # TP / (TP + FN)
    tnr = cm[0][0] / sum(cm[0])  # TN / (TN + FP)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "tpr": tpr,
        "tnr": tnr
    }


def preprocess_function(tokenizer):
    """Create a preprocessing function for text data.
    
    Args:
        tokenizer: HuggingFace tokenizer instance
        
    Returns:
        callable: Function that tokenizes text with given parameters
    """
    return lambda examples: tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )


def evaluate(
    model,
    testset,
    in_key="text",
    out_key="label",
    pred_key="score",
    max_length=512
):
    """Evaluate model performance on a test set.
    
    Args:
        model: Model to evaluate
        testset: Dataset for evaluation
        in_key (str): Key for input text in dataset
        out_key (str): Key for labels in dataset
        pred_key (str): Key for prediction scores in output
        max_length (int): Maximum sequence length
        
    Returns:
        float: Accuracy score on test set
    """
    # Tokenizer settings
    tokenizer_kwargs = {
        "padding": True,
        "truncation": True,
        "max_length": max_length,
    }
    
    # Get model predictions
    preds = model(testset[in_key], **tokenizer_kwargs)
    
    # Convert predictions to labels
    pred_labels = [
        max(range(len(pred)), key=lambda i: pred[i][pred_key])
        for pred in preds
    ]
    
    # Calculate accuracy
    correct = (np.array(pred_labels) == np.array(testset[out_key])).sum()
    return correct / len(testset)
