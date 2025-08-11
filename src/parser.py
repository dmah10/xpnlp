"""
Command line argument parser for the fairness-xai framework.

This module defines all command line arguments for:
- Data configuration
- Model selection and training
- Experiment control
- Evaluation settings
- Output management
"""

from argparse import ArgumentParser
import sys


def parse_args(args=sys.argv[1:]):
    """Parse command line arguments for the fairness-xai framework.
    
    Args:
        args: Command line arguments (default: sys.argv[1:])
        
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = ArgumentParser(description="Fairness and Explainability Analysis Framework")

    # Data Configuration
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument(
        "--data-dir", 
        type=str, 
        default="./data", 
        help="Directory to save the data"
    )
    data_group.add_argument(
        "--train-size", 
        type=int, 
        default=20_000, 
        help="Training set size"
    )
    data_group.add_argument(
        "--test-size", 
        type=int, 
        default=10_000, 
        help="Test set size"
    )
    data_group.add_argument(
        "--max-length", 
        type=int, 
        default=1_000_000, 
        help="Maximum input length in characters"
    )
    data_group.add_argument(
        "--max-tokens", 
        type=int, 
        default=512, 
        help="Maximum input length in tokens"
    )
    data_group.add_argument(
        "--min-length", 
        type=int, 
        default=0, 
        help="Minimum input length in characters"
    )
    data_group.add_argument(
        "--split-idx", 
        type=int, 
        default=2, 
        help="Index of the dataset split"
    )

    # Model and Training
    model_group = parser.add_argument_group("Model and Training")
    model_group.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=["huawei-noah/TinyBERT_General_4L_312D"],
        help="List of Hugging Face model identifiers"
    )
    model_group.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    model_group.add_argument(
        "--batch-size", 
        type=int, 
        default=16, 
        help="Training batch size"
    )
    model_group.add_argument(
        "--train",
        action="store_true",
        help="Enable model fine-tuning"
    )

    # Experiment Control
    exp_group = parser.add_argument_group("Experiment Control")
    exp_group.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        help="List of datasets to use"
    )
    exp_group.add_argument(
        "--reps", 
        type=int, 
        default=1, 
        help="Number of experiment repetitions"
    )
    exp_group.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Global random seed"
    )
    exp_group.add_argument(
        "--seeds", 
        type=int, 
        nargs="*", 
        default=[1, 2, 3, 4, 5],
        help="List of random seeds for multiple runs"
    )
    exp_group.add_argument(
        "--test-run",
        action="store_true",
        help="Run minimal test experiment"
    )
    exp_group.add_argument(
        "--shrink", 
        action="store_true",
        help="Use reduced dataset size"
    )
    exp_group.add_argument(
        "--random",
        action="store_true",
        help="Use random labels (for testing)"
    )

    # Evaluation Settings
    eval_group = parser.add_argument_group("Evaluation Settings")
    eval_group.add_argument(
        "--only-explain",
        action="store_true",
        help="Generate explanations without evaluation"
    )
    eval_group.add_argument(
        "--sensitivity",
        action="store_true",
        help="Compute sensitivity metrics"
    )
    eval_group.add_argument(
        "--soft-only",
        action="store_true",
        help="Only compute soft evaluation metrics"
    )

    # Output Management
    output_group = parser.add_argument_group("Output Management")
    output_group.add_argument(
        "--project", 
        type=str, 
        default="fairness", 
        help="Weights & Biases project name"
    )
    output_group.add_argument(
        "--wandb", 
        type=str, 
        default=None,
        help="Weights & Biases API key"
    )
    output_group.add_argument(
        "--save",
        action="store_true",
        help="Save results locally"
    )

    return parser.parse_args(args)
