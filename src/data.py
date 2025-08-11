"""
Dataset loading and preprocessing utilities for fairness and explainability analysis.

This module provides functions for:
1. Loading and preprocessing various text classification datasets:
   - AG News (news classification)
   - SST2 (sentiment analysis)
   - Trustpilot (review sentiment)
   - CrowS-Pairs (bias analysis)
   - GecoBench (gender bias analysis)
2. Converting between different dataset formats
3. Handling dataset splits and filtering
"""

import numpy as np
import pandas as pd
import json
from datasets import Dataset, Features, Value
from typing import Dict, List, Tuple, Optional, Any, Union

# Dataset configurations mapping dataset names to their file info
# Format: {
#   "dataset_name": {
#     "filename": "file.csv",
#     "input_col": "text/sentence column name",
#     "label_col": "label column name"
#   }
# }
DATASETS = {
    # AG News dataset variants
    "ag_news": {
        "filename": "ag_news_sample.csv",
        "input_col": "text",
        "label_col": "label",
    },
    "ag_news-1": {"filename": "ag_TEM_1_sample.csv", "input_col": "text", "label_col": "label"},
    "ag_news-2": {"filename": "ag_TEM_2_sample.csv", "input_col": "text", "label_col": "label"},
    "ag_news-3": {"filename": "ag_TEM_3_sample.csv", "input_col": "text", "label_col": "label"},
    "ag_news-4": {"filename": "ag_DPPrompt_118_sample.csv", "input_col": "text", "label_col": "label"},
    "ag_news-5": {"filename": "ag_DPPrompt_137_sample.csv", "input_col": "text", "label_col": "label"},
    "ag_news-6": {"filename": "ag_DPPrompt_165_sample.csv", "input_col": "text", "label_col": "label"},
    "ag_news-7": {"filename": "ag_DPBart_500.csv", "input_col": "text", "label_col": "label"},
    "ag_news-8": {"filename": "ag_DPBart_1000.csv", "input_col": "text", "label_col": "label"},
    "ag_news-9": {"filename": "ag_DPBart_1500.csv", "input_col": "text", "label_col": "label"},
    
    # SST2 dataset variants
    "sst2": {"filename": "sst2_train.csv", "input_col": "sentence", "label_col": "label"},
    "sst2-1": {"filename": "sst2_TEM_1.csv", "input_col": "sentence", "label_col": "label"},
    "sst2-2": {"filename": "sst2_TEM_2.csv", "input_col": "sentence", "label_col": "label"},
    "sst2-3": {"filename": "sst2_TEM_3.csv", "input_col": "sentence", "label_col": "label"},
    "sst2-4": {"filename": "sst2_DPPrompt_118.csv", "input_col": "sentence", "label_col": "label"},
    "sst2-5": {"filename": "sst2_DPPrompt_137.csv", "input_col": "sentence", "label_col": "label"},
    "sst2-6": {"filename": "sst2_DPPrompt_165.csv", "input_col": "sentence", "label_col": "label"},
    "sst2-7": {"filename": "sst2_DPBart_500.csv", "input_col": "sentence", "label_col": "label"},
    "sst2-8": {"filename": "sst2_DPBart_1000.csv", "input_col": "sentence", "label_col": "label"},
    "sst2-9": {"filename": "sst2_DPBart_1500.csv", "input_col": "sentence", "label_col": "label"},
    
    # Trustpilot dataset variants
    "trustpilot": {"filename": "trustpilot.csv", "input_col": "text", "label_col": "sentiment"},
    "trustpilot-1": {"filename": "trustpilot_TEM_1.csv", "input_col": "text", "label_col": "sentiment"},
    "trustpilot-2": {"filename": "trustpilot_TEM_2.csv", "input_col": "text", "label_col": "sentiment"},
    "trustpilot-3": {"filename": "trustpilot_TEM_3.csv", "input_col": "text", "label_col": "sentiment"},
    "trustpilot-4": {"filename": "trustpilot_DPPrompt_118.csv", "input_col": "text", "label_col": "sentiment"},
    "trustpilot-5": {"filename": "trustpilot_DPPrompt_137.csv", "input_col": "text", "label_col": "sentiment"},
    "trustpilot-6": {"filename": "trustpilot_DPPrompt_165.csv", "input_col": "text", "label_col": "sentiment"},
    "trustpilot-7": {"filename": "trustpilot_DPBart_500.csv", "input_col": "text", "label_col": "sentiment"},
    "trustpilot-8": {"filename": "trustpilot_DPBart_1000.csv", "input_col": "text", "label_col": "sentiment"},
    "trustpilot-9": {"filename": "trustpilot_DPBart_1500.csv", "input_col": "text", "label_col": "sentiment"},
    
    # Yelp dataset
    "yelp": {"filename": "yelp.csv", "input_col": "review", "label_col": "sentiment_id"},
}


def crows_drop_cols(df: pd.DataFrame, more: bool = True) -> pd.DataFrame:
    """Drop unnecessary columns from CrowS-Pairs dataset.
    
    Args:
        df: Input dataframe containing CrowS-Pairs data
        more: If True, drops sent_less column, otherwise drops sent_more
        
    Returns:
        DataFrame with specified columns removed
    """
    return df.drop(
        columns=[
            "sent_less" if more else "sent_more",
            "annotations",
            "anon_writer",
            "bias_type",
            "anon_annotators",
            "stereo_antistereo",
        ],
        inplace=False,
    )


def load_crows(
    base_path: str = "./data",
    test_size: float = 0.1,
    bias_types: List[str] = ["gender", "race"]
) -> Tuple[Dataset, Dataset]:
    """Load CrowS-Pairs dataset for bias analysis.
    
    Args:
        base_path: Path to data directory
        test_size: Proportion of data to use for testing
        bias_types: Types of bias to analyze
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    # Load and preprocess data
    sent_more_df = pd.read_csv(f"{base_path}/crows.csv")
    sent_less_df = sent_more_df.copy()
    
    # Drop unnecessary columns and rename
    sent_more_df = crows_drop_cols(sent_more_df)
    sent_less_df = crows_drop_cols(sent_less_df, False)
    sent_more_df.rename(columns={"sent_more": "text"}, inplace=True)
    sent_less_df.rename(columns={"sent_less": "text"}, inplace=True)
    
    # Assign labels
    sent_more_df["label"] = 1
    sent_less_df["label"] = 0
    
    # Combine dataframes
    full_df = pd.concat([sent_more_df, sent_less_df], ignore_index=True)
    full_df.drop(
        columns=[col for col in full_df.columns if col not in ["text", "label"]],
        inplace=True,
    )
    
    # Convert to HuggingFace dataset
    dataset = Dataset.from_pandas(
        full_df,
        features=Features({
            "text": Value("string"),
            "label": Value("int32"),
        }),
        preserve_index=False,
    ).class_encode_column("label")
    
    # Create train/test split
    split_d = dataset.train_test_split(test_size=test_size, seed=42)
    return split_d["train"], split_d["test"]


def load_sentiment_dataset(
    name: str,
    base_path: str = "./data",
    test_size: float = 0.1,
    shrink: bool = False,
    max_length: int = 10_000,
) -> Tuple[Dataset, Dataset, int]:
    """Load sentiment analysis datasets (AG News, SST2, Trustpilot).
    
    Args:
        name: Dataset name from DATASETS dictionary
        base_path: Path to data directory
        test_size: Proportion of data to use for testing
        shrink: Whether to use a smaller subset (128 samples)
        max_length: Maximum text length to include
        
    Returns:
        Tuple of (train_dataset, test_dataset, num_labels)
        
    Raises:
        ValueError: If dataset name is not found in DATASETS
    """
    # Load dataset
    df = pd.read_csv(f"{base_path}/{DATASETS[name]['filename']}")
    
    # Use smaller subset if requested
    if shrink:
        df = df.sample(n=128, random_state=42)
    
    # Get column names and number of labels
    input_col, label_col = DATASETS[name]["input_col"], DATASETS[name]["label_col"]
    n_labels = len(df[label_col].unique())
    
    # Clean up dataframe
    to_drop = [col for col in df.columns if col not in [input_col, label_col]]
    df.drop(columns=to_drop, inplace=True)
    df.rename(columns={input_col: "text", label_col: "label"}, inplace=True)
    
    # Filter by length if specified
    df = df[df["text"].str.len() <= max_length]
    
    # Convert to HuggingFace dataset
    dataset = Dataset.from_pandas(
        df,
        features=Features({
            "text": Value("string"),
            "label": Value("int32"),
        }),
        preserve_index=False,
    ).class_encode_column("label")
    
    # Create train/test split
    split_d = dataset.train_test_split(test_size=test_size, seed=42)
    return split_d["train"], split_d["test"], n_labels


def load_gecobench(path: str, return_df: bool = False) -> Union[Dataset, pd.DataFrame]:
    """Load GecoBench dataset for gender bias analysis.
    
    Args:
        path: Path to dataset file
        return_df: Whether to return pandas DataFrame instead of HuggingFace dataset
        
    Returns:
        Dataset or DataFrame containing the processed data
    """
    # Read JSON lines
    lines = []
    with open(path) as f:
        lines = f.read().splitlines()
    
    # Convert to DataFrame
    line_dicts = [json.loads(line) for line in lines]
    df = pd.DataFrame(line_dicts)
    
    # Process text and labels
    df["text"] = df.apply(lambda x: " ".join(x["sentence"]), axis=1)
    df["label"] = df["target"]
    df["gender"] = df.apply(lambda x: "male" if x["label"] == 1 else "female", axis=1)
    df["mask"] = df.apply(lambda x: str(x["ground_truth"]), axis=1)
    
    if return_df:
        return df
    
    # Drop unnecessary columns
    df.drop(
        labels=["ground_truth", "sentence_idx", "sentence", "target"],
        inplace=True,
        axis=1,
    )
    
    # Convert to HuggingFace dataset
    return Dataset.from_pandas(df)


def resplit_gecobench(
    trainset: Dataset,
    testset: Dataset,
    train_rate: float = 0.8
) -> Tuple[Dataset, Dataset]:
    """Resplit GecoBench dataset into new train/test sets.
    
    Args:
        trainset: Original training set
        testset: Original test set
        train_rate: Proportion to use for training
        
    Returns:
        Tuple of (new_trainset, new_testset)
    """
    # Combine datasets
    combined = pd.concat([
        trainset.to_pandas(),
        testset.to_pandas(),
    ])
    
    # Create new split
    n_train = int(len(combined) * train_rate)
    train_idx = np.random.choice(len(combined), size=n_train, replace=False)
    test_idx = list(set(range(len(combined))) - set(train_idx))
    
    # Convert back to datasets
    return (
        Dataset.from_pandas(combined.iloc[train_idx]),
        Dataset.from_pandas(combined.iloc[test_idx]),
    )


def get_df(path: str) -> pd.DataFrame:
    """Load JSON lines file into DataFrame.
    
    Args:
        path: Path to JSON lines file
        
    Returns:
        DataFrame containing parsed data
    """
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient="index")


def parse(path: str) -> Dict:
    """Parse JSON lines file.
    
    Args:
        path: Path to JSON lines file
        
    Returns:
        Generator yielding parsed JSON objects
    """
    g = open(path, "rb")
    for l in g:
        yield json.loads(l)