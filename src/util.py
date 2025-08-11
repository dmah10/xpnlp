import pandas as pd
import numpy as np
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt


def cohens_d(v1, v2):
    """
    Computes the Cohen's d between the two lists of values v1 and v2.
    """
    n1 = len(v1)
    n2 = len(v2)
    s1 = np.std(v1)
    s2 = np.std(v2)
    mu1 = np.mean(v1)
    mu2 = np.mean(v2)
    s = np.sqrt(((n1 - 1) * (s1**2) + (n2 - 1) * (s2**2)) / (n1 + n2 - 2))
    return (mu1 - mu2) / s


def show_boxplot(results):
    """
    Plots boxplots for each metric and explainer, similar to figure 3 in https://arxiv.org/pdf/2205.07277
    """
    sns.set_theme(style="ticks", palette="pastel")
    metrics = [
        "aopc_compr",
        "aopc_suff",
        "taucorr_loo",
        "soft_compr",
        "soft_suff",
        "sparsity",
        "gini_index",
        "mass_acc",
        "sens",
    ]

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 3))

    # Plot the first metric
    for i, metric in enumerate(metrics):
        sns.boxplot(
            x="explainer",
            y="score",
            hue="gender",
            data=results,
            ax=axes[i],
        )
        axes[i].set_title(metric)

    for ax in axes:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")

    plt.show()


def collect_ferret_scores(evals):
    """
    Returns dict with explainers as keys and a dict (metric, scores list) as value,
    given the list of evaluations collected in main.py
    """
    dict_template = {
        "aopc_compr": [],
        "aopc_suff": [],
        "soft_compr": [],
        "soft_suff": [],
        "taucorr_loo": [],
        "gini_index": [],
        "sparsity": [],
        "mass_acc": [],
        "sens": [],
    }
    results = {
        "Partition SHAP": deepcopy(dict_template),
        "LIME": deepcopy(dict_template),
        "Gradient": deepcopy(dict_template),
        "Gradient (x Input)": deepcopy(dict_template),
        "Integrated Gradient": deepcopy(dict_template),
        "Integrated Gradient (x Input)": deepcopy(dict_template),
    }
    for eval in evals:
        for metric_eval in eval:
            scores = metric_eval.evaluation_scores
            for score in scores:
                results[metric_eval.explanation.explainer][score.name].append(
                    score.score
                )
    return results


def df_from_ferret_scores(evals, lengths):
    """
    Builds a pandas dataframe given the male and female evaluations
    """
    scores = collect_ferret_scores(evals)

    explainers = [
        "Partition SHAP",
        "LIME",
        "Gradient",
        "Integrated Gradient",
        "Gradient (x Input)",
        "Integrated Gradient (x Input)",
    ]
    metrics = [
        "aopc_compr",
        "aopc_suff",
        "taucorr_loo",
        "soft_compr",
        "soft_suff",
        "gini_index",
        "sparsity",
        "sens",
    ]

    # create df
    results_df = pd.DataFrame(columns=["explainer", "metric", "score", "length"])
    for i, (explainer, metrics) in enumerate(scores.items()):
        for metric, scorelist in metrics.items():
            for score in scorelist:
                results_df.loc[len(results_df)] = [
                    explainer,
                    metric,
                    score,
                    -1,
                ]
    return results_df


def add_special_all_special_tokens(tokenizer):
    """
    special_tokens_dict = {"cls_token": "<CLS>"}

    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print("We have added", num_added_toks, "tokens")
    # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
    model.resize_token_embeddings(len(tokenizer))

    assert tokenizer.cls_token == "<CLS>"

    """
    original_len: int = len(tokenizer)
    num_added_toks: dict = {}
    if tokenizer.bos_token is None:
        num_added_toks["bos_token"] = "<bos>"
    if tokenizer.cls_token is None:
        num_added_toks["cls_token"] = "<cls>"
    if tokenizer.sep_token is None:
        num_added_toks["sep_token"] = "<s>"
    if tokenizer.mask_token is None:
        num_added_toks["mask_token"] = "<mask>"
    # num_added_toks = {"bos_token": "<bos>", "cls_token": "<cls>", "sep_token": "<s>", "mask_token": "<mask>"}
    # special_tokens_dict = {'additional_special_tokens': new_special_tokens + tokenizer.all_special_tokens}
    num_new_tokens: int = tokenizer.add_special_tokens(num_added_toks)
    assert tokenizer.bos_token is not None
    assert tokenizer.cls_token is not None
    assert tokenizer.sep_token is not None
    assert tokenizer.mask_token is not None
    err_msg = f"Error, not equal: {len(tokenizer)=}, {original_len + num_new_tokens=}"
    assert len(tokenizer) == original_len + num_new_tokens, err_msg
