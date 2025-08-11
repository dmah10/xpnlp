from dataclasses import dataclass
from typing import List


from ferret.explainers.explanation import Explanation


@dataclass
class Evaluation:
    """Generic class to represent an Evaluation"""

    name: str
    score: float


@dataclass
class ExplanationEvaluation:
    """Generic class to represent an Evaluation"""

    explanation: Explanation
    evaluation_scores: List[Evaluation]
