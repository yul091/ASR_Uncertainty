import numpy as np
from collections import Counter
from typing import Iterable, Union


def entropy(x: np.ndarray):
    return np.sum(-x * np.log(np.clip(x, 1e-8, 1)), axis=-1)

def mean_entropy(sampled_probabilities: np.ndarray):
    return entropy(np.mean(sampled_probabilities, axis=1))


def sampled_max_prob(sampled_probabilities: np.ndarray):
    """Computes the max probability for a set of samples.
    Args:
        sampled_probabilities: A numpy array of K forward passes, where each pass contains an array of batch size B X length T X class C.
    Returns:
        max_prob: A numpy array of batch size B X length T.

    def sampled_max_prob(sampled_probabilities):
        mean_probabilities = np.mean(sampled_probabilities, axis=1)
        top_probabilities = np.max(mean_probabilities, axis=-1)
        return 1 - top_probabilities
    """
    if not isinstance(sampled_probabilities, np.ndarray):
        sampled_probabilities = np.array(sampled_probabilities)

    # Compute the mean probability over the K forward passes.
    max_prob = np.max(np.mean(sampled_probabilities, axis=0), axis=-1)  # K X B X T X C -> B X T X C -> B X T
    return np.mean(1 - max_prob, axis=-1)  # B


def probability_variance(sampled_probabilities: np.ndarray):
    """Computes the probability variance for a set of samples.
    Args:
        sampled_probabilities: A numpy array of K forward passes, where each pass contains an array of batch size B X length T X class C.
    Returns:
        variance: A numpy array of batch size B X length T.

    def probability_variance(sampled_probabilities):
        mean_probabilities = np.expand_dims(mean_probabilities, axis=1)
        return ((sampled_probabilities - mean_probabilities) ** 2).mean(1).sum(-1)
    """
    if not isinstance(sampled_probabilities, np.ndarray):
        sampled_probabilities = np.array(sampled_probabilities)

    # Compute the mean probability over the K forward passes.
    mean_probabilities = np.expand_dims(
        np.mean(sampled_probabilities, axis=0), axis=0
    )  # K X B X T X C -> 1 X B X T X C
    variance = np.mean(np.power(sampled_probabilities - mean_probabilities, 2), axis=0)  # B X T X C
    variance = np.mean(np.sum(variance, axis=-1), axis=-1)  # B X T -> B
    return variance  # B


def bald(sampled_probabilities: np.ndarray):
    """Computes the BALD score for a set of samples.
    Args:
        sampled_probabilities: A numpy array of K forward passes, where each pass contains an array of batch size B X length T X class C.
    Returns:
        bald: A numpy array of batch size B X length T.
    """
    if not isinstance(sampled_probabilities, np.ndarray):
        sampled_probabilities = np.array(sampled_probabilities)

    # Compute the mean probability over the K forward passes.
    predictive_entropy = entropy(np.mean(sampled_probabilities, axis=0))  # K X B X T X C -> B X T X C -> B X T
    expected_entropy = np.mean(entropy(sampled_probabilities), axis=0)  # K X B X T X C -> K X B X T -> B X T
    return np.mean(predictive_entropy - expected_entropy, axis=-1)  # B


def find_most_common(row: Iterable[str], mode: Union["elem", "count"]):
    """
    Given iterable of words, return either most common element or its count
    """
    if mode == "elem":
        return Counter(row).most_common(1)[0][0]
    elif mode == "count":
        return Counter(row).most_common(1)[0][1]


def ue_variation_ratio(answers):
    answers = [np.array(e, dtype=object) for e in answers]
    answers = np.stack(answers, -1)

    scores = 1.0 - np.array([find_most_common(ans, "count") / answers.shape[1] for ans in answers])
    return scores