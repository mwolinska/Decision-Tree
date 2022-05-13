from typing import Tuple

import numpy as np


def split_dataset_using_shuffle(dataset: np.ndarray, dataset_ratio_for_training: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ratio_for_training = 0.5 * (1 - dataset_ratio_for_training)
    dataset_size = len(dataset)
    first_split_index = int(dataset_size * dataset_ratio_for_training)
    second_split_index = first_split_index + int(dataset_size * ratio_for_training)

    np.random.shuffle(dataset)

    training_dataset = dataset[:first_split_index]
    test_dataset = dataset[first_split_index: second_split_index]
    validate_dataset = dataset[second_split_index:]

    return training_dataset, test_dataset, validate_dataset
