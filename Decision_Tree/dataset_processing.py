import csv
from typing import Tuple, List

import numpy as np

from Decision_Tree.tree_data_model import Dataset


def prepare_datasets_from_csv(file_name: str, data_delimiter: str = ",", training_dataset_ratio: float = 0.6):
    data_as_array = create_array_from_csv(file_name, data_delimiter)
    full_dataset = process_array_for_splitting(data_as_array)
    training_dataset, test_dataset, validate_dataset = split_dataset(full_dataset, training_dataset_ratio)
    return training_dataset, test_dataset, validate_dataset

def create_array_from_csv(file_name: str, data_delimiter: str = ",", numerical_data_only: bool = False):
    data_list = []
    with open(file_name) as datafile:
        data_reader = csv.reader(datafile, delimiter=data_delimiter)
        for row in data_reader:
            data_list.append(row)

    if numerical_data_only:
        data_array = np.asarray(data_list, dtype=float)
    else:
        data_array = np.asarray(data_list)

    return data_array

def process_array_for_splitting(dataset_as_array: np.ndarray) -> Dataset:
    feature_names = dataset_as_array[0]
    only_data_array = dataset_as_array[1:, :]
    np.random.shuffle(only_data_array)

    feature_data = only_data_array[:, :-1].astype(float)
    label_data = only_data_array[:, -1]
    label_names = np.unique(label_data)
    label_idx = 0
    for el in label_names:
        label_data[label_data == el] = label_idx
        label_idx += 1
    label_data = label_data.astype(int)

    dataset = Dataset(feature_data=feature_data, labels=label_data, feature_names=feature_names, label_names=label_names)

    return dataset

def split_dataset(dataset: Dataset, dataset_ratio_for_training: float = 0.6) -> Tuple[Dataset, Dataset, Dataset]:
    ratio_for_training = 0.5 * (1 - dataset_ratio_for_training)
    dataset_size = len(dataset.feature_data)
    first_split_index = int(dataset_size * dataset_ratio_for_training)
    second_split_index = first_split_index + int(dataset_size * ratio_for_training)

    training_dataset = Dataset(feature_data=dataset.feature_data[:first_split_index],
                               labels=dataset.labels[:first_split_index],
                               feature_names=dataset.feature_names,
                               label_names=dataset.label_names
                               )
    test_dataset = Dataset(feature_data=dataset.feature_data[first_split_index: second_split_index],
                           labels=dataset.labels[first_split_index: second_split_index],
                           feature_names=dataset.feature_names,
                           label_names=dataset.label_names
                           )
    validate_dataset = Dataset(feature_data=dataset.feature_data[second_split_index:],
                               labels=dataset.labels[second_split_index:],
                               feature_names=dataset.feature_names,
                               label_names=dataset.label_names
                               )
    return training_dataset, test_dataset, validate_dataset

def list_to_csv(a_list: List, file_name: str):
    with open(file_name, "w") as target_file:
        for el in a_list:
            target_file.write(str(el))
            target_file.write("\n")


# def split_dataset_using_choice(dataset:np.ndarray, dataset_ratio_for_training: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     dataset_array_of_indices = np.indices((0, len(dataset)))
#     validation_set_size = int(len(dataset) * dataset_ratio_for_training)
#     mask_for_training_set = np.random.choice(dataset_array_of_indices[1], size=validation_set_size, replace=False)
#
#     training_set = dataset[mask_for_training_set]
#     other_sets = dataset[~mask_for_training_set]
#
#     indices_in_new_dataset = np.indices((0, len(other_sets)))
#     training_set_size = int(0.5 * len(other_sets))
#     mask_for_test_set = np.random.choice(indices_in_new_dataset, size=training_set_size, replace=False)
#
#     test_set = other_sets[mask_for_test_set]
#     validate_set = other_sets[~mask_for_test_set]
#
#     return training_set, test_set, validate_set
