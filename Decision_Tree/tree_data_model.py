from typing import Any, Union, Callable, Optional

import numpy as np


class Dataset(object):
    def __init__(self, feature_data: np.ndarray = None, labels: np.ndarray = None):
        self.feature_data: Optional[np.ndarray] = feature_data
        self.labels: Optional[np.ndarray] = labels

        self.feature_names: Optional[np.ndarray] = None
        self.label_names: Optional[np.ndarray] = None

    def get_feature(self, feature_idx: int) -> str:
        if self.feature_names is not None:
            feature_name = self.feature_names[feature_idx]
            return feature_name
        else:
            raise NotImplementedError("Feature names are not defined in this dataset")

    def get_label(self, label_idx: int) -> str:
        if self.label_names is not None:
            label = self.label_names[label_idx]
            return label
        else:
            return label_idx

    @classmethod
    def from_array(cls,
        dataset: np.ndarray,
        feature_labels: Optional[np.ndarray] = None,
        class_labels: Optional[np.ndarray] = None):

        new_dataset = Dataset()

        new_dataset.feature_data = dataset[:, :-1]
        new_dataset.labels = dataset[:, -1]

        if feature_labels is not None:
            new_dataset.feature_names = feature_labels
        if feature_labels is not None:
            new_dataset.label_names = class_labels

        return new_dataset

class Leaf(object):
    def __init__(self, leaf_value: Any):
        self.leaf_value = leaf_value

class SplitCondition(object):
    def __init__(self, feature: int, split_value: Union[int, float, bool, str], operator: Optional[Callable] = None):
        self.feature = feature
        self.operator = operator
        self.split_value = split_value

    @property
    def operator_string(self):
        if self.operator == self.split_value.__eq__:
            return "="
        elif self.operator == self.split_value.__le__:
            return "<="
        elif self.operator == self.split_value.__ge__:
            return ">="
        elif self.operator == self.split_value.__lt__:
            return "<"
        elif self.operator == self.split_value.__gt__:
            return ">"

    def set_operator(self, operator_as_string: str):
        if operator_as_string == "==":
            self.operator = self.split_value.__eq__
        elif operator_as_string == "<=":
            self.operator = self.split_value.__le__
        elif operator_as_string == "<":
            self.operator = self.split_value.__lt__
        elif operator_as_string == ">=":
            self.operator = self.split_value.__ge__
        elif operator_as_string == ">":
            self.operator = self.split_value.__gt__
        else:
            raise NotImplementedError("This operator is not implemented")

class Node(object):
    def __init__(self, split_condition: SplitCondition, left: Union["Node", Leaf], right: Union["Node", Leaf]):
        self.split_condition = split_condition
        self.left: Union["Node", Leaf] = left
        self.right: Union["Node", Leaf] = right

class Split(object):
    def __init__(self, split_condition: SplitCondition, information_gain: float):
        self.split_condition = split_condition
        self.information_gain = information_gain
