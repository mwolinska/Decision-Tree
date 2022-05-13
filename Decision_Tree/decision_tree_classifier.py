from typing import Tuple, Union

import numpy as np

from Decision_Tree.tree_data_model import Leaf, Node, SplitCondition, Split

class DecisionTreeClassifier:
    def __init__(self):
        self.tree = None

    @classmethod
    def from_dataset(cls, dataset: np.ndarray):
        decision_tree = cls()
        tree = decision_tree.build_tree(dataset)
        decision_tree.tree = tree
        return decision_tree

    def build_tree(self, dataset: np.ndarray) -> Union[Leaf, Node]:
        is_pure = self.is_pure(dataset)
        if is_pure:
            leaf = Leaf(leaf_value=dataset[0, -1])
            return leaf
        else:
            split_condition = self.find_split_condition(dataset)
            dataset_left, dataset_right = self.split_dataset(dataset, split_condition)
            my_node = Node(
                left=self.build_tree(dataset_left),
                right=self.build_tree(dataset_right),
                split_condition=split_condition,
            )
            return my_node

    def split_dataset(self, dataset: np.ndarray, split_condition: SplitCondition) -> Tuple[np.ndarray, np.ndarray]:
        feature_column = split_condition.feature

        condition_mask = split_condition.operator(dataset[:, feature_column])
        dataset_left = dataset[condition_mask] # relationship == True branch
        dataset_right = dataset[~condition_mask] # relationship == False branch

        return dataset_left, dataset_right

    @staticmethod
    def is_pure(dataset: np.ndarray) -> bool:
        dataset_labels = dataset[:, -1]
        unique_labels = np.unique(dataset_labels)

        if len(unique_labels) <= 1:
            return True
        else:
            return False

    def find_split_condition(self, dataset: np.ndarray) -> SplitCondition:
        information_gain = - np.inf
        split_condition = None

        for feature in range(dataset.shape[1] - 1):
            split = self.get_dataset_split(dataset, feature)

            if split.information_gain > information_gain:
                information_gain = split.information_gain
                split_condition = split.split_condition
        return split_condition

    def get_dataset_split(self, dataset: np.ndarray, feature) -> Split: # returns split on dataset on a given feature

        information_gain = -np.inf
        best_feature_split = None

        unique_features = np.unique(dataset[:, feature])
        split_values = self.get_split_values(unique_features)

        for split_value in split_values:
            feature_split = self.get_feature_split(dataset, feature, split_value)

            if feature_split.information_gain > information_gain:
                information_gain = feature_split.information_gain
                best_feature_split = feature_split

        return best_feature_split

    def get_split_values(self, unique_features: np.ndarray) -> np.ndarray:
        if len(unique_features) == 1:
            return unique_features
        else:
            upper_bound, lower_bound = unique_features[1:], unique_features[:-1]
            array_of_differences = upper_bound - lower_bound
            split_values = (array_of_differences / 2) + lower_bound
            return split_values

    def get_feature_split(self, dataset: np.ndarray, feature: int, split_value: float) -> Split:
        feature_split_condition = SplitCondition(
            feature=feature,
            split_value=split_value,
        )
        operator = "<="
        feature_split_condition.set_operator(operator)

        feature_information_gain = self.information_gain(dataset, feature, split_value)

        feature_split = Split(split_condition=feature_split_condition,
                              information_gain=feature_information_gain)
        return feature_split

    def information_gain(self, dataset: np.ndarray, feature: int, split_value: float) -> float:
        # get unique class labels
        dataset_entropy = self.entropy(dataset)

        # create sub-dataset
        dataset_split_value_mask = dataset[:, feature] <= split_value
        dataset_cut = dataset[dataset_split_value_mask]
        dataset_cut_2 = dataset[~dataset_split_value_mask]

        feature_entropy_at_split = self.entropy(dataset_cut)
        feature_entropy_at_split_2 = self.entropy(dataset_cut_2)

        information_gain = dataset_entropy - (len(dataset_cut)/len(dataset)*feature_entropy_at_split + len(dataset_cut_2)/len(dataset)*feature_entropy_at_split_2)

        return information_gain

    def entropy(self, dataset: np.ndarray) -> float:
        unique_labels = np.unique(dataset[:, -1])
        entropy = 0
        total_samples = dataset.shape[0]
        for label in unique_labels:
            n_instances_of_label = np.count_nonzero(dataset[:, -1] == label)
            ratio_of_instances = n_instances_of_label / total_samples
            entropy += (- ratio_of_instances * np.log2(ratio_of_instances))

        return entropy

    def predict(self, sample_points: np.ndarray) -> List[float]:
        predictions = []
        tuple_of_sample_size = sample_points.shape
        if len(tuple_of_sample_size) == 1:
            predictions.append(self._recursive_predict(sample_points, self.tree))
        else:
            for point in sample_points:
                predictions.append(self._recursive_predict(point, self.tree))
        return predictions

    def _recursive_predict(self, sample_point: np.ndarray, tree: Union[Node, Leaf]) -> Union[float, Node]:
        if isinstance(tree, Leaf):
            return tree.leaf_value
        else:
            feature_value_from_point = sample_point[tree.split_condition.feature]

            if tree.split_condition.operator(feature_value_from_point):
                next_node = tree.left
                return self._recursive_predict(sample_point, next_node)
            else:
                next_node = tree.right
                return self._recursive_predict(sample_point, next_node)

    def evaluate_tree(self, labeled_test_set: np.ndarray):
        data_points, true_labels = labeled_test_set[:, :-1], labeled_test_set[:, -1]
        dataset_size = len(true_labels)
        correct_prediction_count = 0

        predicted_labels = self.predict(data_points)

        for i in range(dataset_size):
            if true_labels[i] == predicted_labels[i]:
                correct_prediction_count += 1

        success_ratio = correct_prediction_count / dataset_size

        return success_ratio