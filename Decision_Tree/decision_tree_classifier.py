import copy
from typing import Tuple, Union, List, Optional, Any

import graphviz
import numpy as np

from Decision_Tree.dataset_processing import split_dataset_using_shuffle
from Decision_Tree.tree_data_model import Leaf, Node, SplitCondition, Split, Dataset


class DecisionTreeClassifier:
    def __init__(self):
        self.tree = None
        self.tree_visual = graphviz.Digraph()
        self.max_depth = 0

    @classmethod
    def from_dataset(cls, training_dataset: Dataset): # in many libraries this is called fit. because fit decision tree to dataset
        decision_tree = cls()
        tree = decision_tree.build_tree(training_dataset)
        decision_tree.tree = tree
        return decision_tree

    @classmethod
    def from_tree(cls, tree: Node):
        decision_tree = cls()
        decision_tree.tree = tree
        return decision_tree

    def build_tree(self, training_dataset: Dataset, depth: int = 0) -> Union[Leaf, Node]:
        is_pure = self.is_pure(training_dataset)
        if is_pure:
            leaf = Leaf(leaf_value=training_dataset.labels[0], depth=depth)
            if depth > self.max_depth:
                self.max_depth = depth
            return leaf
        else:
            split_condition = self.find_split_condition(training_dataset)
            dataset_left, dataset_right = self.split_dataset(training_dataset, split_condition)
            my_node = Node(
                left=self.build_tree(dataset_left, depth + 1),
                right=self.build_tree(dataset_right, depth + 1),
                split_condition=split_condition,
                depth=depth,
            )
            return my_node

    def split_dataset(self, training_dataset: Dataset, split_condition: SplitCondition) -> Tuple[Dataset, Dataset]:
        feature_column = split_condition.feature

        condition_mask = split_condition.operator(training_dataset.feature_data[:, feature_column])
        dataset_left = Dataset(feature_data=training_dataset.feature_data[condition_mask],
                               labels=training_dataset.labels[condition_mask]) # relationship == True branch
        dataset_right = Dataset(feature_data=training_dataset.feature_data[~condition_mask],
                                labels=training_dataset.labels[~condition_mask]) # relationship == False branch

        return dataset_left, dataset_right

    @staticmethod
    def is_pure(training_dataset: Dataset) -> bool:
        unique_labels = np.unique(training_dataset.labels)

        if len(unique_labels) <= 1:
            return True
        else:
            return False

    def find_split_condition(self, training_dataset: Dataset) -> SplitCondition:
        information_gain = - np.inf
        split_condition = None

        for feature in range(training_dataset.feature_data.shape[1] - 1):
            split = self.get_dataset_split(training_dataset, feature)

            if split.information_gain > information_gain:
                information_gain = split.information_gain
                split_condition = split.split_condition
        return split_condition

    def get_dataset_split(self, training_dataset: Dataset, feature: int) -> Split: # returns split on dataset on a given feature

        information_gain = -np.inf
        best_feature_split = None

        unique_features = np.unique(training_dataset.feature_data[:, feature])
        split_values = self.get_split_values(unique_features)

        for split_value in split_values:
            feature_split = self.get_feature_split(training_dataset, feature, split_value)

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

    def get_feature_split(self, training_dataset: Dataset, feature: int, split_value: float) -> Split:
        feature_split_condition = SplitCondition(
            feature=feature,
            split_value=split_value,
        )
        operator = "<="
        feature_split_condition.set_operator(operator)

        feature_information_gain = self.information_gain(training_dataset, feature, split_value)

        feature_split = Split(split_condition=feature_split_condition,
                              information_gain=feature_information_gain)
        return feature_split

    def information_gain(self, training_dataset: Dataset, feature: int, split_value: float) -> float:
        # get unique class labels
        dataset_entropy = self.entropy(training_dataset)

        # create sub-dataset
        dataset_split_value_mask = training_dataset.feature_data[:, feature] <= split_value
        dataset_cut = Dataset(feature_data=training_dataset.feature_data[dataset_split_value_mask],
                              labels=training_dataset.labels[dataset_split_value_mask])
        dataset_cut_2 = Dataset(feature_data=training_dataset.feature_data[~dataset_split_value_mask],
                                labels=training_dataset.labels[~dataset_split_value_mask])

        feature_entropy_at_split = self.entropy(dataset_cut)
        feature_entropy_at_split_2 = self.entropy(dataset_cut_2)

        information_gain = dataset_entropy - (len(dataset_cut.feature_data) / len(training_dataset.feature_data) * feature_entropy_at_split + len(dataset_cut_2.feature_data) / len(training_dataset.feature_data) * feature_entropy_at_split_2)

        return information_gain

    def entropy(self, training_dataset: Dataset) -> float:
        unique_labels = np.unique(training_dataset.labels)
        entropy = 0
        total_samples = training_dataset.feature_data.shape[0]
        for label in unique_labels:
            n_instances_of_label = np.count_nonzero(training_dataset.labels == label)
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

    def evaluate_tree(self, labeled_test_set: Dataset):
        data_points, true_labels = labeled_test_set.feature_data, labeled_test_set.labels
        dataset_size = len(true_labels)
        correct_prediction_count = 0

        predicted_labels = self.predict(data_points)

        for i in range(dataset_size):
            if true_labels[i] == predicted_labels[i]:
                correct_prediction_count += 1

        success_ratio = correct_prediction_count / dataset_size

        return success_ratio

    def draw(self, dataset: Dataset):
        self.assign_nodes_to_visual(self.tree, '', dataset)
        self.tree_visual.view(filename="tree_visual")

    def assign_nodes_to_visual(self, tree: Union[Leaf, Node], node_name: str, dataset: Dataset):
        if isinstance(tree, Leaf):
            class_label = dataset.get_label(int(tree.leaf_value))
            self.tree_visual.node(node_name, "Class label: " + str(class_label))
        else:
            operator = tree.split_condition.operator_string

            feature_name = dataset.get_feature(tree.split_condition.feature)

            node_label = f"Feature {feature_name} {operator} {tree.split_condition.split_value}"

            self.tree_visual.node(node_name, label=node_label)
            self.tree_visual.edge(node_name, node_name + "-L", label="True")
            self.tree_visual.edge(node_name, node_name + "-R", label="False")
            return self.assign_nodes_to_visual(tree.left, node_name + "-L", dataset), \
                   self.assign_nodes_to_visual(tree.right, node_name + "-R", dataset)


if __name__ == '__main__':
    import numpy as np
    from sklearn.datasets import load_iris
    dataset = load_iris()
    features = dataset["data"]
    labels = dataset["target"]

    feature_names = np.asarray(dataset["feature_names"])
    target_names = np.asarray(dataset["target_names"])

    dataset = np.hstack([features, labels.reshape(-1, 1)])

    training_set, test_set, validation_set = split_dataset_using_shuffle(dataset, dataset_ratio_for_training=0.6)
    training_set.label_names = target_names
    training_set.feature_names = feature_names

    tree = DecisionTreeClassifier.from_dataset(training_set)
    success = tree.evaluate_tree(validation_set)

    tree.draw(training_set)
    print()
