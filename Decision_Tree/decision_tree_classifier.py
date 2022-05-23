import copy
import dill as pickle
from typing import Tuple, Union, List, Optional, Any

import graphviz
import numpy as np

from Decision_Tree.dataset_processing import prepare_datasets_from_csv
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

    @staticmethod
    def get_split_values(unique_features: np.ndarray) -> np.ndarray:
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

    @staticmethod
    def entropy(training_dataset: Dataset) -> float:
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

    def draw(self, dataset: Dataset, file_name: str = "tree_visual"):
        self.assign_nodes_to_visual(self.tree, '', dataset)
        self.tree_visual.view(filename=file_name)

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

    def prune(self, training_set: Dataset, validation_set: Dataset):
        pruned_tree = copy.deepcopy(self)
        best_success_ratio = self.evaluate_tree(validation_set)

        for depth in range(self.max_depth - 1, 1, -1):
            paths_at_depth = self.paths_to_nodes_at_depth(pruned_tree.tree, depth)
            for path in paths_at_depth:
                pruning_test_tree = copy.deepcopy(pruned_tree)
                tree_copy_for_replacement = copy.deepcopy(pruned_tree.tree)
                leaf_value = pruned_tree.find_value_to_replace_node(training_set, path)
                pruning_test_tree.tree = pruned_tree.replace_node_with_leaf(tree_copy_for_replacement, path, leaf_value)
                pruning_success_score = pruning_test_tree.evaluate_tree(validation_set)
                if pruning_success_score >= best_success_ratio:
                    best_success_ratio = pruning_success_score
                    pruned_tree = pruning_test_tree

        return pruned_tree

    def paths_to_nodes_at_depth(self, tree: Node, depth: int, path: Optional[List] = None):
        if path is None:
            path = []

        if tree.depth == depth and isinstance(tree, Node):
            return [path]
        elif isinstance(tree, Leaf):
            return None
        else:
            left_branch = self.paths_to_nodes_at_depth(tree.left, depth, path + ["L"])
            right_branch = self.paths_to_nodes_at_depth(tree.right, depth, path + ["R"])
            all_elements = []
            if left_branch is not None:
                all_elements += left_branch
            if right_branch is not None:
                all_elements += right_branch

            return all_elements

    @staticmethod
    def replace_node_with_leaf(tree: Node, path: List[str], leaf_value: Any) -> Node:
        node = tree
        path_until_last_step = path[:-1]
        last_step = path[-1]
        for el in path_until_last_step:
            if el == "L":
                node = node.left
            elif el == "R":
                node = node.right

        if last_step == "L":
            node.left = Leaf(leaf_value)
        elif last_step == "R":
            node.right = Leaf(leaf_value)

        return tree

    def node_from_path(self, path: List[str]):
        node = self.tree
        for el in path:
            if el == "L":
                node = node.left
            elif el == "R":
                node = node.right
        return node

    def find_value_to_replace_node(self, dataset: Dataset, path_to_node: List[str]):
        labels_count_at_node = self.count_labels_on_split_condition(dataset, path_to_node)

        index_with_max_count = labels_count_at_node[1].argmax()
        label_with_max_count = labels_count_at_node[0][index_with_max_count]
        return label_with_max_count

    def count_labels_on_split_condition(self, dataset: Dataset, path_to_node: List[str]):
        dataset_at_node = self.dataset_at_node(dataset, path_to_node)
        labels_count = np.unique(dataset_at_node.labels, return_counts=True)

        return labels_count

    def dataset_at_node(self, dataset: Dataset, path: List[str]) -> Dataset:
        node = self.tree

        for el in path:
            left_dataset, right_dataset = self.split_dataset(dataset, node.split_condition)
            if el == "L":
                node = node.left
                dataset = left_dataset
            elif el == "R":
                node = node.right
                dataset = right_dataset

        return dataset

    def save(self, file_name: str = "decision_tree"):
        with open(file_name + ".pickle", "wb") as target_file:
            pickle.dump(self, target_file)

def load_decision_tree(file_name: str):
    with open(file_name + ".pickle", "rb") as target_file:
        tree_from_file = pickle.load(target_file)
    return tree_from_file

def main_create_decision_tree(training_set: Dataset, validation_set: Dataset, prune: bool = True, visualise_tree: bool = True):
    tree = DecisionTreeClassifier.from_dataset(training_set)
    unpruned_score = tree.evaluate_tree(validation_set)

    if prune:
        pruned_tree = tree.prune(training_set, validation_set)
        pruned_score = pruned_tree.evaluate_tree(validation_set)
        print(f"Unpruned score: {unpruned_score}, pruned score: {pruned_score}. performed on validation dataset")
        if visualise_tree:
            pruned_tree.draw(training_set, file_name="pruned_tree")
            tree.draw(training_set)

        return pruned_tree

    elif visualise_tree:
        tree.draw(training_set)

    print(f"Unpruned score: {unpruned_score}, pruning not performed.")
    return tree

def main_test_tree(decision_tree: DecisionTreeClassifier, test_set: Dataset):
    test_set_score = decision_tree.evaluate_tree(test_set)
    print(f"Decision tree score on unseen data: {test_set_score}")
    return test_set_score
