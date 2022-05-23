import distutils

from cleo import Command

from Decision_Tree.dataset_processing import prepare_datasets_from_csv, create_array_from_csv, list_to_csv
from Decision_Tree.decision_tree_classifier import main_create_decision_tree, main_test_tree, load_decision_tree


class CreateDecisionTree(Command):
    """
    Enables user to use decision tree package.

    run
        {filename : Which file would you like to use to create a decision tree}
        {output? : Specify the location of the output.}
        {--p|prune=?True : If set to True, the decision tree will be pruned. Set to True by default}
        {--d|draw-tree=?True : If set to True, the decision tree will be saved as a visualisation. Set to True by default}

    """

    def handle(self):
        filename = self.argument('filename')
        output = self.argument('output')
        prune = bool(distutils.util.strtobool(self.option('prune')))
        draw_tree = bool(distutils.util.strtobool(self.option('draw-tree')))

        training_set, test_set, validation_set = prepare_datasets_from_csv(filename)
        decision_tree = main_create_decision_tree(training_set, validation_set, prune, draw_tree)
        main_test_tree(decision_tree, test_set)
        if output is not None:
            decision_tree.save(output)
        # self.line("On unseen data the decision tree has a score of {}".format(decision_tree_score))

class LoadDecisionTree(Command):
    """
    Enables user to load an existing decision tree.

    load
        {filename : Which file would you like to use to create a decision tree}
        {predict : CSV file containing points requiring prediction}
        {output : Name of CSV file containing output}

    """
    def handle(self):
        filename = self.argument('filename')
        file_for_prediction = self.argument('predict')
        output = self.argument("output")

        decision_tree = load_decision_tree(filename)

        array_for_prediction = create_array_from_csv(file_for_prediction, numerical_data_only=True)
        predictions_list = decision_tree.predict(array_for_prediction)
        list_to_csv(predictions_list, output)
