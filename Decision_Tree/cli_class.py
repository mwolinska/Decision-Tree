import distutils

from cleo import Command

from Decision_Tree.dataset_processing import prepare_datasets_from_csv
from Decision_Tree.decision_tree_classifier import main_create_decision_tree, main_test_tree


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
        visual = bool(distutils.util.strtobool(self.option('draw-tree')))

        training_set, test_set, validation_set = prepare_datasets_from_csv(filename)
        decision_tree = main_create_decision_tree(training_set, validation_set, prune, visual)
        main_test_tree(decision_tree, test_set)
        # self.line("On unseen data the decision tree has a score of {}".format(decision_tree_score))
