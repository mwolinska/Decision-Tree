from cleo import Application

from Decision_Tree.cli_class import CreateDecisionTree, LoadDecisionTree


def main():
    application = Application()
    application.add(CreateDecisionTree())
    application.add(LoadDecisionTree())
    application.run()
