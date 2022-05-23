# !/usr/bin/env python


from cleo import Application

from Decision_Tree.cli_class import CreateDecisionTree

application = Application()
application.add(CreateDecisionTree())

if __name__ == '__main__':
    application.run()
