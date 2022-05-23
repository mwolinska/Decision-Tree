# Decision-Tree
## Introduction
This package allows the user to build a decision tree from a previously unseen dataset.
Once the tree is built the user can test the accuracy of the tree, predict the class label
of an unclassified datapoint and create a labeled visualisation of the decision tree. 
An additional feature allows the user to split their dataset into 
training, test and validation sets prior to building the decision tree.

### Getting started with the package
To get started with this package clone this repo:

```bash
git clone https://github.com/mwolinska/Decision-Tree
```
Then enter the correct directory on your machine:
```bash
cd Decision-Tree
```
This package uses [poetry](https://python-poetry.org) dependency manager. 
To install all dependencies run:

```bash
poetry install
```

### Using the package
Ensure the dataset you are using is saved within the package. 
The cli is triggered by using the decision-tree command, which launches the cli script.
The cli has 3 commands:
- run
- load
- help

An example run using the [iris dataset](https://archive.ics.uci.edu/ml/datasets/iris) 
 would look like this.
#### Help command
Default command to view available command.

#### Run command
This function takes a full dataset (in csv format), separates it into training, validation and test sets. 
It then generates a decision tree based on the training data. 
It has optional arguments -prune and -draw-tree (set to True by default).
The decision tree can also be saved in pickle format
by specifying the optional argument output.

To create a decision tree based on the iris.csv dataset and 
save it as "iris_decision_tree.pickle" the following command can be run:

```bash
decision-tree run iris.csv iris_decision_tree.pickle
```

To set either the prune or draw-tree variables to False, use one the following syntaxes:

```bash
decision-tree run iris.csv iris_decision_tree.pickle -p False -d False
```
Or:

```bash
decision-tree run iris.csv iris_decision_tree.pickle --prune False --draw-tree False
```

Once a run is completed, if the draw-tree argument was set to True 
the decision tree will be saved under "tree_visual.pdf" in the
project directory. If the feature and label names are added to the training dataset, those are included in
the tree visualisation. The tree generated using the run above would look like this:

<img src="./Images/SampleDecisionTree/tree_visual_with_names.png">

If the prune variable is set to True the pruned tree visualisation will be saved under 
"pruned_tree.pdf" in the project directory. For this run it would look like this:

<img src="./Images/SampleDecisionTree/pruned_tree.png">

If the feature names are not included in the dataset the tree will be labeled using
column indices as feature numbers. This image is generated using a different run than those above.

<img src="./Images/SampleDecisionTree/tree_visual_no_names.png" height="500">

#### Load command
The load command allows the user to load an existing decision tree (in pickle format)
and generate predictions for a dataset. The user needs to specify the 
filename from which the tree will be loaded,
the csv containing the data points requiring prediction and
the output file where the predicted values should be stored.
An example run would look like this:

```bash
decision-tree load iris_decision_tree.pickle samples_for_prediction.csv predictions.csv
```
