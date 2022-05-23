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
 An example run using the[iris dataset](https://archive.ics.uci.edu/ml/datasets/iris) 
 would look like this.
Ensure the dataset you are using is saved within the package. 
The cli has optional arguments prune and draw-tree, which are set to True by default.

```bash
 python cli.py run iris.csv 
```

To set either the prune or draw-tree variables to False, use one the following syntaxes:

```bash
 python cli.py run iris.csv -p False -d False
```
Or:

```bash
python cli.py run iris.csv --prune False --draw-tree False
```

Once a run is completed, the decision tree will be saved under "tree_visual.pdf" in the
project directory. If the feature and label names are added to the training dataset, those are included in
the tree visualisation. The tree generated using the run above would look like this:

<img src="./Images/SampleDecisionTree/tree_visual_with_names.png">

If the prune variable is set to True the pruned tree visualisation will be saved under 
"pruned_tree.pdf" in the project directory. For this run it would look like this:

<img src="./Images/SampleDecisionTree/pruned_tree.png">

If the feature names are not included in the dataset the tree will be labeled using
column indices as feature numbers. This image is generated using a different run than those above.

<img src="./Images/SampleDecisionTree/tree_visual_no_names.png" height="500">
