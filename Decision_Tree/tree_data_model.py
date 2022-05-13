from typing import Any, Union, Callable, Optional

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

class Node(object):
    def __init__(self, split_condition: SplitCondition, left: Union["Node", Leaf], right: Union["Node", Leaf]):
        self.split_condition = split_condition
        self.left: Union["Node", Leaf] = left
        self.right: Union["Node", Leaf] = right

class Split(object):
    def __init__(self, split_condition: SplitCondition, information_gain: float):
        self.split_condition = split_condition
        self.information_gain = information_gain
