import sys
import os
import random
sys.path.append(os.path.dirname(os.getcwd())+'\Decision Tree')

from DecisionTree import TreeNode, DecisionTree


class MaximumDecisionTree(DecisionTree):
    def __init__(self, max_depth=2):
        super().__init__()
        self.max_depth = max_depth

    # introduce randomness into splitting a node
    def get_split(self, node):
        if node.leaf:
            return None

        min_gini = 99999
        feature_split = 0
        value_split = 0
        best_x = []
        best_y = []
        size = len(node.x[0])
        subset = random.sample(range(len(node.x[0])), k=size-1)

        for row in node.x:
            for feature in subset:
                children_x, children_y = self.split(node.x, node.y, feature, row[feature])
                weighted_sum = self.weighted_split(children_y)
                if weighted_sum < min_gini:
                    min_gini = weighted_sum
                    feature_split = feature
                    value_split = row[feature]
                    best_x = children_x
                    best_y = children_y

        # print('minimum gini is: {}'.format(min_gini))
        node.set_split(feature_split, value_split)
        left_gini = self.calculate_gini(best_y[0])
        left_child_node = TreeNode(best_x[0], best_y[0], left_gini)
        node.set_left_child(left_child_node)

        right_gini = self.calculate_gini(best_y[1])
        right_child_node = TreeNode(best_x[1], best_y[1], right_gini)
        node.set_right_child(right_child_node)

        return [left_child_node, right_child_node]

    # set limit depth on growing a tree
    def fit(self, x, y):
        self.set_classes(y)
        root_gini = self.calculate_gini(y)
        self.root = TreeNode(x, y, root_gini)
        depth = 0
        current_level = [self.root]
        while current_level:
            next_level = []
            for node in current_level:
                children = self.get_split(node)

                if children is not None:
                    for child in children:
                        next_level.append(child)

            current_level = next_level
            depth += 1
            if depth > self.max_depth:
                print(depth)
                break





