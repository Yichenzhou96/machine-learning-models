import numpy as np
from decisionTree.TreeNode import TreeNode


class DecisionTree:
    def __init__(self):
        self.classes = []
        self.root = None

    def set_classes(self, y):
        self.classes = set(y)

    def calculate_gini(self, target):
        size = len(target)
        p = [len(list(filter(lambda x: x == k, target))) / size for k in self.classes]
        gini = 1 - sum(list(map(lambda x: x**2, p)))
        return gini

    def weighted_split(self, children_target):
        weighted_sum = 0
        total_size = 0
        total_gini = []
        for child in children_target:
            size = len(child)
            if size == 0:
                continue

            gini = self.calculate_gini(child)
            weighted_sum += gini * size
            total_size += size
            total_gini.append(gini)

        if total_size == 0:
            return 0

        return weighted_sum / total_size

    @staticmethod
    def split(x, y, feature, value):
        left_child, right_child = [], []
        left_child_y, right_child_y = [], []
        for row, target in zip(x, y):
            if row[feature] < value:
                left_child.append(row)
                left_child_y.append(target)
            else:
                right_child.append(row)
                right_child_y.append(target)

        left_child = np.asarray(left_child)
        right_child = np.asarray(right_child)
        left_child_y = np.asarray(left_child_y)
        right_child_y = np.asarray(right_child_y)

        return [left_child, right_child], [left_child_y, right_child_y]

    def get_split(self, node):
        if node.leaf:
            return None

        min_gini = 99999
        feature_split = 0
        value_split = 0
        best_x = []
        best_y = []
        for row in node.x:
            for feature in range(len(row)):
                children_x, children_y = self.split(node.x, node.y, feature, row[feature])
                weighted_sum = self.weighted_split(children_y)
                if weighted_sum < min_gini:
                    min_gini = weighted_sum
                    feature_split = feature
                    value_split = row[feature]
                    best_x = children_x
                    best_y = children_y

        node.set_split(feature_split, value_split)
        left_gini = self.calculate_gini(best_y[0])
        left_child_node = TreeNode(best_x[0], best_y[0], left_gini)
        node.set_left_child(left_child_node)

        right_gini = self.calculate_gini(best_y[1])
        right_child_node = TreeNode(best_x[1], best_y[1], right_gini)
        node.set_right_child(right_child_node)

        return [left_child_node, right_child_node]

    @staticmethod
    def check_leaf_node(node):
        if node.gini == 0:
            return True

        return False

    def fit(self, x, y):
        self.set_classes(y)
        root_gini = self.calculate_gini(y)
        self.root = TreeNode(x, y, root_gini)
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            children = self.get_split(node)

            if children is not None:
                for child in children:
                    queue.append(child)

    def predict(self, test):
        predictions = []
        for x in test:
            queue = [self.root]
            while queue:
                head = queue.pop(0)
                if head.leaf:
                    predictions.append(head.target)
                    break
                else:
                    if x[head.feature] < head.value:
                        left_child = head.get_left_child()
                        queue.append(left_child)
                    else:
                        right_child = head.get_right_child()
                        queue.append(right_child)

        return predictions

    def print_tree(self):
        current_level = [self.root]

        while current_level:
            print(' '.join("< " + str(node.value) if node.value is not None else "class: " + str(node.target) for node in current_level))
            next_level = []
            for node in current_level:
                if node.get_left_child() is not None:
                    next_level.append(node.get_left_child())

                if node.get_right_child() is not None:
                    next_level.append(node.get_right_child())

            current_level = next_level














