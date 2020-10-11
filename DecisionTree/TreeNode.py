from collections import Counter

class TreeNode:
    def __init__(self, x, y, gini=1):
        self.x = x
        self.y = y
        self.gini = gini
        self.parent = None
        self.left_child = None
        self.right_child = None
        self.feature = None
        self.value = None
        self.leaf = self.check_leaf_node()
        if self.leaf:
            occurence_count = Counter(self.y)
            self.target = occurence_count.most_common(1)[0][0]

    def set_left_child(self, child):
        self.left_child = child

    def set_right_child(self, child):
        self.right_child = child

    def set_parent(self, parent):
        self.parent = parent

    def set_split(self, feature, value):
        self.feature = feature
        self.value = value

    def check_leaf_node(self):
        if self.gini == 0:
            return True
        else:
            return False

    def get_left_child(self):
        return self.left_child

    def get_right_child(self):
        return self.right_child

    def set_leaf(self):
        self.leaf = True
        occurence_count = Counter(self.y)
        self.target = occurence_count.most_common(1)[0][0]