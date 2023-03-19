"""Decision tree definition."""

import numpy as np

def gini(*groups):
    """ Gini impurity for classification problems.
    Args: groups (tuple): tuples containing:
        (ndarray): Group inputs (x).
        (ndarray): Group labels (y).
    Returns:
        (float): Gini impurity index.
    """
    m = np.sum([group[0].shape[0] for group in groups])  # Number of samples

    gini = 0.0

    for group in groups:
        y = group[1]
        group_size = y.shape[0]

        # Count number of observations per class
        _, class_count = np.unique(y, return_counts=True)
        proportions = class_count / group_size
        weight = group_size / m

        gini += (1 - np.sum(proportions ** 2)) * weight

    return gini


def split(x, y, feature_idx, split_value):
    """ Returns two tuples holding two groups resulting from split.
    Args:
        x (ndarray): Input.
        y (ndarray): Labels.
        feature_idx (int): Feature to consider.
        split_value (float): Value used to split.
    Returns:
        (tuple):tuple containing:
            (tuple):tuple containing:
                (ndarray): Inputs of group under split.
                (ndarray): Labels of group under split.
            (tuple):tuple containing:
                (ndarray): Inputs of group over split.
                (ndarray): Labels of group over split.
    """
    bool_mask = x[:, feature_idx] < split_value
    group_1 = (x[bool_mask], y[bool_mask])
    group_2 = (x[bool_mask == 0], y[bool_mask == 0])
    return group_1, group_2


def legal_split(*groups, min_samples_leaf=1):
    """Test if all groups hold enough samples to meet the min_samples_leaf
    requirement """
    for g in groups:
        if g[0].shape[0] < min_samples_leaf:
            return False
    return True


def split_search_feature(x, y, feature_idx, min_samples_leaf):
    """Return best split on dataset given a feature.
    Return error values (np.Nan for floats and None for tuples) if no
    split can be found.
    Args:
        x(ndarray): Inputs.
        y(ndarray): Labels.
        feature_idx(int): Index of feature to consider
        min_samples_leaf(int): Minimum number of samples to be deemed
        a leaf node.
    Returns:
        (tuple):tuple containing:
            (float): gini score.
            (float): value used for splitting.
            (tuple):tuple containing:
                (tuple):tuple containing:
                    (ndarray): Inputs of group under split.
                    (ndarray): Labels of group under split.
                (tuple):tuple containing:
                    (ndarray): Inputs of group over split.
                    (ndarray): Labels of group over split.
    """
    gini_scores = []
    splits = []
    split_values = []
    series = x[:, feature_idx]

    # Greedy search on all input values for relevant feature
    for split_value in series:
        s = split(x, y, feature_idx, split_value)

        # Test if groups hold enough samples, otherwise keep searching
        if legal_split(*s, min_samples_leaf=min_samples_leaf):
            gini_scores.append(gini(*s))
            splits.append(s)
            split_values.append(split_value)

    if len(gini_scores) == 0:
        # Impossible to split
        # This will occur when samples are identical in a given node
        return np.NaN, np.NaN, None

    arg_min = np.argmin(gini_scores)

    return gini_scores[arg_min], split_values[arg_min], splits[arg_min]


def split_search(x, y, min_samples_leaf, feature_search=None):
    """Return best split on dataset.
    Return error values (np.Nan for floats and None for tuples) if no
    split can be found.
    Args:
        x(ndarray): Inputs.
        y(ndarray): Labels.
        feature_search(int): Number of features to use for split search
        min_samples_leaf(int): Minimum number of samples to be deemed
        a leaf node.
    Returns:
        (tuple):tuple containing:
            (int): Index of best feature.
            (float): value used for splitting.
            (tuple):tuple containing:
                (ndarray): Inputs of group under split.
                (ndarray): Labels of group under split.
            (tuple):tuple containing:
                (ndarray): Inputs of group over split.
                (ndarray): Labels of group over split.
    """
    gini_scores = []
    splits = []
    split_values = []

    # Flag to handle cases where no legal splits can be found
    split_flag = False

    if feature_search is None:
        # Default to all features
        feature_indices = np.arange(x.shape[1])
    else:
        if feature_search > x.shape[1]:
            raise Exception('Tried searching more features than '
                            'available features in dataset.')

        # Randomly choose feature_search features to look up
        feature_indices = np.random.choice(x.shape[1],
                                           feature_search,
                                           replace=False)

    # Search over features
    for feature_idx in feature_indices:
        g, s_value, s = split_search_feature(x, y,
                                             feature_idx, min_samples_leaf)
        gini_scores.append(g)
        split_values.append(s_value)
        splits.append(s)

        if g is not np.NaN:
            # At least one legal split
            split_flag = True

    if not split_flag:
        # Impossible to split
        # This will occur when samples are identical in a given node
        return np.NaN, np.NaN, None, None

    arg_min = np.nanargmin(gini_scores)

    group_1, group_2 = splits[arg_min]

    return feature_indices[arg_min], split_values[arg_min], group_1, group_2


class Node:
    def __init__(self, depth=0):
        """Node definition.
        Args:
            depth(int): Depth of this node (root node depth should be 0).
        """
        self._feature_idx = None  # Feature index to use for splitting
        self._split_value = None
        self._leaf = False
        self._label = None
        self._left_child = None
        self._right_child = None
        self._depth = depth

    def train(self, x, y, feature_search=None,
              max_depth=8, min_samples_split=2, min_samples_leaf=1):
        """Training procedure for node.
        Args:
            x(ndarray): Inputs.
            y(ndarray): Labels.
            feature_search(int): Number of features to search for splitting.
            max_depth(int): Max depth of tree.
            min_samples_split(int): Number of samples in a node to allow
            split search.
            min_samples_leaf(int): Number of samples to be deemed a leaf node.
        """
        if self._depth < max_depth and x.shape[0] > min_samples_split:

            # Retrieve best split coordinates based on gini impurity
            # and two groups
            self._feature_idx, self._split_value, group_1, group_2 = \
                split_search(x, y, min_samples_leaf, feature_search)

            if self._feature_idx is not np.NaN:
                # Recursively split and train child nodes
                self._left_child = Node(self._depth + 1)
                self._right_child = Node(self._depth + 1)
                self._left_child.train(*group_1, feature_search, max_depth,
                                       min_samples_split,
                                       min_samples_leaf)
                self._right_child.train(*group_2, feature_search, max_depth,
                                        min_samples_split,
                                        min_samples_leaf)
            else:
                # Impossible to split. Convert to leaf node
                # This will occur when observations are
                # identical in a given node
                self._sprout(y)
        else:
            # End condition met. Convert to leaf node
            self._sprout(y)

    def _sprout(self, y):
        """Flag node as a leaf node."""
        self._leaf = True

        # Count classes in current node to determine class
        _classes, counts = np.unique(y, return_counts=True)
        self._label = _classes[np.argmax(counts)]

    def eval(self, x, y):
        """Return number of correct predictions over a dataset."""
        if self._leaf:
            return np.sum(y == self._label)
        else:
            group_1, group_2 = split(x, y,
                                     self._feature_idx, self._split_value)
            return self._left_child.eval(*group_1) \
                   + self._right_child.eval(*group_2)

    def count(self):
        """Recursively count nodes."""
        if self._leaf:
            return 1
        return 1 + self._left_child.count() + self._right_child.count()

    def predict(self, x):
        """Recursively predict class for a single individual.
        Args:
            x(ndarray): A single individual.
        Returns:
            (int): Class index.
        """
        if self._leaf:
            return self._label
        else:
            if x[self._feature_idx] < self._split_value:
                return self._left_child.predict(x)
            else:
                return self._right_child.predict(x)


class Tree:
    def __init__(self, max_depth=5,
                 min_samples_split=2, min_samples_leaf=1, bootstrap=False):
        """Decision tree for classification.
        Args:
            max_depth(int): Max depth of tree.
            min_samples_split(int): Number of samples in a node to allow
            split search.
            min_samples_leaf(int): Number of samples to be deemed a leaf node.
            bootstrap(boolean): Resample dataset with replacement
        """
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._bootstrap = bootstrap

        # Root node
        self._root = Node()

    def train(self, x, y, feature_search=None):
        """Training routine for tree.
        Args:
            x(ndarray): Inputs.
            y(ndarray): Labels.
            feature_search(int): Number of features to search
            during split search.
        Returns:
            None
        """
        if self._bootstrap:
            # Resample with replacement
            bootstrap_indices = np.random.randint(0, x.shape[0], x.shape[0])
            x, y = x[bootstrap_indices], y[bootstrap_indices]

        self._root.train(x, y, feature_search,
                         self._max_depth, self._min_samples_split,
                         self._min_samples_leaf)

    def eval(self, x, y):
        """Return error on dataset"""
        return 100 * (1 - self._root.eval(x, y) / x.shape[0])

    def node_count(self):
        """Count nodes in tree."""
        return self._root.count()

    def predict(self, x):
        """Predict class for one observation.
        Args:
            x(ndarray): A single observation.
        Returns:
            (int): Predicted class index.
        """
        return self._root.predict(x)

class Forest:
    def __init__(self, max_depth=5, no_trees=7,
                 min_samples_split=2, min_samples_leaf=1, feature_search=None,
                 bootstrap=True):
        """Random Forest implementation using numpy.
        Args:
            max_depth(int): Max depth of trees.
            no_trees(int): Number of trees.
            min_samples_split(int): Number of samples in a node to allow
            split search.
            min_samples_leaf(int): Number of samples to be deemed a leaf node.
            feature_search(int): Number of features to search when splitting.
            bootstrap(boolean): Resample dataset with replacement
        """
        self._trees = []
        self._max_depth = max_depth
        self._no_trees = no_trees
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._feature_search = feature_search
        self._bootstrap = bootstrap

    def train(self, x, y):
        """Training procedure.
        Args:
            x(ndarray): Inputs.
            y(ndarray): Labels.
        Returns:
            None
        """
        print('Training Forest...\n')
        for i in range(self._no_trees):
            print('\nTraining Decision Tree no {}...\n'.format(i + 1))
            tree = Tree(max_depth=self._max_depth,
                        min_samples_split=self._min_samples_split,
                        min_samples_leaf=self._min_samples_leaf,
                        bootstrap=self._bootstrap)
            tree.train(x, y, feature_search=self._feature_search)
            self._trees.append(tree)

    def eval(self, x, y):
        """"Evaluate accuracy on dataset."""
        p = self.predict(x)
        return np.sum(p == y) / x.shape[0]

    def predict(self, x):
        """Return predicted labels for given inputs."""
        return np.array([self._aggregate(x[i]) for i in range(x.shape[0])])

    def _aggregate(self, x):
        """Predict class by pooling predictions from all trees.
        Args:
            x(ndarray): A single example.
        Returns:
            (int): Predicted class index.
        """
        temp = [t.predict(x) for t in self._trees]
        _classes, counts = np.unique(np.array(temp), return_counts=True)

        # Return class with max count
        return _classes[np.argmax(counts)]

    def node_count(self):
        """Return number of nodes in forest."""
        return np.sum([t.node_count() for t in self._trees])

def serialize_node(model):
    serialized_model = {
        'feature_idx':model._feature_idx,
        'split_value':model._split_value,
        'leaf':model._leaf,
        'label':model._label,
        'depth':model._depth
    }

    if not model._leaf:
        serialized_model['left_child'] = serialize_node(model._left_child)
        serialized_model['right_child'] = serialize_node(model._right_child)

    return serialized_model

def deserialize_node(model_dict):
    deserialized_node = Node()
    deserialized_node._feature_idx = model_dict['feature_idx']
    deserialized_node._split_value = model_dict['split_value']
    deserialized_node._leaf = model_dict['leaf']
    deserialized_node._label = model_dict['label']
    deserialized_node._depth = model_dict['depth']
    if not deserialized_node._leaf:
        deserialized_node._left_child = deserialize_node(model_dict['left_child'])
        deserialized_node._left_child = deserialize_node(model_dict['left_child'])
    return deserialized_node

def serialize_decision_tree(model):
    serialized_model = {
        'max_depth':model._max_depth,
        'min_samples_split':model._min_samples_split,
        'min_samples_leaf':model._min_samples_leaf,
        'bootstrap':model._bootstrap,
        'root': serialize_node(model._root)
    }
    return serialized_model

def deserialize_decision_tree(model_dict):
    deserialized_decision_tree = Tree()
    deserialized_decision_tree._max_depth = model_dict['max_depth']
    deserialized_decision_tree._min_samples_split = model_dict['min_samples_split']
    deserialized_decision_tree._min_samples_leaf = model_dict['min_samples_leaf']
    deserialized_decision_tree._bootstrap = model_dict['bootstrap']
    deserialized_decision_tree._root = deserialize_node(model_dict['root'])
    return deserialized_decision_tree

def serialize_rf(model):
    # self._trees = []
    # self._max_depth = max_depth
    # self._no_trees = no_trees
    # self._min_samples_split = min_samples_split
    # self._min_samples_leaf = min_samples_leaf
    # self._feature_search = feature_search
    # self._bootstrap = bootstrap
    serialized_model = {
        'max_depth':model._max_depth,
        'no_trees':model._no_trees,
        'min_samples_split':model._min_samples_split,
        'min_samples_leaf':model._min_samples_leaf,
        'feature_search':model._feature_search,
        'bootstrap':model._bootstrap,
        'trees':[serialize_decision_tree(dectree) for dectree in model._trees]
    }

    return serialized_model

def deserialize_rf(model_dict):
    deserialized_rf = Forest()
    deserialized_rf._max_depth = model_dict['max_depth']
    deserialized_rf._no_trees = model_dict['no_trees']
    deserialized_rf._min_samples_split = model_dict['min_samples_split']
    deserialized_rf._min_samples_leaf = model_dict['min_samples_leaf']
    deserialized_rf._feature_search = model_dict['feature_search']
    deserialized_rf._bootstrap = model_dict['bootstrap']
    deserialized_rf._trees = [deserialize_decision_tree(tree) for tree in model_dict['trees']]
    return deserialized_rf