import numpy as np
import pandas as pd

def serialize_node(model):
    serialized_model = {
        'column':model.column,
        'threshold':model.threshold,
        'probas':model.probas,
        'is_terminal':model.is_terminal,
        'depth':model.depth
    }

    if not model.is_terminal:
        serialized_model['left'] = serialize_node(model.left)
        serialized_model['right'] = serialize_node(model.right)

    return serialized_model

def deserialize_node(model_dict):
    deserialized_node = Node()
    deserialized_node.column = model_dict['column']
    deserialized_node.threshold = model_dict['threshold']
    deserialized_node.probas = model_dict['probas']
    deserialized_node.is_terminal = model_dict['is_terminal']
    deserialized_node.depth = model_dict['depth']
    if not deserialized_node.is_terminal:
        deserialized_node.left = deserialize_node(model_dict['left'])
        deserialized_node.right = deserialize_node(model_dict['right'])
    return deserialized_node

def serialize_decision_tree(model):
    serialized_model = {
        'max_depth':model.max_depth,
        'min_samples_split':model.min_samples_split,
        'min_samples_leaf':model.min_samples_leaf,
        'classes':model.classes,
        'Tree': serialize_node(model.Tree)
    }
    return serialized_model

def deserialize_decision_tree(model_dict):
    deserialized_decision_tree = DecisionTreeClassifier()
    deserialized_decision_tree.max_depth = model_dict['max_depth']
    deserialized_decision_tree.min_samples_split = model_dict['min_samples_split']
    deserialized_decision_tree.min_samples_leaf = model_dict['min_samples_leaf']
    deserialized_decision_tree.classes = model_dict['classes']
    deserialized_decision_tree.Tree = deserialize_node(model_dict['Tree'])
    return deserialized_decision_tree

class Node:
    def __init__(self):
        
        # links to the left and right child nodes
        self.right = None
        self.left = None
        
        # derived from splitting criteria
        self.column = None
        self.threshold = None
        
        # probability for object inside the Node to belong for each of the given classes
        self.probas = None
        # depth of the given node
        self.depth = None
        
        # if it is the root Node or not
        self.is_terminal = False

class DecisionTreeClassifier:
    def __init__(self, max_depth = 3, min_samples_leaf = 1, min_samples_split = 2):
        
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        
        self.classes = None
        
        # Decision tree itself
        self.Tree = None
    
    def nodeProbas(self, y):
        '''
        Calculates probability of class in a given node
        '''
        
        probas = []
        
        # for each unique label calculate the probability for it
        for one_class in self.classes:
            proba = y[y == one_class].shape[0] / y.shape[0]
            probas.append(proba)
        return np.asarray(probas)

    def gini(self, probas):
        '''
        Calculates gini criterion
        '''
        
        return 1 - np.sum(probas**2)
    
    def calcImpurity(self, y):
        '''
        Wrapper for the impurity calculation. Calculates probas first and then passses them
        to the Gini criterion
        '''
        
        return self.gini(self.nodeProbas(y))
    
    def calcBestSplit(self, X, y):
        '''
        Calculates the best possible split for the concrete node of the tree
        '''
        
        bestSplitCol = None
        bestThresh = None
        bestInfoGain = -999
        
        impurityBefore = self.calcImpurity(y)
        
        # for each column in X
        for col in range(X.shape[1]):
            x_col = X[:, col]
            
            # for each value in the column
            for x_i in x_col:
                threshold = x_i
                y_right = y[x_col > threshold]
                y_left = y[x_col <= threshold]
                
                if y_right.shape[0] == 0 or y_left.shape[0] == 0:
                    continue
                    
                # calculate impurity for the right and left nodes
                impurityRight = self.calcImpurity(y_right)
                impurityLeft = self.calcImpurity(y_left)
                
                # calculate information gain
                infoGain = impurityBefore
                infoGain -= (impurityLeft * y_left.shape[0] / y.shape[0]) + (impurityRight * y_right.shape[0] / y.shape[0])
                
                # is this infoGain better then all other?
                if infoGain > bestInfoGain:
                    bestSplitCol = col
                    bestThresh = threshold
                    bestInfoGain = infoGain
                    
        
        # if we still didn't find the split
        if bestInfoGain == -999:
            return None, None, None, None, None, None
        
        # making the best split
        
        x_col = X[:, bestSplitCol]
        x_left, x_right = X[x_col <= bestThresh, :], X[x_col > bestThresh, :]
        y_left, y_right = y[x_col <= bestThresh], y[x_col > bestThresh]
        
        return bestSplitCol, bestThresh, x_left, y_left, x_right, y_right
                
    
    def buildDT(self, X, y, node):
        '''
        Recursively builds decision tree from the top to bottom
        '''
        
        # checking for the terminal conditions
        
        if node.depth >= self.max_depth:
            node.is_terminal = True
            return
        
        if X.shape[0] < self.min_samples_split:
            node.is_terminal = True
            return
        
        if np.unique(y).shape[0] == 1:
            node.is_terminal = True
            return
        
        # calculating current split
        splitCol, thresh, x_left, y_left, x_right, y_right = self.calcBestSplit(X, y)
        
        if splitCol is None:
            node.is_terminal = True
            
        if x_left.shape[0] < self.min_samples_leaf or x_right.shape[0] < self.min_samples_leaf:
            node.is_terminal = True
            return
        
        node.column = splitCol
        node.threshold = thresh
        
        # creating left and right child nodes
        node.left = Node()
        node.left.depth = node.depth + 1
        node.left.probas = self.nodeProbas(y_left)
        
        node.right = Node()
        node.right.depth = node.depth + 1
        node.right.probas = self.nodeProbas(y_right)
        
        # splitting recursevely
        self.buildDT(x_right, y_right, node.right)
        self.buildDT(x_left, y_left, node.left)
    
    def fit(self, X, y):
        '''
        Standard fit function to run all the model training
        '''
        if type(X) == pd.DataFrame:
            X = np.asarray(X)
        
        self.classes = np.unique(y)
        # root node creation
        self.Tree = Node()
        self.Tree.depth = 1
        self.Tree.probas = self.nodeProbas(y)
        
        self.buildDT(X, y, self.Tree)
    
    def predictSample(self, x, node):
        '''
        Passes one object through decision tree and return the probability of it to belong to each class
        '''
        # if we have reached the terminal node of the tree
        if node.is_terminal:
            return node.probas
        
        if x[node.column] > node.threshold:
            probas = self.predictSample(x, node.right)
        else:
            probas = self.predictSample(x, node.left)
            
        return probas
        
    def predict(self, X):
        '''
        Returns the labels for each X
        '''
        
        if type(X) == pd.DataFrame:
            X = np.asarray(X)
            
        predictions = []
        for x in X:
            pred = np.argmax(self.predictSample(x, self.Tree))
            predictions.append(pred)
        
        return np.asarray(predictions)
    
    def eval(self, X, y):
        """"Evaluate accuracy on dataset."""
        p = self.predict(X)
        return np.sum(p == y) / X.shape[0]
    
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
            tree = DecisionTreeClassifier(max_depth=self._max_depth,
                        min_samples_split=self._min_samples_split,
                        min_samples_leaf=self._min_samples_leaf)
            tree.fit(x, y)
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