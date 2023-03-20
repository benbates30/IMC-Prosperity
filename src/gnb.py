import numpy as np
from statistics import NormalDist
from sklearn.metrics import confusion_matrix


def serialize_gnb(model):
    serialized_model = {
        'classes':model.classes,
        'mean':model.mean.tolist(),
        'std':model.std.tolist(),
        'c_mean':model.c_mean.tolist(),
        'c_std':model.c_std.tolist(),
        'prior':model.prior.tolist()
    }

    return serialized_model

def deserialize_gnb(model_dict):
    deserialized_gnb = NaiveBayes()
    deserialized_gnb.classes = model_dict['classes']
    deserialized_gnb.mean = np.array(model_dict['mean'])
    deserialized_gnb.std = np.array(model_dict['std'])
    deserialized_gnb.c_mean = np.array(model_dict['c_mean'])
    deserialized_gnb.c_std = np.array(model_dict['c_std'])
    deserialized_gnb.prior = np.array(model_dict['prior'])
    
    return deserialized_gnb

class NaiveBayes(object):

    def train (self, X, y):

        """
            Calculates population and class-wise mean and standard deviation
        """

        # Population mean and standard deviation

        self.classes = set(y)
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

        # Class mean and standard deviation

        self.c_mean = np.zeros((len(self.classes), X.shape[1]))
        self.c_std = np.zeros((len(self.classes), X.shape[1]))
        self.prior = np.zeros((len(self.classes),))

        for c in self.classes:
            indices = np.where(y == c)
            self.prior[c] = indices[0].shape[0] / float(y.shape[0])
            self.c_mean[c] = np.mean(X[indices], axis=0)
            self.c_std[c] = np.std(X[indices], axis=0)

        return

    def predict (self, X):

        """
            Calculates observations' posteriors and returns class with 
            maximum posterior.
        """

        p = []

        for obs in X:

            tiled = np.repeat([obs], len(self.classes), axis=0)

            # Probability of observation in population

            evidence = NormalDist().pdf((self.mean - obs) / self.std)
            evidence = np.prod(evidence)

            # Probability of observation in each class

            likelihood = NormalDist().pdf((tiled - self.c_mean) / self.c_std)
            likelihood = np.prod(likelihood, axis=1)

            # Probability of each class given observation

            posterior = self.prior * likelihood / evidence
            p.append(np.argmax(posterior))

        return p
    
    def eval(self, X, y):
        """"Evaluate accuracy on dataset."""
        p = self.predict(X)
        print(confusion_matrix(y, p))
        return np.sum(p == y) / X.shape[0]