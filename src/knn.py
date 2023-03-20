import numpy as np

class KNN:
    def __init__(self, X, y, K=20):
        self.X = X
        self.y = y
        self.K = K

    def euclidian_distance(self, v1, v2):
        return np.linalg.norm(v1-v2)

    def predict(self, new_sample):
        # Initializing dict of distances and variable with size of training set
        distances, train_length = {}, len(self.X)

        # Calculating the Euclidean distance between the new
        # sample and the values of the training sample
        for i in range(train_length):
            d = self.euclidian_distance(self.X[i], new_sample)
            distances[i] = d
        
        # Selecting the K nearest neighbors
        k_neighbors = sorted(distances, key=distances.get)[:self.K]
        
        # Initializing labels counters
        ycounter = np.zeros(len(self.y[0]))
        for index in k_neighbors:
            ycounter += self.y[index]
        return np.exp(ycounter)/sum(np.exp(ycounter))
 