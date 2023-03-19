import math

def euclidian_distance(v1, v2):
    # Getting vector 1 size and initializing summing variable 
    length, summation = len(v1), 0

    # Adding the square of the difference of the values of the two vectors
    for i in range(length - 1):
        # Adding the square of the difference of the values of the two vectors
        summation += math.pow(v1[i] - v2[i], 2)

    # Returning the square root of the sum
    return math.sqrt(summation)

def knn(train, new_sample, K):
    # Initializing dict of distances and variable with size of training set
    distances, train_length = {}, len(train)

    # Calculating the Euclidean distance between the new
    # sample and the values of the training sample
    for i in range(train_length):
        d = euclidian_distance(train[i], new_sample)
        distances[i] = d
    
    # Selecting the K nearest neighbors
    k_neighbors = sorted(distances, key=distances.get)[:]
    
    # Initializing labels counters
    label_1, label_2 = 0, 0
    for index in k_neighbors:
        if train[index][-1] == 1:
            label_1 += 1
        else:
            label_2 += 1
    if label_1 > label_2:
        return 1
    else:
        return 2    