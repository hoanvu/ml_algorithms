class KNearestNeighbors(object):
    def __init__(self, n_neighbors=1, p=2):
        """ Initialize the classifier """
        self.n_neighbors = n_neighbors
        self.p = p # `p` is how we decide the formula for distance calculation
        self.X_train = np.array([])
        self.y_train = np.array([])
        
    def fit(self, X_train, y_train):
        """ 
        Train the data. In K-nearest neighbors algorithm, no need to train the data, we will simply 
        return the input
        """
        self.X_train = X_train
        self.y_train = y_train
        
    def calculate_distance(self, input_data, new_data):
        """
        A generalized method to calculate distance between new data point and all existing points 
        in training data based on Minkowski formula.
        
        self.p = 1 => Manhattan distance
        self.p = 2 => Euclidean distance
        
        Basically we can choose whichever value for `p`, the bigger it is, the more a large difference 
        in one dimension it will influence the total difference.
        
        The formula is: distance(X, Y) = (sum of (Xi - Yi)^p ) ^ 1/p (with i=1 to number of features in input)
        """
        sum_of_power_p = np.sum(np.power(input_data - new_data, self.p), axis=1)
        return np.power(sum_of_power_p, 1.0/self.p)
    
    def predict_single(self, observation):
        """
        Predict a single observation. The input should be a 1D array that has same number of features as input
        except the label. Later this method will be fetched to the main predict() method for batch processing
        """
        distances = self.calculate_distance(self.X_train, observation)
            
        label_dist = np.c_[self.y_train, distances]
        label_dist = label_dist[label_dist[:, 1].argsort()][:self.n_neighbors]
        label, counts = np.unique(label_dist[:, 0], return_counts=True)
        label_counts_dict = dict(zip(label, counts))
        return max(label_counts_dict.items(), key=operator.itemgetter(1))[0]
        
    def predict(self, observations):
        """
        Predict all data, `observations` is a 2D matrix by applying the predict_single() method to all rows
        """
        return np.apply_along_axis(self.predict_single, arr=observations, axis=1)
    
