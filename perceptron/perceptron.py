class Perceptron(object):
    def __init__(self):
        """
        weights: some random values used as weights for input data. 
                This weight will be adjusted later through learning process.
        """
        self.weights = []
        self.lr = 0.1 # learning rate

    def init_weights(self, inputs):
        self.weights = [np.random.rand(1)[0]] * len(inputs[0])
        
    def __guess(self, data_point):
        weighted_sum = np.sum(data_point * self.weights)

        # Using np.sign() to get the output for the weighted sum
        # If weighted_sum >= 0, return 1, otherwise -1        
        return np.sign(weighted_sum)
        
    def fit(self, data_point, label):
        prediction = self.__guess(data_point)

        # Calculate the error
        error = label - prediction
        
        # Adjust all the weights based on errors
        self.weights += error * data_point * self.lr
    
    def predict(self, inputs):
        weighted_sums = np.dot(inputs, self.weights)        
        return np.array(list(map(np.sign, weighted_sums)))
