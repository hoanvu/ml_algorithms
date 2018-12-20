class KMeansClustering(object):
    def __init__(self, k=1):
        self.k = k
        
    def __init_centroids(self, data):
        """
        Randomly initialize 'k' centroids for the input data. 
        'k' is defined by self.k
        """
        indices = np.random.choice(data.shape[0], size=self.k, replace=False)
        return data[indices]
    
    def __get_distance_mask(self, data, centroids):
        """
        Calculate pairwise distance between a centroid and all other points and
        then return a ndarray that 
        """
        results = []
        # Calculate Euclidean distance from each centroid to all data points
        for c in centroids:
            distances = np.sqrt(np.sum((c - data) ** 2, axis=1))
            results.append(distances)
            
        # After getting all pairwise distances, get the indices of minimum distance
        # This is also the indices of centroids that we need to update each data point
        return np.argmin(results, axis=0)
    
    def __update_centroids(self, data, mask):
        """
        For each iteration, compute the average of all data points in each cluster
        and update centroids accordingly
        """
        new_centroids = []
        
        # Clusters will be encoded as 0, 1, 2, ..., k
        for c in range(self.k):
            c_subset = data[mask == c, :]
            new_centroids.append(np.mean(c_subset, axis=0))
            
        return np.array(new_centroids)
    
    def fit(self, data):
        """
        This is the core method, where it takes input data
        and put each data point to corresponding cluster.
        Number of clusters is defined while initializing the object
        """
        centroids = self.__init_centroids(data)
        # 'centroids_list' stores list of centroids at each iteration
        centroids_list = [centroids]
        # 'masks' stores which data point belong to which cluster in each iteration
        masks = []   
        # Tracks the number of iteration
        count = 0
        
        # Repeat the process until the algorithm converge
        while True:
            count += 1
            mask = self.__get_distance_mask(data, centroids_list[-1])
            new_centroids = self.__update_centroids(data, mask)
            
            masks.append(mask)
            
            # Exit the loop when previous and current centroids lists are same
            # No change to centroids means that the algorithm has converged
            if not (new_centroids - centroids_list[-1]).all():
                centroids_list.append(new_centroids)
                break
            
            centroids_list.append(new_centroids)
            
        return (centroids_list[-1], masks[-1], count)   
