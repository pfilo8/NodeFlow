class EmbeddableTree:

    def fit(self, X, y):
        """Method for fitting Tree model."""
        pass

    def apply(self, X):
        """Method for extracting leafs given data."""
        pass

    def embed(self, X):
        """Method for embedding data using Tree model."""
        pass

    def pred_dist_param(self, X):
        """
        Method for predicting distribution parameters.

        Distribution parameters will be later used as a prior for CNF model.
        Possible values:
          - None - N(0, 1) prior will be used
          - 2D array - First column should be mean and the second logstd.
        """
        pass
