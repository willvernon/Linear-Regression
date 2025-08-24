import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations = 1000):
        """
        Initialize the model with hyperparameters.
        - learning_rate: Step size for gradient descent updates.
        - iterations: Number of gradient descent steps.
        """
        self.learning_rate = learning_rate
        self.interations = interations
        self.weights = None # Slopes (one per feature)
        self.bias = None # Intercept

    def fit(self, X, y):
        """
        Train the model using gradient descent.
        - X: 2D array of features (samples x features)
        - y: 1D array of targets
        """
        # Get number of samples and features
        m, n_features = X.shape
        self.weights = np.zeros(n_features) # Initialize weights to zeros
        self.bias = 0                       # Initialize bias to zero

        for _ in range(self.iterations):
            # Predict: y_pred = X * weights + bias (matrix multiplication)
            y_pred = np.dot(X, self.weights) + self.bias

            # Compute gradients (partial derivatives)
            # dw = (1/m) * (X.T dot (y_pred - y))
            dw = (1/m) * np.dot(X.T, (y_pred - y))
            # db = (1/m) * sum(y_pred - y)
            db = (1 / m) * np.sum(y_pred -y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Make predictions on new data.
        - X: 2D array of features
        Returns: Predicted targets
        """
        return np.dot(X, self.weights) + self.bias


## Using with Sample Data
# Gen sample data (or load your own)
np.random.seed(42) # For reproducibility
X = np.random.rand(100, 1) * 10 # 100 samples, 1 feature (e.g., house size)
y = 2 * X.squeeze() + 3 + np.random.randn(100) # True line: y = 2x + 3 + noise

# Train the model
model = LinearRegression(learning_rate=0.01, iterations=1000)
model.fit(X, y)

#Make predictions
new_X = np.array([[5], [10]]) # New data points
predictions = model.predict(new_X)
print("Predictions:", predictions) # Should be around [13, 23]

# Learned parameters (for inspection)
print("Weights:", model.weights) # Should be ~[2]
print("Bias:", model.bias)       # Should be ~3

