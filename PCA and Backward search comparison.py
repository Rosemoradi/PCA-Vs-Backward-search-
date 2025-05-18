import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Question 1:

# Set a seed for reproducibility
np.random.seed(45)

# Define means for two classes
mean_class_1 = np.array([2.1, 3.0, 3.15, 2.0, 3.3, 2.5, 2.25, 2.8, 3.5, 1.5])
mean_class_2 = np.array([3.8, 2.5, 2.25, 2.0, 3.0, 3.1, 2.1, 3.5, 2.4, 3.2])

# Define a covariance matrix (correlation among attributes)
cov_matrix_i = np.array([
    [1.0, 0.7, 0.5, 0.6, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0],  # Feature 1
    [0.7, 1.0, 0.6, 0.5, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0],  # Feature 2
    [0.5, 0.6, 1.0, 0.7, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],  # Feature 3
    [0.6, 0.5, 0.7, 1.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0],  # Feature 4
    [0.4, 0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Feature 5
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.3, 0.2, 0.0, 0.0],  # Feature 6
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 1.0, 0.4, 0.2, 0.0],  # Feature 7
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 1.0, 0.5, 0.3],  # Feature 8
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.5, 1.0, 0.6],  # Feature 9
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.6, 1.0]  # Feature 10
])
# Generate random samples for each class
n_samples = 1000
class_1_data = np.random.multivariate_normal(mean_class_1, cov_matrix_i, size=n_samples)
class_2_data = np.random.multivariate_normal(mean_class_2, cov_matrix_i, size=n_samples)

# Combine into one dataset

X = np.concatenate((class_1_data, class_2_data))
y = np.concatenate((np.zeros(1000), np.ones(1000)))
# Question 2a
# Center the data
X_mean = np.mean(X, axis=0)
X_centered = X - X_mean

# Calculating the covariance matrix of the centered data
cov_matrix = np.cov(X_centered, rowvar=False)
# Eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
# Sorting eigenvalues and eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]  # Indices of sorted eigenvalues (descending order)
eigenvalues = eigenvalues[sorted_indices]  # Sorted eigenvalues
eigenvectors = eigenvectors[:, sorted_indices]


# Function to project data into PCA space
def calculate_scores(X_centered, eigenvectors, n_components):
    """
    Projects the centered data onto the top `n_components` principal components.
    """
    top_eigenvectors = eigenvectors[:, :n_components]  # Select top `n_components` eigenvectors
    scores = np.dot(X_centered, top_eigenvectors)  # Compute scores by projecting data
    return scores


# Function to reconstruct the data from PCA space
def reconstruct_data(scores, eigenvectors, n_components, mean):
    """
    Reconstructs the data from PCA scores.
    """
    top_eigenvectors = eigenvectors[:, :n_components]  # Select top eigenvectors
    reconstructed_centered = np.dot(scores, top_eigenvectors.T)  # Dot product with eigenvectors' transpose
    reconstructed = reconstructed_centered + mean  # Add the mean back to restore original scale
    return reconstructed


# List of dimensions to retain
dimensions = [10, 9, 8, 7, 6, 5]

# Store MSE values for each dimensionality
mse_list = []

# Iterate over dimensions
for d in dimensions:
    # Step 1: Project data into `d` dimensions (scores)
    X_scores_d = calculate_scores(X_centered, eigenvectors, d)

    # Step 2: Reconstruct the data from the PCA scores
    X_reconstructed_d = reconstruct_data(X_scores_d, eigenvectors, d, X_mean)

    # Step 3: Compute the Mean Squared Error (MSE)
    mse = np.mean((X - X_reconstructed_d) ** 2)  # MSE between original and reconstructed data
    mse_list.append(mse)

# Plot the results

plt.bar(dimensions, mse_list)
plt.xlabel('Number of Retained Dimensions')
plt.ylabel('Reconstruction MSE')
plt.title('Reconstruction MSE vs Retained Dimensions')
plt.grid()
plt.show()

# Question 2.b:


# List of dimensions to retain
dimensions = [10, 9, 8, 7, 6, 5]

# Store classification error for each dimensionality
classification_errors = []

for d in dimensions:
    # Step 1: Compute PCA scores with `d` dimensions
    X_scores_d = calculate_scores(X_centered, eigenvectors, d)

    # Step 2: Split PCA scores into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scores_d,y, test_size=0.4, random_state=42)

    # Step 3: Train the LDA classifier on the training set
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)

    # Step 4: Predict class labels for the test set
    y_pred = lda.predict(X_test)

    # Step 5: Compute classification error on the test set
    error = 1 - accuracy_score(y_test, y_pred)
    classification_errors.append(error)

# Plot the classification error vs retained dimensions
plt.plot(dimensions, classification_errors, marker='o')
plt.xlabel('Number of Retained Dimensions')
plt.ylabel('Classification Error')
plt.title('Classification Error vs Retained Dimensions (Train-Test Split)')
plt.grid()
plt.show()


# Question 3:

# Function for backward search
def backward_feature_elimination(X, y, start_dimensions):
    """
    Performs backward search to reduce the dimensionality of the dataset by minimizing error.
    """
    remaining_features = list(range(X.shape[1]))  # Start with all features
    classification_errors = []

    for target_dimensions in range(start_dimensions, 4, -1):  # Reduce to 10, 9, ..., 5
        print(f"\nRetaining {target_dimensions} dimensions...")

        # Initialize the best error as infinity
        best_error = float('inf')
        worst_feature = None

        # Iterate over all features and test their removal
        for feature in remaining_features:
            # Remove the current feature from consideration
            test_features = [f for f in remaining_features if f != feature]
            X_reduced = X[:, test_features]  # Subset dataset to exclude `feature`

            # Train-test split (use entire dataset for simplicity)
            lda = LinearDiscriminantAnalysis()
            lda.fit(X_reduced, y)
            y_pred = lda.predict(X_reduced)

            # Compute classification error
            error = 1 - accuracy_score(y, y_pred)
            if error < best_error:
                best_error = error
                worst_feature = feature

        # Remove the feature that minimizes classification error
        remaining_features.remove(worst_feature)
        classification_errors.append(best_error)

        print(f"Removed feature: {worst_feature}")
        print(f"Classification error: {best_error}")

    return classification_errors

# Apply backward search
start_dimensions = 10
classification_errors = backward_feature_elimination(X, y, start_dimensions)

# Plot the classification error vs retained dimensions
retained_dimensions = list(range(start_dimensions, 4, -1))
plt.plot(retained_dimensions, classification_errors, marker='o')
plt.xlabel('Number of Retained Dimensions')
plt.ylabel('Classification Error')
plt.title('Classification Error vs Retained Dimensions (Backward Search)')
plt.grid()
plt.show()

# Q4: Comparison:
# for 2b (PCA):
# Shape of the Curve:

# The classification error starts higher when fewer dimensions (5) are retained
# and decreases as more dimensions are added.
# The minimum classification error is achieved at around 10 dimensions, which is
# expected as no information is lost in the original 10-dimensional dataset.
# Trend:
#
# The curve has some minor fluctuations, particularly between 7 and 9 dimensions.
# There is a gradual decrease in classification error as dimensions increase, which
# suggests that PCA retains important information, but small variances in the data affect the classifier performance.
# for question number 3: (Backward Search)
# Shape of the Curve:
# The classification error also decreases as the number of retained dimensions increases,
# but the decrease is more consistent and smoother than in Part 2.b.
# The classification error stabilizes at around 8 retained dimensions, which suggests that backward
# search may be better at selecting the most informative features early.
# Trend:
# Backward search appears more effective at reducing classification error for fewer dimensions (5–8). For example:
# At 5 dimensions, the error for backward search (~0.015) is lower than PCA (~0.022).
# At 8 dimensions, backward search achieves slightly better performance compared to PCA.
# Conclusion
# PCA is useful for reducing dimensionality while preserving variance but may not always yield the best
# classification performance.
# Backward Search is more tailored to the classification task and achieves lower classification error,
# especially when retaining fewer dimensions.
# If computational resources allow, Backward Search is the preferred method when the goal is to maximize
# classification accuracy.
# However, if computational efficiency is important, PCA is a reasonable alternative that provides decent performance.