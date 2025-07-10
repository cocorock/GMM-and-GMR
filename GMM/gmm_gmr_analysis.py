import scipy.io
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_mat_to_dataframe(file_path):
    """
    Loads a .mat file and converts its content to a pandas DataFrame.
    Assumes the .mat file contains a single array that can be directly converted.
    """
    mat_data = scipy.io.loadmat(file_path)
    # Assuming the relevant data is stored under a key that is not '__header__', '__version__', '__globals__'
    # We need to find the actual data key. Let's assume it's the first non-meta key.
    data_key = [key for key in mat_data if not key.startswith('__')][0]
    data_array = mat_data[data_key]
    return pd.DataFrame(data_array)

def gmr_predict(gmm, x_query, input_dim, output_dim):
    """
    Performs Gaussian Mixture Regression (GMR) to predict output_dim from input_dim.
    gmm: Trained GaussianMixture model
    x_query: Input data for prediction (e.g., time points)
    input_dim: Indices of input dimensions in the original data
    output_dim: Indices of output dimensions in the original data
    """
    n_components = gmm.n_components
    means = gmm.means_
    covariances = gmm.covariances_
    weights = gmm.weights_

    y_pred = np.zeros((x_query.shape[0], len(output_dim)))
    y_cov = np.zeros((x_query.shape[0], len(output_dim), len(output_dim)))

    for i, x_q in enumerate(x_query):
        # Calculate responsibilities for each component
        responsibilities = np.zeros(n_components)
        for k in range(n_components):
            # Marginalize over output dimensions for the input part
            mu_x = means[k, input_dim]
            sigma_x = covariances[k][np.ix_(input_dim, input_dim)]
            
            # Handle singular covariance matrices by adding a small diagonal
            if np.linalg.det(sigma_x) == 0:
                sigma_x += np.eye(sigma_x.shape[0]) * 1e-6

            try:
                responsibilities[k] = weights[k] * \
                                      (1 / np.sqrt(2 * np.pi * np.linalg.det(sigma_x))) * \
                                      np.exp(-0.5 * (x_q - mu_x).T @ np.linalg.inv(sigma_x) @ (x_q - mu_x))
            except np.linalg.LinAlgError:
                responsibilities[k] = 0 # If inverse fails, responsibility is zero

        responsibilities /= np.sum(responsibilities) # Normalize

        # Predict output
        mean_y_given_x = np.zeros(len(output_dim))
        cov_y_given_x = np.zeros((len(output_dim), len(output_dim)))

        for k in range(n_components):
            mu_x = means[k, input_dim]
            mu_y = means[k, output_dim]
            sigma_xx = covariances[k][np.ix_(input_dim, input_dim)]
            sigma_xy = covariances[k][np.ix_(input_dim, output_dim)]
            sigma_yx = covariances[k][np.ix_(output_dim, input_dim)]
            sigma_yy = covariances[k][np.ix_(output_dim, output_dim)]

            if np.linalg.det(sigma_xx) == 0:
                sigma_xx += np.eye(sigma_xx.shape[0]) * 1e-6

            try:
                inv_sigma_xx = np.linalg.inv(sigma_xx)
            except np.linalg.LinAlgError:
                inv_sigma_xx = np.zeros_like(sigma_xx) # Fallback if inverse fails

            # Conditional mean
            mean_k = mu_y + sigma_yx @ inv_sigma_xx @ (x_q - mu_x)
            mean_y_given_x += responsibilities[k] * mean_k

            # Conditional covariance
            cov_k = sigma_yy - sigma_yx @ inv_sigma_xx @ sigma_xy
            cov_y_given_x += responsibilities[k] * (cov_k + np.outer(mean_k, mean_k))

        y_pred[i] = mean_y_given_x
        y_cov[i] = cov_y_given_x - np.outer(mean_y_given_x, mean_y_given_x) # Total covariance

    return y_pred, y_cov


# --- Main Script ---
file_path = "C:/Users/quepe/Documents/GitHub/GMM-and-GMR/left_leg_linear_kinematics.mat"
df = load_mat_to_dataframe(file_path)

# Select a subset of only 10 gait cycles (each cycle is 200 columns)
num_gait_cycles = 10
data_subset = df.iloc[:, :num_gait_cycles * 200].T # Transpose to have rows as samples

# Prepare data for GMM: [time, right_ankle_x, right_ankle_y, left_ankle_x, left_ankle_y, right_ankle_vx, right_ankle_vy, left_ankle_vx, left_ankle_vy]
# Assuming the rows are in the order specified:
#    * Row 0: time
#    * Row 1: right ankle x position
#    * Row 2: left ankle x position
#    * Row 3: right ankle y position
#    * Row 4: left ankle y position
#    * Row 5: right ankle x velocity
#    * Row 6: left ankle x velocity
#    * Row 7: right ankle y velocity
#    * Row 8: left ankle y velocity

# Create the GMM input data (X)
# The data is already transposed, so columns are now rows.
# We need to select the first 9 rows (original columns) as features.
# New order: [time, right_ankle_x, right_ankle_y, left_ankle_x, left_ankle_y, right_ankle_vx, right_ankle_vy, left_ankle_vx, left_ankle_vy]
X = data_subset.iloc[:, [0, 1, 3, 2, 4, 5, 7, 6, 8]].values

# Train GMM models with 4 to 9 components
n_components_range = range(4, 30)
bic = []
gmm_models = {}

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, random_state=0, covariance_type='full')
    gmm.fit(X)
    bic.append(gmm.bic(X))
    gmm_models[n_components] = gmm

# Choose the model with the best (lowest) BIC
best_n_components = n_components_range[np.argmin(bic)]
best_gmm = gmm_models[best_n_components]
print(f"Best GMM model has {best_n_components} components (lowest BIC).")

# PCA for visualization
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# Get cluster assignments from the best GMM model
cluster_assignments = best_gmm.predict(X)

plt.figure(figsize=(18, 8))

plt.subplot(1, 2, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_assignments, cmap='viridis', s=50, alpha=0.3)
plt.title('2D PCA of Data (PC1 vs PC2) with GMM Cluster Assignments')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Cluster')
plt.grid(True)

plt.subplot(1, 2, 2)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 2], c=cluster_assignments, cmap='viridis', s=50, alpha=0.3)
plt.title('2D PCA of Data (PC1 vs PC3) with GMM Cluster Assignments')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 3')
plt.colorbar(scatter, label='Cluster')
plt.grid(True)

plt.tight_layout()
plt.savefig('pca_clusters.png')
plt.show()

# GMR to recover trajectory
# Input: time (dimension 0)
# Output: right ankle x, y (dimensions 1, 2) and left ankle x, y (dimensions 3, 4)

time_query = np.linspace(0, 1, 200).reshape(-1, 1) # A single gait cycle for prediction

# Predict for right ankle (x, y)
right_ankle_pred, _ = gmr_predict(best_gmm, time_query, input_dim=[0], output_dim=[1, 2])

# Predict for left ankle (x, y)
left_ankle_pred, _ = gmr_predict(best_gmm, time_query, input_dim=[0], output_dim=[3, 4])

# Plotting recovered trajectories and original gait cycles
plt.figure(figsize=(14, 7))

# Right Ankle Plot
plt.subplot(1, 2, 1)
plt.plot(right_ankle_pred[:, 0], right_ankle_pred[:, 1], 'r-', linewidth=2, label='Recovered Trajectory')
for i in range(num_gait_cycles):
    start_idx = i * 200
    end_idx = (i + 1) * 200
    plt.plot(X[start_idx:end_idx, 1], X[start_idx:end_idx, 2], 'b--', alpha=0.5, label='Original Gait Cycles' if i == 0 else "")
plt.title('Right Ankle: Recovered vs. Original Trajectories (X vs Y)')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid(True)

# Left Ankle Plot
plt.subplot(1, 2, 2)
plt.plot(left_ankle_pred[:, 0], left_ankle_pred[:, 1], 'g-', linewidth=2, label='Recovered Trajectory')
for i in range(num_gait_cycles):
    start_idx = i * 200
    end_idx = (i + 1) * 200
    plt.plot(X[start_idx:end_idx, 3], X[start_idx:end_idx, 4], 'c--', alpha=0.5, label='Original Gait Cycles' if i == 0 else "")
plt.title('Left Ankle: Recovered vs. Original Trajectories (X vs Y)')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('recovered_trajectories.png')
plt.show()
