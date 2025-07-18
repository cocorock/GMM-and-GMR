import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.linalg import block_diag, inv
from typing import Dict, List, Tuple, Optional, Union
import pickle
import warnings
warnings.filterwarnings('ignore')

class TPGMMGaitModel:
    """
    Task-Parameterized Gaussian Mixture Model for Gait Data
    Adapted for the specific data structure with time + 2 reference frames (5D each)
    """
    
    def __init__(self, n_components: int = 5, orientation_unit: str = 'degrees', 
                 max_iter: int = 100, tol: float = 1e-3, reg_covar: float = 1e-6):
        """
        Initialize TP-GMM model
        
        Args:
            n_components: Number of Gaussian components
            orientation_unit: 'degrees' or 'radians' for orientation data
            max_iter: Maximum EM iterations
            tol: Convergence tolerance
            reg_covar: Regularization for covariance matrices
        """
        self.n_components = n_components
        self.orientation_unit = orientation_unit.lower()
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        
        # Data structure for gait data
        self.data_structure = {
            'total_dim': 11,      # 1 time + 10 spatial (5 per frame)
            'time_dim': 1,        # Time dimension
            'frame_dims': 5,      # Dimensions per frame (x, y, vx, vy, orientation)
            'n_frames': 2         # Number of reference frames
        }
        
        # Model parameters (will be set after training)
        self.weights_ = None
        self.means_ = None      # Shape: (n_components, n_frames, frame_dims)
        self.covariances_ = None # Shape: (n_components, n_frames, frame_dims, frame_dims)
        self.global_means_ = None     # Shape: (n_components, total_dim)
        self.global_covariances_ = None # Shape: (n_components, total_dim, total_dim)
        
        self.is_fitted_ = False
        
    def _convert_angles_to_radians(self, data: np.ndarray) -> np.ndarray:
        """Convert orientation columns to radians if needed"""
        if self.orientation_unit == 'degrees':
            data_rad = data.copy()
            # Convert orientation columns (indices 5 and 10) from degrees to radians
            if data.shape[1] > 5:
                data_rad[:, 5] = np.deg2rad(data[:, 5])  # FR1 orientation
            if data.shape[1] > 10:
                data_rad[:, 10] = np.deg2rad(data[:, 10])  # FR2 orientation
            return data_rad
        return data.copy()
    
    def _convert_angles_from_radians(self, data: np.ndarray) -> np.ndarray:
        """Convert orientation columns from radians if needed"""
        if self.orientation_unit == 'degrees':
            data_deg = data.copy()
            # Convert orientation columns back to degrees
            if data.shape[1] > 5:
                data_deg[:, 5] = np.rad2deg(data[:, 5])  # FR1 orientation
            if data.shape[1] > 10:
                data_deg[:, 10] = np.rad2deg(data[:, 10])  # FR2 orientation
            return data_deg
        return data.copy()
    
    def _normalize_angles(self, angles: np.ndarray) -> np.ndarray:
        """Normalize angles to [-π, π]"""
        return ((angles + np.pi) % (2 * np.pi)) - np.pi
    
    def _extract_frame_data(self, data: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Extract time and frame data from the input
        
        Args:
            data: Shape (n_samples, 11) - [time, FR1_data(5D), FR2_data(5D)]
            
        Returns:
            time_data: Shape (n_samples, 1)
            frame_data: List of 2 arrays, each shape (n_samples, 5)
        """
        time_data = data[:, :1]  # First column is time
        frame1_data = data[:, 1:6]   # Columns 1-5: FR1 data
        frame2_data = data[:, 6:11]  # Columns 6-10: FR2 data
        
        return time_data, [frame1_data, frame2_data]
    
    def _initialize_parameters(self, data: np.ndarray) -> None:
        """Initialize TP-GMM parameters using K-means"""
        n_samples = data.shape[0]
        
        # Initialize with K-means on the full data
        kmeans = KMeans(n_clusters=self.n_components, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        
        # Initialize weights
        self.weights_ = np.zeros(self.n_components)
        for k in range(self.n_components):
            self.weights_[k] = np.sum(labels == k) / n_samples
        
        # Initialize means and covariances for each frame
        self.means_ = np.zeros((self.n_components, self.data_structure['n_frames'], 
                               self.data_structure['frame_dims']))
        self.covariances_ = np.zeros((self.n_components, self.data_structure['n_frames'],
                                    self.data_structure['frame_dims'], 
                                    self.data_structure['frame_dims']))
        
        # Extract frame data
        time_data, frame_data = self._extract_frame_data(data)
        
        # Initialize parameters for each component and frame
        for k in range(self.n_components):
            mask = labels == k
            if np.sum(mask) > 0:
                for f in range(self.data_structure['n_frames']):
                    frame_k_data = frame_data[f][mask]
                    self.means_[k, f] = np.mean(frame_k_data, axis=0)
                    cov = np.cov(frame_k_data.T)
                    if frame_k_data.shape[0] == 1:
                        cov = np.eye(self.data_structure['frame_dims']) * self.reg_covar
                    self.covariances_[k, f] = cov + np.eye(self.data_structure['frame_dims']) * self.reg_covar
            else:
                # Handle empty clusters
                for f in range(self.data_structure['n_frames']):
                    self.means_[k, f] = np.mean(frame_data[f], axis=0)
                    self.covariances_[k, f] = np.eye(self.data_structure['frame_dims'])
        
        # Initialize global parameters (concatenated frames)
        self._update_global_parameters()
    
    def _update_global_parameters(self) -> None:
        """Update global parameters from frame-specific parameters"""
        total_dim = self.data_structure['total_dim']
        frame_dims = self.data_structure['frame_dims']
        
        self.global_means_ = np.zeros((self.n_components, total_dim))
        self.global_covariances_ = np.zeros((self.n_components, total_dim, total_dim))
        
        for k in range(self.n_components):
            # Global mean: [time_mean, frame1_mean, frame2_mean]
            # For time, we'll use 0.5 as default (middle of [0,1] range)
            self.global_means_[k, 0] = 0.5  # Time dimension
            self.global_means_[k, 1:1+frame_dims] = self.means_[k, 0]  # Frame 1
            self.global_means_[k, 1+frame_dims:] = self.means_[k, 1]   # Frame 2
            
            # Global covariance: block diagonal structure
            # Time variance (small since it's normalized [0,1])
            self.global_covariances_[k, 0, 0] = 0.01
            
            # Frame covariances
            start_idx = 1
            for f in range(self.data_structure['n_frames']):
                end_idx = start_idx + frame_dims
                self.global_covariances_[k, start_idx:end_idx, start_idx:end_idx] = self.covariances_[k, f]
                start_idx = end_idx
    
    def _e_step(self, data: np.ndarray) -> np.ndarray:
        """
        Expectation step: Compute responsibilities
        
        Returns:
            responsibilities: Shape (n_samples, n_components)
        """
        n_samples = data.shape[0]
        log_responsibilities = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            # Compute log probability for each sample under component k
            log_prob = self._compute_log_probability(data, k)
            log_responsibilities[:, k] = np.log(self.weights_[k] + 1e-10) + log_prob
        
        # Normalize responsibilities
        log_prob_norm = np.log(np.sum(np.exp(log_responsibilities), axis=1, keepdims=True) + 1e-10)
        log_responsibilities -= log_prob_norm
        
        return np.exp(log_responsibilities)
    
    def _compute_log_probability(self, data: np.ndarray, component: int) -> np.ndarray:
        """
        Compute log probability of data under a specific component
        using the product of frame-specific Gaussians
        """
        n_samples = data.shape[0]
        time_data, frame_data = self._extract_frame_data(data)
        
        log_prob = np.zeros(n_samples)
        
        # For TP-GMM, we compute the product of probabilities from each frame
        for f in range(self.data_structure['n_frames']):
            frame_mean = self.means_[component, f]
            frame_cov = self.covariances_[component, f]
            
            # Compute multivariate Gaussian log probability
            diff = frame_data[f] - frame_mean
            log_prob += self._multivariate_gaussian_log_prob(diff, frame_cov)
        
        return log_prob
    
    def _multivariate_gaussian_log_prob(self, diff: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Compute log probability of multivariate Gaussian"""
        try:
            cov_inv = inv(cov)
            cov_det = np.linalg.det(cov)
            
            # Regularize if determinant is too small
            if cov_det < 1e-12:
                cov += np.eye(cov.shape[0]) * self.reg_covar
                cov_inv = inv(cov)
                cov_det = np.linalg.det(cov)
            
            k = cov.shape[0]
            mahalanobis = np.sum(diff @ cov_inv * diff, axis=1)
            log_prob = -0.5 * (k * np.log(2 * np.pi) + np.log(cov_det) + mahalanobis)
            
            return log_prob
        except np.linalg.LinAlgError:
            # If matrix is singular, return very low probability
            return np.full(diff.shape[0], -1e10)
    
    def _m_step(self, data: np.ndarray, responsibilities: np.ndarray) -> None:
        """Maximization step: Update parameters"""
        n_samples = data.shape[0]
        
        # Update weights
        resp_sum = np.sum(responsibilities, axis=0)
        self.weights_ = resp_sum / n_samples
        
        # Avoid zero weights
        self.weights_ = np.maximum(self.weights_, 1e-10)
        self.weights_ /= np.sum(self.weights_)
        
        time_data, frame_data = self._extract_frame_data(data)
        
        # Update means and covariances for each component and frame
        for k in range(self.n_components):
            resp_k = responsibilities[:, k]
            resp_sum_k = np.sum(resp_k)
            
            if resp_sum_k > 1e-10:
                for f in range(self.data_structure['n_frames']):
                    # Update means
                    weighted_sum = np.sum(resp_k[:, np.newaxis] * frame_data[f], axis=0)
                    self.means_[k, f] = weighted_sum / resp_sum_k
                    
                    # Update covariances
                    diff = frame_data[f] - self.means_[k, f]
                    weighted_cov = np.zeros((self.data_structure['frame_dims'], 
                                           self.data_structure['frame_dims']))
                    
                    for i in range(n_samples):
                        weighted_cov += resp_k[i] * np.outer(diff[i], diff[i])
                    
                    self.covariances_[k, f] = weighted_cov / resp_sum_k
                    
                    # Add regularization
                    self.covariances_[k, f] += np.eye(self.data_structure['frame_dims']) * self.reg_covar
        
        # Update global parameters
        self._update_global_parameters()
    
    def _compute_log_likelihood(self, data: np.ndarray) -> float:
        """Compute log likelihood of the data"""
        n_samples = data.shape[0]
        log_likelihood = 0.0
        
        for i in range(n_samples):
            sample_likelihood = 0.0
            for k in range(self.n_components):
                log_prob = self._compute_log_probability(data[i:i+1], k)
                sample_likelihood += self.weights_[k] * np.exp(log_prob[0])
            
            log_likelihood += np.log(sample_likelihood + 1e-10)
        
        return log_likelihood
    
    def fit(self, data: np.ndarray, verbose: bool = True) -> 'TPGMMGaitModel':
        """
        Fit TP-GMM model to data
        
        Args:
            data: Shape (n_samples, 11) - [time, FR1_data(5D), FR2_data(5D)]
            verbose: Print training progress
            
        Returns:
            self
        """
        # Convert angles to radians for internal processing
        data_rad = self._convert_angles_to_radians(data)
        
        if verbose:
            print(f"Training TP-GMM with {self.n_components} components")
            print(f"Data shape: {data.shape}")
            print(f"Orientation unit: {self.orientation_unit}")
        
        # Initialize parameters
        self._initialize_parameters(data_rad)
        
        # EM algorithm
        prev_log_likelihood = -np.inf
        
        for iteration in range(self.max_iter):
            # E-step
            responsibilities = self._e_step(data_rad)
            
            # M-step
            self._m_step(data_rad, responsibilities)
            
            # Check convergence
            log_likelihood = self._compute_log_likelihood(data_rad)
            
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: Log-likelihood = {log_likelihood:.2f}")
            
            if np.abs(log_likelihood - prev_log_likelihood) < self.tol:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
            
            prev_log_likelihood = log_likelihood
        
        self.is_fitted_ = True
        
        if verbose:
            print(f"✓ TP-GMM training completed")
            print(f"Final log-likelihood: {log_likelihood:.2f}")
        
        return self
    
    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        """Predict component probabilities for new data"""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        data_rad = self._convert_angles_to_radians(data)
        return self._e_step(data_rad)
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict component assignments for new data"""
        proba = self.predict_proba(data)
        return np.argmax(proba, axis=1)
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to file"""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'weights_': self.weights_,
            'means_': self.means_,
            'covariances_': self.covariances_,
            'global_means_': self.global_means_,
            'global_covariances_': self.global_covariances_,
            'n_components': self.n_components,
            'orientation_unit': self.orientation_unit,
            'data_structure': self.data_structure,
            'is_fitted_': self.is_fitted_
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'TPGMMGaitModel':
        """Load trained model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create new instance
        model = cls(
            n_components=model_data['n_components'],
            orientation_unit=model_data['orientation_unit']
        )
        
        # Restore parameters
        model.weights_ = model_data['weights_']
        model.means_ = model_data['means_']
        model.covariances_ = model_data['covariances_']
        model.global_means_ = model_data['global_means_']
        model.global_covariances_ = model_data['global_covariances_']
        model.data_structure = model_data['data_structure']
        model.is_fitted_ = model_data['is_fitted_']
        
        print(f"✓ Model loaded from {filepath}")
        return model


class GaussianMixtureRegression:
    """
    Gaussian Mixture Regression for trajectory prediction
    Compatible with TP-GMM models
    """
    
    def __init__(self, tpgmm_model: TPGMMGaitModel):
        """
        Initialize GMR with a trained TP-GMM model
        
        Args:
            tpgmm_model: Fitted TP-GMM model
        """
        if not tpgmm_model.is_fitted_:
            raise ValueError("TP-GMM model must be fitted")
        
        self.tpgmm = tpgmm_model
        self.orientation_unit = tpgmm_model.orientation_unit
    
    def _gaussian_conditioning(self, mu: np.ndarray, sigma: np.ndarray, 
                             input_dims: List[int], output_dims: List[int], 
                             input_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gaussian conditioning: P(Y|X) where X=input_values
        
        Args:
            mu: Mean vector
            sigma: Covariance matrix
            input_dims: Indices of input dimensions
            output_dims: Indices of output dimensions
            input_values: Values for input dimensions
            
        Returns:
            Conditional mean and covariance
        """
        # Extract submatrices
        mu_x = mu[input_dims]
        mu_y = mu[output_dims]
        
        sigma_xx = sigma[np.ix_(input_dims, input_dims)]
        sigma_yy = sigma[np.ix_(output_dims, output_dims)]
        sigma_xy = sigma[np.ix_(input_dims, output_dims)]
        sigma_yx = sigma[np.ix_(output_dims, input_dims)]
        
        # Conditional parameters
        try:
            sigma_xx_inv = inv(sigma_xx)
            mu_cond = mu_y + sigma_yx @ sigma_xx_inv @ (input_values - mu_x)
            sigma_cond = sigma_yy - sigma_yx @ sigma_xx_inv @ sigma_xy
        except np.linalg.LinAlgError:
            # Fallback if matrix is singular
            mu_cond = mu_y
            sigma_cond = sigma_yy
        
        return mu_cond, sigma_cond
    
    def predict_trajectory(self, time_points: np.ndarray, 
                          frame_weights: Optional[List[float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict trajectory for given time points
        
        Args:
            time_points: Array of time points in [0, 1]
            frame_weights: Weights for combining frame predictions [frame1_weight, frame2_weight]
                          If None, uses equal weights
            
        Returns:
            trajectory: Shape (n_points, 10) - [FR1_data(5D), FR2_data(5D)]
            covariances: Shape (n_points, 10, 10) - Prediction uncertainties
        """
        if frame_weights is None:
            frame_weights = [0.5, 0.5]  # Equal weights
        
        n_points = len(time_points)
        trajectory = np.zeros((n_points, 10))  # 5D per frame × 2 frames
        covariances = np.zeros((n_points, 10, 10))
        
        for i, t in enumerate(time_points):
            # Collect predictions from each component and frame
            frame_predictions = []
            frame_covariances = []
            
            for f in range(2):  # Two frames
                # Collect component predictions for this frame
                component_mus = []
                component_sigmas = []
                component_weights = []
                
                for k in range(self.tpgmm.n_components):
                    # Use frame-specific parameters for conditioning
                    mu_k = self.tpgmm.means_[k, f]
                    sigma_k = self.tpgmm.covariances_[k, f]
                    weight_k = self.tpgmm.weights_[k]
                    
                    # For gait data, we don't condition on time at frame level
                    # Instead, we weight by time-based likelihood
                    time_factor = np.exp(-0.5 * (t - 0.5)**2 / 0.1)  # Gaussian around t=0.5
                    
                    component_mus.append(mu_k)
                    component_sigmas.append(sigma_k)
                    component_weights.append(weight_k * time_factor)
                
                # Normalize weights
                total_weight = sum(component_weights)
                if total_weight > 1e-10:
                    component_weights = [w / total_weight for w in component_weights]
                
                # Combine components for this frame
                frame_mu = np.zeros(5)
                frame_sigma = np.zeros((5, 5))
                
                for k in range(self.tpgmm.n_components):
                    w = component_weights[k]
                    frame_mu += w * component_mus[k]
                    frame_sigma += w * (component_sigmas[k] + np.outer(component_mus[k], component_mus[k]))
                
                frame_sigma -= np.outer(frame_mu, frame_mu)
                
                frame_predictions.append(frame_mu)
                frame_covariances.append(frame_sigma)
            
            # Combine frame predictions
            combined_mu = np.concatenate([
                frame_weights[0] * frame_predictions[0] + frame_weights[1] * frame_predictions[0],
                frame_weights[0] * frame_predictions[1] + frame_weights[1] * frame_predictions[1]
            ])
            
            # Combine covariances (simplified approach)
            combined_sigma = block_diag(
                frame_weights[0]**2 * frame_covariances[0] + frame_weights[1]**2 * frame_covariances[0],
                frame_weights[0]**2 * frame_covariances[1] + frame_weights[1]**2 * frame_covariances[1]
            )
            
            trajectory[i] = combined_mu
            covariances[i] = combined_sigma
        
        # Convert angles back to original units
        trajectory_output = self.tpgmm._convert_angles_from_radians(
            np.column_stack([np.zeros(n_points), trajectory])
        )[:, 1:]  # Remove time column
        
        return trajectory_output, covariances
    
    def predict_single_frame(self, time_points: np.ndarray, frame_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict trajectory for a single frame
        
        Args:
            time_points: Array of time points in [0, 1]
            frame_idx: Frame index (0 or 1)
            
        Returns:
            trajectory: Shape (n_points, 5) - Single frame data
            covariances: Shape (n_points, 5, 5) - Prediction uncertainties
        """
        n_points = len(time_points)
        trajectory = np.zeros((n_points, 5))
        covariances = np.zeros((n_points, 5, 5))
        
        for i, t in enumerate(time_points):
            # Collect predictions from each component
            component_mus = []
            component_sigmas = []
            component_weights = []
            
            for k in range(self.tpgmm.n_components):
                mu_k = self.tpgmm.means_[k, frame_idx]
                sigma_k = self.tpgmm.covariances_[k, frame_idx]
                weight_k = self.tpgmm.weights_[k]
                
                # Time-based weighting
                time_factor = np.exp(-0.5 * (t - 0.5)**2 / 0.1)
                
                component_mus.append(mu_k)
                component_sigmas.append(sigma_k)
                component_weights.append(weight_k * time_factor)
            
            # Normalize weights
            total_weight = sum(component_weights)
            if total_weight > 1e-10:
                component_weights = [w / total_weight for w in component_weights]
            
            # Combine components
            frame_mu = np.zeros(5)
            frame_sigma = np.zeros((5, 5))
            cd ..
            for k in range(self.tpgmm.n_components):
                w = component_weights[k]
                frame_mu += w * component_mus[k]
                frame_sigma += w * (component_sigmas[k] + np.outer(component_mus[k], component_mus[k]))
            
            frame_sigma -= np.outer(frame_mu, frame_mu)
            
            trajectory[i] = frame_mu
            covariances[i] = frame_sigma
        
        # Convert angles back to original units
        if frame_idx == 0:
            # FR1 data (orientation at index 4)
            trajectory_output = self.tpgmm._convert_angles_from_radians(
                np.column_stack([np.zeros(n_points), trajectory, np.zeros((n_points, 5))])
            )[:, 1:6]
        else:
            # FR2 data (orientation at index 4, but will be converted as index 9 in full data)
            trajectory_output = self.tpgmm._convert_angles_from_radians(
                np.column_stack([np.zeros(n_points), np.zeros((n_points, 5)), trajectory])
            )[:, 6:11]
        
        return trajectory_output, covariances


# Example usage and demonstration
def demonstrate_tpgmm_gait():
    """Demonstrate TP-GMM training and prediction on synthetic gait data"""
    print("=== TP-GMM Gait Model Demonstration ===\n")
    
    # Generate synthetic gait data
    np.random.seed(42)
    n_samples = 200
    
    # Time points
    time_points = np.linspace(0, 1, n_samples)
    
    # Generate synthetic trajectory data
    data = np.zeros((n_samples, 11))
    data[:, 0] = time_points  # Time
    
    # FR1 data (robot frame) - sinusoidal motion
    data[:, 1] = 0.5 * np.sin(2 * np.pi * time_points)  # x
    data[:, 2] = 0.3