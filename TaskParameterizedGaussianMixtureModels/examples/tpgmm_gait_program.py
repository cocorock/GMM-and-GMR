#!/usr/bin/env python3
"""
TPGMM for Gait Data with Transformed Reference Frame

This program demonstrates the application of Task Parameterized Gaussian Mixture Models (TPGMM)
to gait data analysis with multiple frames of reference.
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List

np.set_printoptions(threshold=sys.maxsize)
# Add parent directory to path for imports
sys.path.append("/".join(sys.path[0].split("/")[:-1]))

from tpgmm.tpgmm.tpgmm import TPGMM
from tpgmm.gmr.gmr import GaussianMixtureRegression
from tpgmm.utils.plot.plot import plot_trajectories, plot_ellipsoids, scatter


class GaitDataProcessor:
    """Process gait data for TPGMM analysis."""
    
    def __init__(self, data_path: str):
        """
        Initialize the processor with data path.
        
        Args:
            data_path: Path to the JSON file containing gait data
        """
        self.data_path = data_path
        self.data = None
        self.trajectories_fr1 = None
        self.trajectories_fr2 = None
        self.scaler1 = StandardScaler()
        self.scaler2 = StandardScaler()
        
    def load_data(self) -> None:
        """Load gait data from JSON file."""
        print(f"Loading data from {self.data_path}...")
        with open(self.data_path, "r") as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} demonstrations")
        
    def prepare_trajectories(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare trajectory data from both frames of reference.
        
        Returns:
            Tuple of (trajectories_fr1, trajectories_fr2)
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        print("\nPreparing trajectories...")
        trajectories_fr1 = []
        trajectories_fr2 = []

        for demo in self.data:
            # For FR1, stack spatial features first, then time
            traj1 = np.hstack([
                demo['ankle_pos_FR1'], 
                demo['ankle_pos_FR1_velocity'], 
                # demo['ankle_orientation_FR1'],  # Commented out as in original
                demo['time']
            ])
            trajectories_fr1.append(traj1)
            
            # For FR2, stack spatial features first, then time
            traj2 = np.hstack([
                demo['ankle_pos_FR2'], 
                demo['ankle_pos_FR2_velocity'], 
                # demo['ankle_orientation_FR2'],  # Commented out as in original
                demo['time']
            ])
            trajectories_fr2.append(traj2)

        # Convert lists to numpy arrays
        self.trajectories_fr1 = np.array(trajectories_fr1)
        self.trajectories_fr2 = np.array(trajectories_fr2)
        
        print(f"Trajectory shape: {self.trajectories_fr1.shape}")
        
        return self.trajectories_fr1, self.trajectories_fr2
    
    def prepare_for_tpgmm(self) -> np.ndarray:
        """
        Reshape and scale data for TPGMM fitting.
        
        Returns:
            Scaled trajectories ready for TPGMM
        """
        if self.trajectories_fr1 is None or self.trajectories_fr2 is None:
            raise ValueError("Trajectories not prepared. Call prepare_trajectories() first.")
            
        print("\nReshaping data for TPGMM...")
        # Stack the data for the two frames
        num_trajectories, num_samples, num_features = self.trajectories_fr1.shape
        reshaped_trajectories = np.stack([self.trajectories_fr1, self.trajectories_fr2], axis=0)
        
        # Reshape to (num_frames, num_trajectories * num_samples, num_features)
        reshaped_trajectories = reshaped_trajectories.reshape(2, num_trajectories * num_samples, num_features)
        
        print("Applying feature scaling...")
        # Fit and transform the data for each frame
        scaled_fr1 = self.scaler1.fit_transform(reshaped_trajectories[0])
        scaled_fr2 = self.scaler2.fit_transform(reshaped_trajectories[1])
        
        # Stack the scaled data back together
        scaled_trajectories = np.stack([scaled_fr1, scaled_fr2], axis=0)
        
        print(f"Scaled trajectories shape: {scaled_trajectories.shape}")
        
        return scaled_trajectories
    
    def check_data_validity(self, data: np.ndarray) -> bool:
        """
        Check if data contains NaN or Inf values.
        
        Args:
            data: Data to check
            
        Returns:
            True if data is valid, False otherwise
        """
        if np.isnan(data).any() or np.isinf(data).any():
            print('Data contains NaN or Inf values. Cannot proceed with fitting.')
            print(f'Number of NaNs: {np.isnan(data).sum()}')
            print(f'Number of Infs: {np.isinf(data).sum()}')
            return False
        else:
            print('Data check passed.')
            return True


class TPGMMGaitAnalyzer:
    """Analyze gait data using TPGMM."""
    
    def __init__(self, n_components: int = 17, reg_factor: float = 1e-4):
        """
        Initialize the analyzer.
        
        Args:
            n_components: Number of Gaussian components
            reg_factor: Regularization factor for numerical stability
        """
        self.n_components = n_components
        self.reg_factor = reg_factor
        self.tpgmm = None
        self.gmr = None
        
    def fit_model(self, scaled_trajectories: np.ndarray) -> None:
        """
        Fit TPGMM to the scaled trajectory data.
        
        Args:
            scaled_trajectories: Scaled trajectory data
        """
        print(f"\nInitializing TPGMM with {self.n_components} components...")
        self.tpgmm = TPGMM(
            n_components=self.n_components, 
            reg_factor=self.reg_factor, 
            verbose=True
        )
        
        print("Fitting the model...")
        self.tpgmm.fit(scaled_trajectories)
        print("Model fitting complete!")
        
    def setup_gmr(self) -> None:
        """Setup Gaussian Mixture Regression from fitted TPGMM."""
        if self.tpgmm is None:
            raise ValueError("TPGMM not fitted. Call fit_model() first.")
            
        print("\nSetting up Gaussian Mixture Regression...")
        self.gmr = GaussianMixtureRegression(self.tpgmm)
        
    def predict_trajectories(self, trajectories: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict trajectories using original and transformed reference frames.
        
        Args:
            trajectories: Original trajectory data
            
        Returns:
            Tuple of (mu, sigma, mu_transformed, sigma_transformed)
        """
        if self.gmr is None:
            raise ValueError("GMR not setup. Call setup_gmr() first.")
            
        print("\nPredicting trajectories...")
        
        # Original reference frames (start and end of the first demonstration)
        p_in = np.array([
            trajectories[0, 0, 1:3], 
            trajectories[0, -1, 3:5]
        ])
        A_in = np.array([np.eye(2), np.eye(2)])
        
        # Transformed reference frame for FR2
        p_in_transformed = np.array([
            trajectories[0, 0, 1:3], 
            trajectories[0, -1, 3:5] + np.array([0.5, 0.5])
        ])
        A_in_transformed = np.array([np.eye(2), np.eye(2)])
        
        # GMR with original frames
        print("Predicting with original reference frames...")
        mu, sigma = self.gmr.predict(p_in, A_in)
        
        # GMR with transformed frames
        print("Predicting with transformed reference frames...")
        mu_transformed, sigma_transformed = self.gmr.predict(p_in_transformed, A_in_transformed)
        
        return mu, sigma, mu_transformed, sigma_transformed


def visualize_results(trajectories: np.ndarray, mu: np.ndarray, mu_transformed: np.ndarray) -> None:
    """
    Visualize the original trajectories and GMR predictions.
    
    Args:
        trajectories: Original trajectory data
        mu: GMR predictions with original frames
        mu_transformed: GMR predictions with transformed frames
    """
    print("\nCreating visualization...")
    
    # Extract spatial components for plotting
    fig, ax = plot_trajectories(
        trajectories[:, :, [0, 1, 2]], 
        color='b', 
        label='Original Trajectories'
    )
    
    plot_trajectories(
        mu, 
        fig=fig, 
        ax=ax, 
        color='r', 
        label='GMR Original'
    )
    
    plot_trajectories(
        mu_transformed, 
        fig=fig, 
        ax=ax, 
        color='g', 
        label='GMR Transformed'
    )
    
    ax.legend()
    plt.title('TPGMM Gait Analysis Results')
    plt.show()


def main():
    """Main function to run the gait analysis."""
    
    # Configuration
    data_path = "data/new_processed_gait_data#39_16.json"
    n_components = 17
    reg_factor = 1e-4
    
    # Initialize processor
    processor = GaitDataProcessor(data_path)
    
    # Load and prepare data
    processor.load_data()
    trajectories_fr1, trajectories_fr2 = processor.prepare_trajectories()
    scaled_trajectories = processor.prepare_for_tpgmm()
    
    # Check data validity
    if not processor.check_data_validity(scaled_trajectories):
        print("Data validation failed. Exiting...")
        return
    
    # Initialize and fit TPGMM
    analyzer = TPGMMGaitAnalyzer(n_components=n_components, reg_factor=reg_factor)
    analyzer.fit_model(scaled_trajectories)
    analyzer.setup_gmr()
    
    # Predict trajectories
    mu, sigma, mu_transformed, sigma_transformed = analyzer.predict_trajectories(trajectories_fr1)
    
    # Visualize results
    visualize_results(trajectories_fr1, mu, mu_transformed)
    
    print("\nGait analysis complete!")


if __name__ == "__main__":
    main()