import numpy as np
import matplotlib.pyplot as plt
import joblib
from scipy.spatial.transform import Rotation as R
from sklearn.mixture import GaussianMixture
from typing import Dict, List, Tuple, Optional
import os

class TPGMMTrajectoryRecovery:
    """
    Trajectory recovery system for TP-GMM gait models with frame adaptation
    """
    
    def __init__(self, model_file: str):
        """
        Initialize trajectory recovery system
        
        Args:
            model_file: Path to trained TP-GMM model file
        """
        self.model_data = self.load_model(model_file)
        self.gmm_model = self.model_data['gmm_model']
        
        # Extract model structure info
        self.data_structure = self.model_data['data_structure']
        self.frame_info = self.model_data['frame_info']
        
        # Dimensions
        self.dims_per_frame = self.frame_info['dims_per_frame']  # 5D per frame
        self.total_dim = self.data_structure['total_dim']        # 11D total (1 time + 10 spatial)
        
        print(f"✓ Loaded TP-GMM model with {self.gmm_model.n_components} components")
        print(f"  Data dimension: {self.total_dim}")
        print(f"  Dimensions per frame: {self.dims_per_frame}")
    
    def load_model(self, filename: str) -> Dict:
        """Load trained TP-GMM model"""
        try:
            model_data = joblib.load(filename)
            print(f"✓ Model loaded from: {filename}")
            return model_data
        except Exception as e:
            raise Exception(f"Error loading model: {e}")
    
    def gaussian_conditioning(self, mu: np.ndarray, sigma: np.ndarray, 
                            input_dims: List[int], output_dims: List[int], 
                            input_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gaussian conditioning: P(Y|X) where X are input_dims and Y are output_dims
        
        Args:
            mu: Mean vector
            sigma: Covariance matrix
            input_dims: Dimensions being conditioned on
            output_dims: Dimensions to predict
            input_values: Values for input dimensions
            
        Returns:
            Conditioned mean and covariance
        """
        # Extract submatrices
        mu_x = mu[input_dims]
        mu_y = mu[output_dims]
        
        sigma_xx = sigma[np.ix_(input_dims, input_dims)]
        sigma_yy = sigma[np.ix_(output_dims, output_dims)]
        sigma_xy = sigma[np.ix_(input_dims, output_dims)]
        sigma_yx = sigma[np.ix_(output_dims, input_dims)]
        
        # Conditioning formulas
        try:
            sigma_xx_inv = np.linalg.inv(sigma_xx + 1e-6 * np.eye(len(input_dims)))
        except np.linalg.LinAlgError:
            sigma_xx_inv = np.linalg.pinv(sigma_xx)
        
        mu_cond = mu_y + sigma_yx @ sigma_xx_inv @ (input_values - mu_x)
        sigma_cond = sigma_yy - sigma_yx @ sigma_xx_inv @ sigma_xy
        
        return mu_cond, sigma_cond
    
    def predict_trajectory_single_frame(self, time_points: np.ndarray, 
                                      frame_constraints: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict trajectory using standard GMM regression (single frame approach)
        
        Args:
            time_points: Array of time points [0, 1]
            frame_constraints: Optional constraints for specific dimensions
            
        Returns:
            Predicted trajectory and covariance
        """
        n_points = len(time_points)
        trajectory = np.zeros((n_points, self.total_dim - 1))  # Exclude time dimension
        covariances = np.zeros((n_points, self.total_dim - 1, self.total_dim - 1))
        
        for i, t in enumerate(time_points):
            # Prepare input (time + any constraints)
            input_dims = [0]  # Time dimension
            input_values = np.array([t])
            output_dims = list(range(1, self.total_dim))  # All spatial dimensions
            
            # Add frame constraints if provided
            if frame_constraints:
                for dim, value in frame_constraints.items():
                    if dim in output_dims:
                        input_dims.append(dim)
                        input_values = np.append(input_values, value)
                        output_dims.remove(dim)
            
            # GMM regression
            mu_pred = np.zeros(len(output_dims))
            sigma_pred = np.zeros((len(output_dims), len(output_dims)))
            
            for k in range(self.gmm_model.n_components):
                # Component weight
                weight = self.gmm_model.weights_[k]
                mu_k = self.gmm_model.means_[k]
                sigma_k = self.gmm_model.covariances_[k]
                
                # Gaussian conditioning for this component
                mu_k_cond, sigma_k_cond = self.gaussian_conditioning(
                    mu_k, sigma_k, input_dims, output_dims, input_values
                )
                
                # Compute responsibility (activation) of this component
                input_part = mu_k[input_dims]
                input_cov = sigma_k[np.ix_(input_dims, input_dims)]
                
                try:
                    diff = input_values - input_part
                    activation = weight * np.exp(-0.5 * diff.T @ np.linalg.inv(input_cov + 1e-6 * np.eye(len(input_dims))) @ diff)
                except:
                    activation = weight
                
                # Weighted combination
                mu_pred += activation * mu_k_cond
                sigma_pred += activation * (sigma_k_cond + np.outer(mu_k_cond, mu_k_cond))
            
            # Normalize by total weight
            total_weight = np.sum([self.gmm_model.weights_[k] for k in range(self.gmm_model.n_components)])
            mu_pred /= total_weight
            sigma_pred = sigma_pred / total_weight - np.outer(mu_pred, mu_pred)
            
            # Store results
            trajectory[i, :len(output_dims)] = mu_pred
            covariances[i, :len(output_dims), :len(output_dims)] = sigma_pred
        
        return trajectory, covariances
    
    def adapt_trajectory_to_new_frame(self, time_points: np.ndarray, 
                                    new_frame_params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adapt trajectory to new FR2 frame parameters using TP-GMM approach
        
        Args:
            time_points: Array of time points [0, 1]
            new_frame_params: New frame parameters
                - 'rotation': 2x2 rotation matrix or angle (radians)
                - 'translation': 2D translation vector
                - 'scale': Optional scaling factor (default: 1.0)
                
        Returns:
            Adapted trajectory in global coordinates and uncertainties
        """
        n_points = len(time_points)
        
        # Parse new frame parameters
        if 'rotation' in new_frame_params:
            if np.isscalar(new_frame_params['rotation']):
                # Angle provided
                angle = new_frame_params['rotation']
                A_new = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
            else:
                # Matrix provided
                A_new = np.array(new_frame_params['rotation'])
        else:
            A_new = np.eye(2)
        
        b_new = np.array(new_frame_params.get('translation', [0, 0]))
        scale = new_frame_params.get('scale', 1.0)
        
        print(f"Adapting to new frame:")
        print(f"  Rotation matrix:\n{A_new}")
        print(f"  Translation: {b_new}")
        print(f"  Scale: {scale}")
        
        # Initialize output arrays
        adapted_trajectory = np.zeros((n_points, 10))  # 5D FR1 + 5D FR2
        adapted_covariances = np.zeros((n_points, 10, 10))
        
        for i, t in enumerate(time_points):
            # Step 1: Get predictions from each frame
            
            # Frame 1 prediction (robot frame - typically identity)
            mu_fr1_list = []
            sigma_fr1_list = []
            weights_fr1 = []
            
            # Frame 2 prediction (task frame)
            mu_fr2_list = []
            sigma_fr2_list = []
            weights_fr2 = []
            
            for k in range(self.gmm_model.n_components):
                weight = self.gmm_model.weights_[k]
                mu_k = self.gmm_model.means_[k]
                sigma_k = self.gmm_model.covariances_[k]
                
                # Condition on time for Frame 1 (dimensions 1-5)
                input_dims = [0]  # Time
                fr1_output_dims = list(range(1, 6))  # FR1 dimensions
                input_values = np.array([t])
                
                mu_fr1_k, sigma_fr1_k = self.gaussian_conditioning(
                    mu_k, sigma_k, input_dims, fr1_output_dims, input_values
                )
                
                # Condition on time for Frame 2 (dimensions 6-10)
                fr2_output_dims = list(range(6, 11))  # FR2 dimensions
                
                mu_fr2_k, sigma_fr2_k = self.gaussian_conditioning(
                    mu_k, sigma_k, input_dims, fr2_output_dims, input_values
                )
                
                mu_fr1_list.append(mu_fr1_k)
                sigma_fr1_list.append(sigma_fr1_k)
                weights_fr1.append(weight)
                
                mu_fr2_list.append(mu_fr2_k)
                sigma_fr2_list.append(sigma_fr2_k)
                weights_fr2.append(weight)
            
            # Step 2: Combine GMM components for each frame
            # Frame 1 (robot frame)
            mu_fr1 = np.zeros(5)
            sigma_fr1 = np.zeros((5, 5))
            total_weight_fr1 = sum(weights_fr1)
            
            for k in range(self.gmm_model.n_components):
                w = weights_fr1[k] / total_weight_fr1
                mu_fr1 += w * mu_fr1_list[k]
                sigma_fr1 += w * (sigma_fr1_list[k] + np.outer(mu_fr1_list[k], mu_fr1_list[k]))
            
            sigma_fr1 -= np.outer(mu_fr1, mu_fr1)
            
            # Frame 2 (task frame) 
            mu_fr2 = np.zeros(5)
            sigma_fr2 = np.zeros((5, 5))
            total_weight_fr2 = sum(weights_fr2)
            
            for k in range(self.gmm_model.n_components):
                w = weights_fr2[k] / total_weight_fr2
                mu_fr2 += w * mu_fr2_list[k]
                sigma_fr2 += w * (sigma_fr2_list[k] + np.outer(mu_fr2_list[k], mu_fr2_list[k]))
            
            sigma_fr2 -= np.outer(mu_fr2, mu_fr2)
            
            # Step 3: Transform Frame 2 to new global coordinates
            # Transform position (first 2 dimensions)
            mu_fr2_pos_global = A_new @ (mu_fr2[:2] * scale) + b_new
            
            # Transform velocity (dimensions 2-4)
            mu_fr2_vel_global = A_new @ (mu_fr2[2:4] * scale)
            
            # Transform orientation (dimension 4)
            if A_new.shape[0] == 2:  # 2D rotation
                rotation_angle = np.arctan2(A_new[1, 0], A_new[0, 0])
                mu_fr2_orient_global = mu_fr2[4] + rotation_angle
            else:
                mu_fr2_orient_global = mu_fr2[4]
            
            # Combine transformed Frame 2 data
            mu_fr2_global = np.concatenate([
                mu_fr2_pos_global,
                mu_fr2_vel_global, 
                [mu_fr2_orient_global]
            ])
            
            # Transform covariance for Frame 2
            # Create transformation matrix for all 5 dimensions
            T_fr2 = np.eye(5)
            T_fr2[:2, :2] = A_new * scale  # Position transformation
            T_fr2[2:4, 2:4] = A_new * scale  # Velocity transformation
            # Orientation transformation is additive, so covariance unchanged
            
            sigma_fr2_global = T_fr2 @ sigma_fr2 @ T_fr2.T
            
            # Step 4: Combine predictions from both frames
            # For this implementation, we'll use a weighted combination
            # In practice, you might use Gaussian product or other fusion methods
            
            frame_weights = [0.3, 0.7]  # Weight for [FR1, FR2] - adjust based on task
            
            # Combine means
            combined_mu = np.concatenate([
                frame_weights[0] * mu_fr1 + frame_weights[1] * mu_fr2_global,
                mu_fr2_global  # Keep FR2 prediction for reference
            ])
            
            # Combine covariances (simplified approach)
            combined_sigma = np.block([
                [frame_weights[0]**2 * sigma_fr1 + frame_weights[1]**2 * sigma_fr2_global, 
                 np.zeros((5, 5))],
                [np.zeros((5, 5)), sigma_fr2_global]
            ])
            
            adapted_trajectory[i] = combined_mu
            adapted_covariances[i] = combined_sigma
        
        return adapted_trajectory, adapted_covariances
    
    def generate_trajectory_samples(self, time_points: np.ndarray,
                                  trajectory: np.ndarray, 
                                  covariances: np.ndarray,
                                  n_samples: int = 5) -> np.ndarray:
        """
        Generate trajectory samples from predicted distribution
        
        Args:
            time_points: Time points
            trajectory: Mean trajectory
            covariances: Trajectory covariances
            n_samples: Number of samples to generate
            
        Returns:
            Array of trajectory samples [n_samples x n_points x dims]
        """
        n_points, n_dims = trajectory.shape
        samples = np.zeros((n_samples, n_points, n_dims))
        
        for i in range(n_points):
            mean = trajectory[i]
            cov = covariances[i]
            
            # Add regularization to ensure positive definite
            cov_reg = cov + 1e-4 * np.eye(n_dims)
            
            # Generate samples
            for j in range(n_samples):
                try:
                    samples[j, i] = np.random.multivariate_normal(mean, cov_reg)
                except np.linalg.LinAlgError:
                    # Fallback to diagonal covariance
                    cov_diag = np.diag(np.diag(cov_reg))
                    samples[j, i] = np.random.multivariate_normal(mean, cov_diag)
        
        return samples
    
    def visualize_adapted_trajectory(self, time_points: np.ndarray,
                                   original_trajectory: np.ndarray,
                                   adapted_trajectory: np.ndarray,
                                   adapted_covariances: np.ndarray,
                                   new_frame_params: Dict,
                                   save_plots: bool = True):
        """
        Visualize original vs adapted trajectories
        
        Args:
            time_points: Time array
            original_trajectory: Original predicted trajectory
            adapted_trajectory: Adapted trajectory
            adapted_covariances: Trajectory uncertainties
            new_frame_params: New frame parameters used
            save_plots: Whether to save plots
        """
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        fig.suptitle('TP-GMM Trajectory Adaptation Results', fontsize=16)
        
        # Generate some samples for uncertainty visualization
        samples = self.generate_trajectory_samples(
            time_points, adapted_trajectory, adapted_covariances, n_samples=10
        )
        
        # Column titles
        axes[0, 0].set_title('Frame FR1 (Robot Frame)', fontsize=14, fontweight='bold')
        axes[0, 1].set_title('Frame FR2 (Adapted Task Frame)', fontsize=14, fontweight='bold')
        
        # Extract trajectories for each frame
        if original_trajectory.shape[1] == 10:
            orig_fr1 = original_trajectory[:, :5]
            orig_fr2 = original_trajectory[:, 5:]
        else:
            orig_fr1 = original_trajectory
            orig_fr2 = None
        
        adapt_fr1 = adapted_trajectory[:, :5]
        adapt_fr2 = adapted_trajectory[:, 5:]
        
        # Row 1: Position trajectories
        # FR1 Position
        axes[0, 0].plot(orig_fr1[:, 0], orig_fr1[:, 1], 'b-', linewidth=3, 
                       label='Original FR1', alpha=0.7)
        axes[0, 0].plot(adapt_fr1[:, 0], adapt_fr1[:, 1], 'r-', linewidth=3,
                       label='Adapted FR1')
        
        # Plot uncertainty samples
        for i in range(5):
            axes[0, 0].plot(samples[i, :, 0], samples[i, :, 1], 'r-', 
                           alpha=0.2, linewidth=1)
        
        axes[0, 0].set_xlabel('Position X')
        axes[0, 0].set_ylabel('Position Y')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # FR2 Position
        if orig_fr2 is not None:
            axes[0, 1].plot(orig_fr2[:, 0], orig_fr2[:, 1], 'b-', linewidth=3,
                           label='Original FR2', alpha=0.7)
        axes[0, 1].plot(adapt_fr2[:, 0], adapt_fr2[:, 1], 'g-', linewidth=3,
                       label='Adapted FR2')
        
        # Plot uncertainty samples for FR2
        for i in range(5):
            axes[0, 1].plot(samples[i, :, 5], samples[i, :, 6], 'g-',
                           alpha=0.2, linewidth=1)
        
        # Mark new frame origin
        b_new = new_frame_params.get('translation', [0, 0])
        axes[0, 1].scatter(b_new[0], b_new[1], color='red', s=100, 
                          marker='*', label='New Frame Origin')
        
        axes[0, 1].set_xlabel('Position X')
        axes[0, 1].set_ylabel('Position Y')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Row 2: Velocity trajectories
        # FR1 Velocity
        axes[1, 0].plot(orig_fr1[:, 2], orig_fr1[:, 3], 'b-', linewidth=3,
                       label='Original FR1', alpha=0.7)
        axes[1, 0].plot(adapt_fr1[:, 2], adapt_fr1[:, 3], 'r-', linewidth=3,
                       label='Adapted FR1')
        axes[1, 0].set_xlabel('Velocity X')
        axes[1, 0].set_ylabel('Velocity Y')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # FR2 Velocity
        if orig_fr2 is not None:
            axes[1, 1].plot(orig_fr2[:, 2], orig_fr2[:, 3], 'b-', linewidth=3,
                           label='Original FR2', alpha=0.7)
        axes[1, 1].plot(adapt_fr2[:, 2], adapt_fr2[:, 3], 'g-', linewidth=3,
                       label='Adapted FR2')
        axes[1, 1].set_xlabel('Velocity X')
        axes[1, 1].set_ylabel('Velocity Y')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Row 3: Orientation over time
        # FR1 Orientation
        axes[2, 0].plot(time_points, orig_fr1[:, 4], 'b-', linewidth=3,
                       label='Original FR1', alpha=0.7)
        axes[2, 0].plot(time_points, adapt_fr1[:, 4], 'r-', linewidth=3,
                       label='Adapted FR1')
        
        # Plot uncertainty bands
        std_orient_fr1 = np.sqrt(adapted_covariances[:, 4, 4])
        axes[2, 0].fill_between(time_points, 
                               adapt_fr1[:, 4] - std_orient_fr1,
                               adapt_fr1[:, 4] + std_orient_fr1,
                               alpha=0.2, color='red')
        
        axes[2, 0].set_xlabel('Time')
        axes[2, 0].set_ylabel('Orientation (rad)')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # FR2 Orientation
        if orig_fr2 is not None:
            axes[2, 1].plot(time_points, orig_fr2[:, 4], 'b-', linewidth=3,
                           label='Original FR2', alpha=0.7)
        axes[2, 1].plot(time_points, adapt_fr2[:, 4], 'g-', linewidth=3,
                       label='Adapted FR2')
        
        # Plot uncertainty bands
        std_orient_fr2 = np.sqrt(adapted_covariances[:, 9, 9])
        axes[2, 1].fill_between(time_points,
                               adapt_fr2[:, 4] - std_orient_fr2,
                               adapt_fr2[:, 4] + std_orient_fr2,
                               alpha=0.2, color='green')
        
        axes[2, 1].set_xlabel('Time')
        axes[2, 1].set_ylabel('Orientation (rad)')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            os.makedirs('plots', exist_ok=True)
            plt.savefig('plots/trajectory_adaptation.png', dpi=300, bbox_inches='tight')
            
        plt.show()
    
    def demonstrate_adaptation_scenarios(self, n_points: int = 100):
        """
        Demonstrate trajectory adaptation for multiple scenarios
        
        Args:
            n_points: Number of points in trajectory
        """
        time_points = np.linspace(0, 1, n_points)
        
        # Scenario 1: Original trajectory (reference frame unchanged)
        print("=== Scenario 1: Original Reference ===")
        original_traj, original_cov = self.predict_trajectory_single_frame(time_points)
        
        # Scenario 2: Translated target
        print("\n=== Scenario 2: Translated Target ===")
        new_frame_params_1 = {
            'translation': [0.5, 0.3],
            'rotation': 0.0
        }
        adapted_traj_1, adapted_cov_1 = self.adapt_trajectory_to_new_frame(
            time_points, new_frame_params_1
        )
        
        # Scenario 3: Rotated and translated target
        print("\n=== Scenario 3: Rotated + Translated Target ===")
        new_frame_params_2 = {
            'translation': [0.8, -0.2],
            'rotation': np.pi/4  # 45 degrees
        }
        adapted_traj_2, adapted_cov_2 = self.adapt_trajectory_to_new_frame(
            time_points, new_frame_params_2
        )
        
        # Scenario 4: Scaled, rotated, and translated
        print("\n=== Scenario 4: Scaled + Rotated + Translated Target ===")
        new_frame_params_3 = {
            'translation': [-0.3, 0.6],
            'rotation': -np.pi/6,  # -30 degrees
            'scale': 1.5
        }
        adapted_traj_3, adapted_cov_3 = self.adapt_trajectory_to_new_frame(
            time_points, new_frame_params_3
        )
        
        # Visualize all scenarios
        self.visualize_multiple_scenarios(
            time_points,
            [original_traj, adapted_traj_1, adapted_traj_2, adapted_traj_3],
            [new_frame_params_1, new_frame_params_2, new_frame_params_3]
        )
        
        return {
            'original': (original_traj, original_cov),
            'translated': (adapted_traj_1, adapted_cov_1),
            'rotated_translated': (adapted_traj_2, adapted_cov_2),
            'scaled_rotated_translated': (adapted_traj_3, adapted_cov_3)
        }
    
    def visualize_multiple_scenarios(self, time_points: np.ndarray,
                                   trajectories: List[np.ndarray],
                                   frame_params: List[Dict]):
        """
        Visualize multiple adaptation scenarios
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('TP-GMM Adaptation: Multiple Scenarios', fontsize=16)
        
        colors = ['blue', 'red', 'green', 'orange']
        labels = ['Original', 'Translated', 'Rot+Trans', 'Scale+Rot+Trans']
        
        # Plot FR1 position trajectories
        axes[0, 0].set_title('FR1 Position Trajectories')
        for i, (traj, color, label) in enumerate(zip(trajectories, colors, labels)):
            if traj.shape[1] >= 5:
                axes[0, 0].plot(traj[:, 0], traj[:, 1], color=color, 
                               linewidth=3, label=label, alpha=0.8)
        axes[0, 0].set_xlabel('Position X')
        axes[0, 0].set_ylabel('Position Y')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot FR2 position trajectories  
        axes[0, 1].set_title('FR2 Position Trajectories')
        for i, (traj, color, label) in enumerate(zip(trajectories, colors, labels)):
            if traj.shape[1] >= 10:
                axes[0, 1].plot(traj[:, 5], traj[:, 6], color=color,
                               linewidth=3, label=label, alpha=0.8)
        
        # Mark frame origins
        for i, params in enumerate(frame_params):
            if 'translation' in params:
                trans = params['translation']
                axes[0, 1].scatter(trans[0], trans[1], color=colors[i+1], 
                                 s=100, marker='*', alpha=0.8)
        
        axes[0, 1].set_xlabel('Position X')
        axes[0, 1].set_ylabel('Position Y')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot orientations over time
        axes[1, 0].set_title('FR1 Orientation Evolution')
        for i, (traj, color, label) in enumerate(zip(trajectories, colors, labels)):
            if traj.shape[1] >= 5:
                axes[1, 0].plot(time_points, traj[:, 4], color=color,
                               linewidth=3, label=label, alpha=0.8)
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Orientation (rad)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('FR2 Orientation Evolution')
        for i, (traj, color, label) in enumerate(zip(trajectories, colors, labels)):
            if traj.shape[1] >= 10:
                axes[1, 1].plot(time_points, traj[:, 9], color=color,
                               linewidth=3, label=label, alpha=0.8)
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Orientation (rad)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/multiple_adaptation_scenarios.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """
    Main demonstration of TP-GMM trajectory recovery and adaptation
    """
    print("=== TP-GMM Trajectory Recovery and Adaptation ===\n")
    
    # Initialize recovery system
    basepath = 'data/tpgmm_gait_model'
    especific_path = '#39_16'
    extension = '.pkl'
    model_file = f'{basepath}{especific_path}{extension}'
    
    try:
        recovery_system = TPGMMTrajectoryRecovery(model_file)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have trained the model using tpgmm_gait_training.py first")
        return
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    print("\n=== Demonstrating Multiple Adaptation Scenarios ===")
    
    # Run multiple scenario demonstration
    results = recovery_system.demonstrate_adaptation_scenarios(n_points=50)
    
    print("\n=== Individual Scenario Analysis ===")
    
    # Example 1: Simple translation
    print("\n1. Testing simple translation adaptation...")
    time_points = np.linspace(0, 1, 50)
    
    # Original trajectory
    original_traj, original_cov = recovery_system.predict_trajectory_single_frame(time_points)
    
    # Adapted trajectory with translation
    new_frame_params = {
        'translation': [0.4, 0.2],
        'rotation': 0.0
    }
    
    adapted_traj, adapted_cov = recovery_system.adapt_trajectory_to_new_frame(
        time_points, new_frame_params
    )
    
    # Visualize adaptation
    recovery_system.visualize_adapted_trajectory(
        time_points, original_traj, adapted_traj, adapted_cov, new_frame_params
    )
    
    print("\n=== Advanced Scenario: Dynamic Frame Changes ===")
    
    # Example 2: Time-varying frame adaptation
    dynamic_results = recovery_system.demonstrate_dynamic_adaptation(time_points)
    
    print("\n=== Trajectory Analysis Summary ===")
    recovery_system.print_adaptation_summary(results)
    
    print("\n✓ TP-GMM trajectory recovery demonstration complete!")
    print("Check the 'plots' folder for visualizations.")

def demonstrate_dynamic_adaptation(self, time_points: np.ndarray):
    """
    Demonstrate adaptation with time-varying frame parameters
    
    Args:
        time_points: Array of time points
        
    Returns:
        Dictionary with dynamic adaptation results
    """
    print("Demonstrating dynamic frame adaptation...")
    
    n_points = len(time_points)
    dynamic_trajectory = np.zeros((n_points, 10))
    
    # Create time-varying frame parameters
    for i, t in enumerate(time_points):
        # Frame parameters that change over time
        translation = [0.3 * np.sin(2 * np.pi * t), 0.2 * np.cos(2 * np.pi * t)]
        rotation = 0.5 * np.sin(np.pi * t)  # Oscillating rotation
        
        new_frame_params = {
            'translation': translation,
            'rotation': rotation
        }
        
        # Get adapted trajectory for this time point
        single_point_time = np.array([t])
        adapted_point, _ = self.adapt_trajectory_to_new_frame(
            single_point_time, new_frame_params
        )
        
        dynamic_trajectory[i] = adapted_point[0]
    
    # Visualize dynamic adaptation
    self.visualize_dynamic_adaptation(time_points, dynamic_trajectory)
    
    return dynamic_trajectory

def visualize_dynamic_adaptation(self, time_points: np.ndarray, 
                               dynamic_trajectory: np.ndarray):
    """
    Visualize dynamic adaptation results
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Dynamic Frame Adaptation', fontsize=16)
    
    # Extract FR1 and FR2 data
    fr1_data = dynamic_trajectory[:, :5]
    fr2_data = dynamic_trajectory[:, 5:]
    
    # Plot position trajectories
    axes[0, 0].plot(fr1_data[:, 0], fr1_data[:, 1], 'b-', linewidth=3, label='FR1')
    axes[0, 0].plot(fr2_data[:, 0], fr2_data[:, 1], 'r-', linewidth=3, label='FR2')
    axes[0, 0].set_xlabel('Position X')
    axes[0, 0].set_ylabel('Position Y')
    axes[0, 0].set_title('Position Trajectories')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot velocity trajectories
    axes[0, 1].plot(fr1_data[:, 2], fr1_data[:, 3], 'b-', linewidth=3, label='FR1')
    axes[0, 1].plot(fr2_data[:, 2], fr2_data[:, 3], 'r-', linewidth=3, label='FR2')
    axes[0, 1].set_xlabel('Velocity X')
    axes[0, 1].set_ylabel('Velocity Y')
    axes[0, 1].set_title('Velocity Trajectories')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot orientations over time
    axes[0, 2].plot(time_points, fr1_data[:, 4], 'b-', linewidth=3, label='FR1')
    axes[0, 2].plot(time_points, fr2_data[:, 4], 'r-', linewidth=3, label='FR2')
    axes[0, 2].set_xlabel('Time')
    axes[0, 2].set_ylabel('Orientation (rad)')
    axes[0, 2].set_title('Orientation Evolution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot frame parameters over time
    frame_translations_x = [0.3 * np.sin(2 * np.pi * t) for t in time_points]
    frame_translations_y = [0.2 * np.cos(2 * np.pi * t) for t in time_points]
    frame_rotations = [0.5 * np.sin(np.pi * t) for t in time_points]
    
    axes[1, 0].plot(time_points, frame_translations_x, 'g-', linewidth=2, label='Translation X')
    axes[1, 0].plot(time_points, frame_translations_y, 'm-', linewidth=2, label='Translation Y')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Translation')
    axes[1, 0].set_title('Frame Translation Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(time_points, frame_rotations, 'orange', linewidth=2)
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Rotation (rad)')
    axes[1, 1].set_title('Frame Rotation Over Time')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 3D trajectory visualization
    from mpl_toolkits.mplot3d import Axes3D
    ax3d = fig.add_subplot(2, 3, 6, projection='3d')
    ax3d.plot(fr1_data[:, 0], fr1_data[:, 1], time_points, 'b-', linewidth=3, label='FR1')
    ax3d.plot(fr2_data[:, 0], fr2_data[:, 1], time_points, 'r-', linewidth=3, label='FR2')
    ax3d.set_xlabel('Position X')
    ax3d.set_ylabel('Position Y')
    ax3d.set_zlabel('Time')
    ax3d.set_title('3D Trajectory Evolution')
    ax3d.legend()
    
    plt.tight_layout()
    plt.savefig('plots/dynamic_adaptation.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_adaptation_summary(self, results: Dict):
    """
    Print summary of adaptation results
    """
    print("📊 Adaptation Summary:")
    print("-" * 50)
    
    for scenario_name, (trajectory, covariance) in results.items():
        print(f"\n{scenario_name.upper().replace('_', ' ')}:")
        
        # Calculate trajectory statistics
        fr1_pos_range = np.ptp(trajectory[:, :2], axis=0)  # Range for FR1 position
        fr2_pos_range = np.ptp(trajectory[:, 5:7], axis=0)  # Range for FR2 position
        
        print(f"  FR1 Position Range: X={fr1_pos_range[0]:.3f}, Y={fr1_pos_range[1]:.3f}")
        print(f"  FR2 Position Range: X={fr2_pos_range[0]:.3f}, Y={fr2_pos_range[1]:.3f}")
        
        # Calculate average uncertainty
        avg_uncertainty = np.mean(np.sqrt(np.diag(covariance.mean(axis=0))))
        print(f"  Average Uncertainty: {avg_uncertainty:.4f}")
        
        # Calculate smoothness (velocity changes)
        if trajectory.shape[0] > 1:
            fr1_vel_smoothness = np.mean(np.diff(trajectory[:, 2:4], axis=0)**2)
            fr2_vel_smoothness = np.mean(np.diff(trajectory[:, 7:9], axis=0)**2)
            print(f"  FR1 Velocity Smoothness: {fr1_vel_smoothness:.4f}")
            print(f"  FR2 Velocity Smoothness: {fr2_vel_smoothness:.4f}")

# Add these methods to the TPGMMTrajectoryRecovery class
TPGMMTrajectoryRecovery.demonstrate_dynamic_adaptation = demonstrate_dynamic_adaptation
TPGMMTrajectoryRecovery.visualize_dynamic_adaptation = visualize_dynamic_adaptation
TPGMMTrajectoryRecovery.print_adaptation_summary = print_adaptation_summary

# Example usage for custom adaptation
def example_custom_adaptation():
    """
    Example of how to use the recovery system for custom scenarios
    """
    print("\n=== Custom Adaptation Example ===")
    
    # Load the recovery system
    basepath = 'data/tpgmm_gait_model'
    especific_path = '#39_16'
    extension = '.pkl'
    recovery_system = TPGMMTrajectoryRecovery(f'{basepath}{especific_path}{extension}')
    
    # Define custom scenario parameters
    time_points = np.linspace(0, 1, 60)  # 60 points for smooth trajectory
    
    # Custom frame parameters for your specific application
    custom_frame_params = {
        'translation': [0.6, -0.4],  # Move target 0.6m right, 0.4m down
        'rotation': np.pi/3,         # Rotate 60 degrees
        'scale': 1.2                 # Scale by 20%
    }
    
    print(f"Custom frame parameters:")
    print(f"  Translation: {custom_frame_params['translation']}")
    print(f"  Rotation: {custom_frame_params['rotation']:.2f} rad ({np.degrees(custom_frame_params['rotation']):.1f}°)")
    print(f"  Scale: {custom_frame_params['scale']}")
    
    # Generate adapted trajectory
    adapted_trajectory, uncertainties = recovery_system.adapt_trajectory_to_new_frame(
        time_points, custom_frame_params
    )
    
    # Generate samples for uncertainty visualization
    samples = recovery_system.generate_trajectory_samples(
        time_points, adapted_trajectory, uncertainties, n_samples=8
    )
    
    print(f"\n✓ Generated adapted trajectory with {len(adapted_trajectory)} points")
    print(f"✓ Generated {len(samples)} uncertainty samples")
    
    # You can now use 'adapted_trajectory' for your robot control
    # The trajectory contains: [FR1_data | FR2_data] where each is 5D
    
    return adapted_trajectory, uncertainties, samples

if __name__ == "__main__":
    main()
    
    # Run custom example
    try:
        example_custom_adaptation()
    except Exception as e:
        print(f"Custom example failed: {e}")
        print("Make sure the model file exists first.")