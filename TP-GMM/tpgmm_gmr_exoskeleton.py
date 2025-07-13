#!/usr/bin/env python3
"""
TP-GMM Gaussian Mixture Regression (GMR) for Lower Limb Exoskeleton Control

This script performs Gaussian Mixture Regression on a pre-trained TP-GMM model
for gait dynamics recovery. The model uses 2 frames of reference:
- FR1: Hip frame (exoskeleton coordinate system)
- FR2: Global frame (world coordinate system)

Both frames store true ankle positions after inverse transformation from
the original joint angle representations.

Author: Generated for Exoskeleton Control Application
"""

import numpy as np
import matplotlib.pyplot as plt
import joblib
from typing import Dict, Tuple, List, Optional, Union
from scipy.stats import multivariate_normal
import warnings
warnings.filterwarnings('ignore')


class TPGMMGaussianMixtureRegression:
    """
    Gaussian Mixture Regression for TP-GMM gait models
    
    Performs regression to recover gait dynamics from time-indexed input,
    supporting both single and multi-frame predictions.
    """
    
    def __init__(self, model_path: str, info_path: Optional[str] = None):
        """
        Initialize GMR with pre-trained TP-GMM model
        
        Args:
            model_path: Path to the .pkl model file
            info_path: Optional path to model info file
        """
        self.model_data = self.load_model(model_path)
        self.gmm = self.model_data['gmm_model']
        self.data_structure = self.model_data['data_structure']
        self.frame_info = self.model_data['frame_info']
        
        # Extract dimension information
        self.n_components = self.model_data['n_components']
        self.total_dim = self.data_structure['total_dim']
        self.time_dim = self.data_structure['time_dim']
        
        # Frame dimensions
        self.frame1_dims = self.data_structure['frame1_dims']  # [1,2,3,4,5]
        self.frame2_dims = self.data_structure['frame2_dims']  # [6,7,8,9,10]
        
        # Feature mappings
        self.pos_dims = self.data_structure['position_dims']
        self.vel_dims = self.data_structure['velocity_dims']
        self.orient_dims = self.data_structure['orientation_dims']
        
        if info_path:
            self.load_model_info(info_path)
            
        print(f"✓ TP-GMM GMR initialized:")
        print(f"  Components: {self.n_components}")
        print(f"  Total dimensions: {self.total_dim}")
        print(f"  Coordinate system: {self.frame_info['coordinate_system']}")
        print(f"  Frames: {self.frame_info['num_frames']}")
    
    def load_model(self, model_path: str) -> Dict:
        """Load the pre-trained TP-GMM model"""
        try:
            model_data = joblib.load(model_path)
            print(f"✓ Model loaded from: {model_path}")
            return model_data
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def load_model_info(self, info_path: str):
        """Load and display model information"""
        try:
            with open(info_path, 'r') as f:
                info = f.read()
            print("\n=== Model Information ===")
            print(info)
        except Exception as e:
            print(f"Warning: Could not load model info: {e}")
    
    def perform_gmr(self, input_time: Union[float, np.ndarray], 
                   input_dims: List[int], output_dims: List[int],
                   frame_preference: str = 'both') -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Gaussian Mixture Regression
        
        Args:
            input_time: Time value(s) for regression (0.0 to 1.0)
            input_dims: List of input dimension indices
            output_dims: List of output dimension indices  
            frame_preference: 'fr1', 'fr2', or 'both'
            
        Returns:
            Tuple of (predicted_output, prediction_variance)
        """
        # Handle single time input
        if np.isscalar(input_time):
            input_time = np.array([input_time])
        
        # Ensure time is in valid range
        input_time = np.clip(input_time, 0.0, 1.0)
        
        n_points = len(input_time)
        n_output_dims = len(output_dims)
        
        # Initialize output arrays
        predicted_output = np.zeros((n_points, n_output_dims))
        prediction_variance = np.zeros((n_points, n_output_dims, n_output_dims))
        
        # Process each time point
        for i, t in enumerate(input_time):
            # Create input vector
            input_vector = np.zeros(self.total_dim)
            input_vector[self.time_dim] = t
            
            # Compute component responsibilities
            responsibilities = self._compute_responsibilities(input_vector, input_dims)
            
            # Perform regression
            mu_pred, sigma_pred = self._gmr_prediction(
                input_vector, input_dims, output_dims, responsibilities
            )
            
            predicted_output[i] = mu_pred
            prediction_variance[i] = sigma_pred
        
        # Return single prediction if single input
        if n_points == 1:
            return predicted_output[0], prediction_variance[0]
        
        return predicted_output, prediction_variance
    
    def _compute_responsibilities(self, input_vector: np.ndarray, 
                                input_dims: List[int]) -> np.ndarray:
        """Compute posterior responsibilities for each Gaussian component"""
        responsibilities = np.zeros(self.n_components)
        
        for k in range(self.n_components):
            # Extract mean and covariance for input dimensions
            mu_input = self.gmm.means_[k][input_dims]
            sigma_input = self.gmm.covariances_[k][np.ix_(input_dims, input_dims)]
            
            # Add small regularization to avoid numerical issues
            sigma_input += np.eye(len(input_dims)) * 1e-6
            
            try:
                # Compute likelihood
                likelihood = multivariate_normal.pdf(
                    input_vector[input_dims], mu_input, sigma_input
                )
                responsibilities[k] = self.gmm.weights_[k] * likelihood
            except np.linalg.LinAlgError:
                # Handle singular matrices
                responsibilities[k] = 1e-10
        
        # Normalize responsibilities
        total_resp = np.sum(responsibilities)
        if total_resp > 1e-10:
            responsibilities /= total_resp
        else:
            responsibilities = np.ones(self.n_components) / self.n_components
            
        return responsibilities
    
    def _gmr_prediction(self, input_vector: np.ndarray, input_dims: List[int],
                       output_dims: List[int], responsibilities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform the actual GMR prediction"""
        n_output = len(output_dims)
        
        # Initialize prediction
        mu_pred = np.zeros(n_output)
        sigma_pred = np.zeros((n_output, n_output))
        
        for k in range(self.n_components):
            if responsibilities[k] < 1e-10:
                continue
            
            # Extract component parameters
            mu_k = self.gmm.means_[k]
            sigma_k = self.gmm.covariances_[k]
            
            # Partition mean vector
            mu_input = mu_k[input_dims]
            mu_output = mu_k[output_dims]
            
            # Partition covariance matrix
            sigma_ii = sigma_k[np.ix_(input_dims, input_dims)]
            sigma_oo = sigma_k[np.ix_(output_dims, output_dims)]
            sigma_io = sigma_k[np.ix_(input_dims, output_dims)]
            sigma_oi = sigma_k[np.ix_(output_dims, input_dims)]
            
            # Add regularization
            sigma_ii += np.eye(len(input_dims)) * 1e-6
            
            try:
                # Compute conditional parameters
                sigma_ii_inv = np.linalg.inv(sigma_ii)
                
                # Conditional mean
                mu_cond = mu_output + sigma_oi @ sigma_ii_inv @ (input_vector[input_dims] - mu_input)
                
                # Conditional covariance  
                sigma_cond = sigma_oo - sigma_oi @ sigma_ii_inv @ sigma_io
                
                # Accumulate weighted predictions
                mu_pred += responsibilities[k] * mu_cond
                sigma_pred += responsibilities[k] * (sigma_cond + np.outer(mu_cond, mu_cond))
                
            except np.linalg.LinAlgError:
                # Handle numerical issues
                mu_pred += responsibilities[k] * mu_output
                sigma_pred += responsibilities[k] * sigma_oo
        
        # Compute final covariance
        sigma_pred -= np.outer(mu_pred, mu_pred)
        
        # Ensure positive definite covariance
        sigma_pred += np.eye(n_output) * 1e-6
        
        return mu_pred, sigma_pred
    
    def predict_ankle_trajectory(self, time_vector: np.ndarray, 
                               frame: str = 'fr1') -> Dict[str, np.ndarray]:
        """
        Predict complete ankle trajectory (position, velocity, orientation)
        
        Args:
            time_vector: Array of time values (0.0 to 1.0) 
            frame: 'fr1' (hip frame) or 'fr2' (global frame)
            
        Returns:
            Dictionary with 'position', 'velocity', 'orientation', and 'variance'
        """
        # Ensure time_vector is array
        if np.isscalar(time_vector):
            time_vector = np.array([time_vector])
        
        if frame.lower() == 'fr1':
            pos_dims = self.pos_dims['frame1_true']      # [1, 2]
            vel_dims = self.vel_dims['frame1_true']      # [3, 4] 
            orient_dims = self.orient_dims['frame1_true'] # [5]
        elif frame.lower() == 'fr2':
            pos_dims = self.pos_dims['frame2_true']      # [6, 7]
            vel_dims = self.vel_dims['frame2_true']      # [8, 9]
            orient_dims = self.orient_dims['frame2_true'] # [10]
        else:
            raise ValueError("Frame must be 'fr1' or 'fr2'")
        
        # Predict position
        pos_pred, pos_var = self.perform_gmr(
            time_vector, [self.time_dim], pos_dims
        )
        
        # Predict velocity
        vel_pred, vel_var = self.perform_gmr(
            time_vector, [self.time_dim], vel_dims
        )
        
        # Predict orientation
        orient_pred, orient_var = self.perform_gmr(
            time_vector, [self.time_dim], orient_dims
        )
        
        # Ensure arrays have consistent shapes
        if len(time_vector) == 1:
            # Single point prediction - ensure 2D arrays
            if pos_pred.ndim == 1:
                pos_pred = pos_pred.reshape(1, -1)
            if vel_pred.ndim == 1:
                vel_pred = vel_pred.reshape(1, -1)
            if np.isscalar(orient_pred):
                orient_pred = np.array([orient_pred])
        
        return {
            'position': pos_pred,
            'velocity': vel_pred, 
            'orientation': orient_pred,
            'position_variance': pos_var,
            'velocity_variance': vel_var,
            'orientation_variance': orient_var,
            'time': time_vector
        }
    
    def predict_joint_angles(self, time_vector: np.ndarray,
                           hip_position: np.ndarray = None,
                           hip_orientation: float = 0.0) -> Dict[str, np.ndarray]:
        """
        Convert ankle predictions to joint angles and velocities for exoskeleton control
        
        Args:
            time_vector: Array of time values
            hip_position: Hip position [x, y] in global frame
            hip_orientation: Hip orientation in global frame
            
        Returns:
            Dictionary with joint angles, velocities and ankle positions
        """
        # Ensure time_vector is array
        if np.isscalar(time_vector):
            time_vector = np.array([time_vector])
        
        # Get ankle trajectory in hip frame (FR1)
        ankle_traj = self.predict_ankle_trajectory(time_vector, frame='fr1')
        
        # Extract positions and velocities - handle both single point and multiple points
        ankle_pos = ankle_traj['position']
        ankle_vel = ankle_traj['velocity']
        
        if ankle_pos.ndim == 1:
            ankle_pos = ankle_pos.reshape(1, -1)  # Make it [1, 2] for single point
        if ankle_vel.ndim == 1:
            ankle_vel = ankle_vel.reshape(1, -1)  # Make it [1, 2] for single point
        
        # Convert to joint angles and velocities (simplified 2-link model)
        # Assuming hip at origin, links pointing down when theta=0
        L1 = 0.39  # Thigh length (m)
        L2 = 0.413  # Shank length (m)
        
        n_points = ankle_pos.shape[0]
        theta1 = np.zeros(n_points)     # Hip angle
        theta2 = np.zeros(n_points)     # Knee angle
        theta1_dot = np.zeros(n_points) # Hip angular velocity
        theta2_dot = np.zeros(n_points) # Knee angular velocity
        
        for i in range(n_points):
            x, y = ankle_pos[i, 0], ankle_pos[i, 1]
            vx, vy = ankle_vel[i, 0], ankle_vel[i, 1]
            
            # Distance from hip to ankle
            r = np.sqrt(x**2 + y**2)
            
            # Clamp to workspace
            r = np.clip(r, abs(L1-L2) + 0.01, L1+L2 - 0.01)
            
            # Knee angle using law of cosines
            cos_theta2 = (L1**2 + L2**2 - r**2) / (2*L1*L2)
            cos_theta2 = np.clip(cos_theta2, -1, 1)
            theta2[i] = np.pi - np.arccos(cos_theta2)  # Knee flexion
            
            # Hip angle
            alpha = np.arctan2(x, -y)  # Angle to ankle from vertical
            beta = np.arctan2(L2*np.sin(theta2[i]), L1 + L2*np.cos(theta2[i]))
            theta1[i] = alpha - beta
            
            # Calculate joint velocities using Jacobian inverse
            # Forward kinematics: [x, y] = [L1*sin(theta1) + L2*sin(theta1+theta2), 
            #                               -L1*cos(theta1) - L2*cos(theta1+theta2)]
            
            # Jacobian matrix
            c1 = np.cos(theta1[i])
            s1 = np.sin(theta1[i])
            c12 = np.cos(theta1[i] + theta2[i])
            s12 = np.sin(theta1[i] + theta2[i])
            
            J11 = L1*c1 + L2*c12  # dx/dtheta1
            J12 = L2*c12          # dx/dtheta2
            J21 = L1*s1 + L2*s12  # dy/dtheta1
            J22 = L2*s12          # dy/dtheta2
            
            # Jacobian matrix
            J = np.array([[J11, J12],
                         [J21, J22]])
            
            # Calculate joint velocities: theta_dot = J^(-1) * [vx, vy]
            try:
                J_inv = np.linalg.inv(J)
                joint_velocities = J_inv @ np.array([vx, vy])
                theta1_dot[i] = joint_velocities[0]
                theta2_dot[i] = joint_velocities[1]
            except np.linalg.LinAlgError:
                # Handle singular configurations (near singularities)
                # Use pseudo-inverse or set to zero
                try:
                    J_pinv = np.linalg.pinv(J)
                    joint_velocities = J_pinv @ np.array([vx, vy])
                    theta1_dot[i] = joint_velocities[0]
                    theta2_dot[i] = joint_velocities[1]
                except:
                    theta1_dot[i] = 0.0
                    theta2_dot[i] = 0.0
        
        return {
            'hip_angle': theta1,
            'knee_angle': theta2,
            'hip_velocity': theta1_dot,
            'knee_velocity': theta2_dot,
            'ankle_position': ankle_pos,
            'ankle_velocity': ankle_traj['velocity'],
            'ankle_orientation': ankle_traj['orientation'],
            'time': time_vector
        }
    
    def generate_gait_cycle(self, n_points: int = 100, 
                          cycle_time: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Generate a complete gait cycle
        
        Args:
            n_points: Number of points in gait cycle
            cycle_time: Duration of gait cycle in seconds
            
        Returns:
            Complete gait cycle data
        """
        # Create normalized time vector
        time_normalized = np.linspace(0, 1, n_points)
        time_absolute = time_normalized * cycle_time
        
        # Get joint angles
        joint_data = self.predict_joint_angles(time_normalized)
        
        # Get trajectories for both frames
        traj_fr1 = self.predict_ankle_trajectory(time_normalized, 'fr1')
        traj_fr2 = self.predict_ankle_trajectory(time_normalized, 'fr2')
        
        return {
            'time_normalized': time_normalized,
            'time_absolute': time_absolute,
            'joint_angles': joint_data,
            'ankle_trajectory_hip_frame': traj_fr1,
            'ankle_trajectory_global_frame': traj_fr2
        }
    
    def plot_predictions(self, time_vector: np.ndarray, 
                        show_variance: bool = True, 
                        save_path: Optional[str] = None):
        """Plot GMR predictions for both frames including joint velocities"""
        
        # Get predictions for both frames
        traj_fr1 = self.predict_ankle_trajectory(time_vector, 'fr1')
        traj_fr2 = self.predict_ankle_trajectory(time_vector, 'fr2')
        joint_data = self.predict_joint_angles(time_vector)
        
        fig, axes = plt.subplots(4, 3, figsize=(18, 20))
        fig.suptitle('TP-GMM GMR Predictions for Gait Control', fontsize=16, fontweight='bold')
        
        # Frame 1 (Hip frame) plots
        axes[0, 0].plot(time_vector, traj_fr1['position'][:, 0], 'b-', linewidth=2, label='X')
        axes[0, 0].plot(time_vector, traj_fr1['position'][:, 1], 'r-', linewidth=2, label='Y')
        if show_variance and len(traj_fr1['position_variance'].shape) == 3:
            std_x = np.sqrt(traj_fr1['position_variance'][:, 0, 0])
            std_y = np.sqrt(traj_fr1['position_variance'][:, 1, 1])
            axes[0, 0].fill_between(time_vector, traj_fr1['position'][:, 0] - std_x,
                                  traj_fr1['position'][:, 0] + std_x, alpha=0.3, color='blue')
            axes[0, 0].fill_between(time_vector, traj_fr1['position'][:, 1] - std_y,
                                  traj_fr1['position'][:, 1] + std_y, alpha=0.3, color='red')
        axes[0, 0].set_title('FR1 - Ankle Position')
        axes[0, 0].set_ylabel('Position (m)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(time_vector, traj_fr1['velocity'][:, 0], 'b-', linewidth=2, label='Vx')
        axes[0, 1].plot(time_vector, traj_fr1['velocity'][:, 1], 'r-', linewidth=2, label='Vy')
        axes[0, 1].set_title('FR1 - Ankle Velocity')
        axes[0, 1].set_ylabel('Velocity (m/s)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].plot(time_vector, traj_fr1['orientation'], 'g-', linewidth=2)
        axes[0, 2].set_title('FR1 - Ankle Orientation')
        axes[0, 2].set_ylabel('Orientation (rad)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Frame 2 (Global frame) plots
        axes[1, 0].plot(time_vector, traj_fr2['position'][:, 0], 'b--', linewidth=2, label='X')
        axes[1, 0].plot(time_vector, traj_fr2['position'][:, 1], 'r--', linewidth=2, label='Y')
        axes[1, 0].set_title('FR2 - Ankle Position')
        axes[1, 0].set_ylabel('Position (m)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(time_vector, traj_fr2['velocity'][:, 0], 'b--', linewidth=2, label='Vx')
        axes[1, 1].plot(time_vector, traj_fr2['velocity'][:, 1], 'r--', linewidth=2, label='Vy')
        axes[1, 1].set_title('FR2 - Ankle Velocity')
        axes[1, 1].set_ylabel('Velocity (m/s)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].plot(time_vector, traj_fr2['orientation'], 'g--', linewidth=2)
        axes[1, 2].set_title('FR2 - Ankle Orientation')
        axes[1, 2].set_ylabel('Orientation (rad)')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Joint angles
        axes[2, 0].plot(time_vector, np.rad2deg(joint_data['hip_angle']), 'purple', linewidth=2)
        axes[2, 0].set_title('Hip Joint Angle')
        axes[2, 0].set_ylabel('Angle (deg)')
        axes[2, 0].grid(True, alpha=0.3)
        
        axes[2, 1].plot(time_vector, np.rad2deg(joint_data['knee_angle']), 'orange', linewidth=2)
        axes[2, 1].set_title('Knee Joint Angle')
        axes[2, 1].set_ylabel('Angle (deg)')
        axes[2, 1].grid(True, alpha=0.3)
        
        # 2D trajectory
        axes[2, 2].plot(traj_fr1['position'][:, 0], traj_fr1['position'][:, 1], 
                       'b-', linewidth=2, label='Hip Frame')
        axes[2, 2].plot(traj_fr2['position'][:, 0], traj_fr2['position'][:, 1], 
                       'r--', linewidth=2, label='Global Frame')
        axes[2, 2].set_title('2D Ankle Trajectory')
        axes[2, 2].set_xlabel('X Position (m)')
        axes[2, 2].set_ylabel('Y Position (m)')
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)
        axes[2, 2].axis('equal')
        
        # Joint velocities (NEW ROW)
        axes[3, 0].plot(time_vector, np.rad2deg(joint_data['hip_velocity']), 'darkviolet', linewidth=2)
        axes[3, 0].set_title('Hip Joint Velocity')
        axes[3, 0].set_xlabel('Normalized Time')
        axes[3, 0].set_ylabel('Angular Velocity (deg/s)')
        axes[3, 0].grid(True, alpha=0.3)
        
        axes[3, 1].plot(time_vector, np.rad2deg(joint_data['knee_velocity']), 'darkorange', linewidth=2)
        axes[3, 1].set_title('Knee Joint Velocity')
        axes[3, 1].set_xlabel('Normalized Time')
        axes[3, 1].set_ylabel('Angular Velocity (deg/s)')
        axes[3, 1].grid(True, alpha=0.3)
        
        # Joint phase plot (angle vs velocity)
        axes[3, 2].plot(np.rad2deg(joint_data['hip_angle']), np.rad2deg(joint_data['hip_velocity']), 
                       'purple', linewidth=2, label='Hip', alpha=0.7)
        axes[3, 2].plot(np.rad2deg(joint_data['knee_angle']), np.rad2deg(joint_data['knee_velocity']), 
                       'orange', linewidth=2, label='Knee', alpha=0.7)
        axes[3, 2].set_title('Joint Phase Plot')
        axes[3, 2].set_xlabel('Joint Angle (deg)')
        axes[3, 2].set_ylabel('Joint Velocity (deg/s)')
        axes[3, 2].legend()
        axes[3, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to: {save_path}")
        
        plt.show()
    
    def real_time_prediction(self, current_time: float) -> Dict[str, float]:
        """
        Real-time prediction for exoskeleton control
        
        Args:
            current_time: Current normalized time (0.0 to 1.0)
            
        Returns:
            Dictionary with current predictions including joint velocities
        """
        joint_data = self.predict_joint_angles(np.array([current_time]))
        traj_fr1 = self.predict_ankle_trajectory(np.array([current_time]), 'fr1')
        
        return {
            'hip_angle_deg': np.rad2deg(joint_data['hip_angle'][0]),
            'knee_angle_deg': np.rad2deg(joint_data['knee_angle'][0]),
            'hip_velocity_deg_s': np.rad2deg(joint_data['hip_velocity'][0]),
            'knee_velocity_deg_s': np.rad2deg(joint_data['knee_velocity'][0]),
            'ankle_pos_x': traj_fr1['position'][0, 0],
            'ankle_pos_y': traj_fr1['position'][0, 1],
            'ankle_vel_x': traj_fr1['velocity'][0, 0],
            'ankle_vel_y': traj_fr1['velocity'][0, 1],
            'ankle_orientation': traj_fr1['orientation'][0]
        }


def main():
    """Example usage of TP-GMM GMR for exoskeleton control"""
    
    model_pat='models/tpgmm_gait_model_#39-47.pkl'
    info_pat='models/tpgmm_gait_model_#39-47_info.txt'
    # Initialize GMR
    gmr = TPGMMGaussianMixtureRegression(
        model_pat,
        info_pat
    )
    
    # Generate gait cycle
    print("\n=== Generating Gait Cycle ===")
    gait_cycle = gmr.generate_gait_cycle(n_points=100, cycle_time=1.2)
    
    # Display some results
    print(f"Generated {len(gait_cycle['time_normalized'])} points")
    print(f"Hip angle range: {np.rad2deg(gait_cycle['joint_angles']['hip_angle'].min()):.1f}° to "
          f"{np.rad2deg(gait_cycle['joint_angles']['hip_angle'].max()):.1f}°")
    print(f"Knee angle range: {np.rad2deg(gait_cycle['joint_angles']['knee_angle'].min()):.1f}° to "
          f"{np.rad2deg(gait_cycle['joint_angles']['knee_angle'].max()):.1f}°")
    print(f"Hip velocity range: {np.rad2deg(gait_cycle['joint_angles']['hip_velocity'].min()):.1f}°/s to "
          f"{np.rad2deg(gait_cycle['joint_angles']['hip_velocity'].max()):.1f}°/s")
    print(f"Knee velocity range: {np.rad2deg(gait_cycle['joint_angles']['knee_velocity'].min()):.1f}°/s to "
          f"{np.rad2deg(gait_cycle['joint_angles']['knee_velocity'].max()):.1f}°/s")
    
    # Plot predictions
    print("\n=== Generating Plots ===")
    gmr.plot_predictions(gait_cycle['time_normalized'], 
                        show_variance=True,
                        save_path='plots/tpgmm_gmr_predictions' + model_pat[24:-4] + '.png')
    
    # Real-time example
    print("\n=== Real-time Prediction Example ===")
    for phase in [0.0, 0.25, 0.5, 0.75, 1.0]:
        prediction = gmr.real_time_prediction(phase)
        print(f"Phase {phase:.2f}: Hip={prediction['hip_angle_deg']:.1f}° ({prediction['hip_velocity_deg_s']:.1f}°/s), "
              f"Knee={prediction['knee_angle_deg']:.1f}° ({prediction['knee_velocity_deg_s']:.1f}°/s)")
    
    print("\n✓ GMR analysis complete!")


if __name__ == "__main__":
    main()