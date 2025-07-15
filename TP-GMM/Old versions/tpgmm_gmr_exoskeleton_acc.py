#!/usr/bin/env python3
"""
TP-GMM Gaussian Mixture Regression (GMR) for Lower Limb Exoskeleton Control

This script performs Gaussian Mixture Regression on a pre-trained TP-GMM model
for gait dynamics recovery. The model uses 2 frames of reference:
- FR1: Hip frame (exoskeleton coordinate system)
- FR2: Global frame (world coordinate system)

Both frames store true ankle positions after inverse transformation from
the original joint angle representations.

Updated to include acceleration data (7D per frame: position + velocity + acceleration + orientation)

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
    Gaussian Mixture Regression for TP-GMM gait models with acceleration
    
    Performs regression to recover gait dynamics from time-indexed input,
    supporting both single and multi-frame predictions including acceleration.
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
        
        # Frame dimensions (now 7D per frame with acceleration)
        self.frame1_dims = self.data_structure['frame1_dims']  # [1,2,3,4,5,6,7]
        self.frame2_dims = self.data_structure['frame2_dims']  # [8,9,10,11,12,13,14]
        
        # Feature mappings
        self.pos_dims = self.data_structure['position_dims']
        self.vel_dims = self.data_structure['velocity_dims']
        self.acc_dims = self.data_structure['acceleration_dims']  # NEW
        self.orient_dims = self.data_structure['orientation_dims']
        
        if info_path:
            self.load_model_info(info_path)
            
        print(f"✓ TP-GMM GMR initialized (with acceleration):")
        print(f"  Components: {self.n_components}")
        print(f"  Total dimensions: {self.total_dim}")
        print(f"  Features per frame: 7D (pos + vel + acc + orient)")
        print(f"  Coordinate system: {self.frame_info['coordinate_system']}")
        print(f"  Frames: {self.frame_info['num_frames']}")
    
    def load_model(self, model_path: str) -> Dict:
        """Load the pre-trained TP-GMM model"""
        try:
            model_data = joblib.load(model_path)
            print(f"✓ Model loaded from: {model_path}")
            
            # Validate acceleration dimensions exist
            if 'acceleration_dims' not in model_data['data_structure']:
                raise ValueError("Model does not contain acceleration data. Please use a model trained with acceleration.")
            
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
        Predict complete ankle trajectory (position, velocity, acceleration, orientation)
        
        Args:
            time_vector: Array of time values (0.0 to 1.0) 
            frame: 'fr1' (hip frame) or 'fr2' (global frame)
            
        Returns:
            Dictionary with 'position', 'velocity', 'acceleration', 'orientation', and 'variance'
        """
        # Ensure time_vector is array
        if np.isscalar(time_vector):
            time_vector = np.array([time_vector])
        
        if frame.lower() == 'fr1':
            pos_dims = self.pos_dims['frame1_true']      # [1, 2]
            vel_dims = self.vel_dims['frame1_true']      # [3, 4] 
            acc_dims = self.acc_dims['frame1_true']      # [5, 6] NEW
            orient_dims = self.orient_dims['frame1_true'] # [7] NEW index
        elif frame.lower() == 'fr2':
            pos_dims = self.pos_dims['frame2_true']      # [8, 9] NEW indices
            vel_dims = self.vel_dims['frame2_true']      # [10, 11] NEW indices
            acc_dims = self.acc_dims['frame2_true']      # [12, 13] NEW
            orient_dims = self.orient_dims['frame2_true'] # [14] NEW index
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
        
        # Predict acceleration - NEW
        acc_pred, acc_var = self.perform_gmr(
            time_vector, [self.time_dim], acc_dims
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
            if acc_pred.ndim == 1:  # NEW
                acc_pred = acc_pred.reshape(1, -1)
            if np.isscalar(orient_pred):
                orient_pred = np.array([orient_pred])
        
        return {
            'position': pos_pred,
            'velocity': vel_pred, 
            'acceleration': acc_pred,  # NEW
            'orientation': orient_pred,
            'position_variance': pos_var,
            'velocity_variance': vel_var,
            'acceleration_variance': acc_var,  # NEW
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
            Dictionary with joint angles, velocities, accelerations and ankle positions
        """
        # Ensure time_vector is array
        if np.isscalar(time_vector):
            time_vector = np.array([time_vector])
        
        # Get ankle trajectory in hip frame (FR1)
        ankle_traj = self.predict_ankle_trajectory(time_vector, frame='fr1')
        
        # Extract positions, velocities, and accelerations - handle both single point and multiple points
        ankle_pos = ankle_traj['position']
        ankle_vel = ankle_traj['velocity']
        ankle_acc = ankle_traj['acceleration']  # NEW
        
        if ankle_pos.ndim == 1:
            ankle_pos = ankle_pos.reshape(1, -1)  # Make it [1, 2] for single point
        if ankle_vel.ndim == 1:
            ankle_vel = ankle_vel.reshape(1, -1)  # Make it [1, 2] for single point
        if ankle_acc.ndim == 1:  # NEW
            ankle_acc = ankle_acc.reshape(1, -1)  # Make it [1, 2] for single point
        
        # Convert to joint angles, velocities, and accelerations (simplified 2-link model)
        # Assuming hip at origin, links pointing down when theta=0
        L1 = 0.39  # Thigh length (m)
        L2 = 0.413  # Shank length (m)
        
        n_points = ankle_pos.shape[0]
        theta1 = np.zeros(n_points)         # Hip angle
        theta2 = np.zeros(n_points)         # Knee angle
        theta1_dot = np.zeros(n_points)     # Hip angular velocity
        theta2_dot = np.zeros(n_points)     # Knee angular velocity
        theta1_ddot = np.zeros(n_points)    # Hip angular acceleration NEW
        theta2_ddot = np.zeros(n_points)    # Knee angular acceleration NEW
        
        for i in range(n_points):
            x, y = ankle_pos[i, 0], ankle_pos[i, 1]
            vx, vy = ankle_vel[i, 0], ankle_vel[i, 1]
            ax, ay = ankle_acc[i, 0], ankle_acc[i, 1]  # NEW
            
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
                
                # Calculate joint accelerations: theta_ddot = J^(-1) * ([ax, ay] - J_dot * theta_dot) NEW
                # J_dot calculation (time derivative of Jacobian)
                J11_dot = -L1*s1*theta1_dot[i] - L2*s12*(theta1_dot[i] + theta2_dot[i])
                J12_dot = -L2*s12*(theta1_dot[i] + theta2_dot[i])
                J21_dot = L1*c1*theta1_dot[i] + L2*c12*(theta1_dot[i] + theta2_dot[i])
                J22_dot = L2*c12*(theta1_dot[i] + theta2_dot[i])
                
                J_dot = np.array([[J11_dot, J12_dot],
                                 [J21_dot, J22_dot]])
                
                # Calculate accelerations
                joint_accelerations = J_inv @ (np.array([ax, ay]) - J_dot @ np.array([theta1_dot[i], theta2_dot[i]]))
                theta1_ddot[i] = joint_accelerations[0]
                theta2_ddot[i] = joint_accelerations[1]
                
            except np.linalg.LinAlgError:
                # Handle singular configurations (near singularities)
                # Use pseudo-inverse or set to zero
                try:
                    J_pinv = np.linalg.pinv(J)
                    joint_velocities = J_pinv @ np.array([vx, vy])
                    theta1_dot[i] = joint_velocities[0]
                    theta2_dot[i] = joint_velocities[1]
                    
                    # For accelerations, use simplified approach if singular
                    joint_accelerations = J_pinv @ np.array([ax, ay])  # Simplified without J_dot term
                    theta1_ddot[i] = joint_accelerations[0]
                    theta2_ddot[i] = joint_accelerations[1]
                except:
                    theta1_dot[i] = 0.0
                    theta2_dot[i] = 0.0
                    theta1_ddot[i] = 0.0  # NEW
                    theta2_ddot[i] = 0.0  # NEW
        
        return {
            'hip_angle': theta1,
            'knee_angle': theta2,
            'hip_velocity': theta1_dot,
            'knee_velocity': theta2_dot,
            'hip_acceleration': theta1_ddot,    # NEW
            'knee_acceleration': theta2_ddot,   # NEW
            'ankle_position': ankle_pos,
            'ankle_velocity': ankle_traj['velocity'],
            'ankle_acceleration': ankle_traj['acceleration'],  # NEW
            'ankle_orientation': ankle_traj['orientation'],
            'time': time_vector
        }
    
    def generate_gait_cycle(self, n_points: int = 100, 
                          cycle_time: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Generate a complete gait cycle with acceleration
        
        Args:
            n_points: Number of points in gait cycle
            cycle_time: Duration of gait cycle in seconds
            
        Returns:
            Complete gait cycle data including acceleration
        """
        # Create normalized time vector
        time_normalized = np.linspace(0, 1, n_points)
        time_absolute = time_normalized * cycle_time
        
        # Get joint angles (now includes accelerations)
        joint_data = self.predict_joint_angles(time_normalized)
        
        # Get trajectories for both frames (now includes accelerations)
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
        """Plot GMR predictions for both frames including joint velocities and accelerations"""
        
        # Get predictions for both frames
        traj_fr1 = self.predict_ankle_trajectory(time_vector, 'fr1')
        traj_fr2 = self.predict_ankle_trajectory(time_vector, 'fr2')
        joint_data = self.predict_joint_angles(time_vector)
        
        fig, axes = plt.subplots(5, 3, figsize=(18, 25))  # Changed to 5x3 for acceleration
        fig.suptitle('TP-GMM GMR Predictions for Gait Control (with Acceleration)', fontsize=16, fontweight='bold')
        
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
        
        # NEW: FR1 Acceleration plot
        axes[0, 2].plot(time_vector, traj_fr1['acceleration'][:, 0], 'b-', linewidth=2, label='Ax')
        axes[0, 2].plot(time_vector, traj_fr1['acceleration'][:, 1], 'r-', linewidth=2, label='Ay')
        axes[0, 2].set_title('FR1 - Ankle Acceleration')
        axes[0, 2].set_ylabel('Acceleration (m/s²)')
        axes[0, 2].legend()
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
        
        # NEW: FR2 Acceleration plot
        axes[1, 2].plot(time_vector, traj_fr2['acceleration'][:, 0], 'b--', linewidth=2, label='Ax')
        axes[1, 2].plot(time_vector, traj_fr2['acceleration'][:, 1], 'r--', linewidth=2, label='Ay')
        axes[1, 2].set_title('FR2 - Ankle Acceleration')
        axes[1, 2].set_ylabel('Acceleration (m/s²)')
        axes[1, 2].legend()
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
        
        # 2D trajectory with orientation
        axes[2, 2].plot(traj_fr1['position'][:, 0], traj_fr1['position'][:, 1], 
                       'b-', linewidth=2, label='Hip Frame')
        axes[2, 2].plot(traj_fr2['position'][:, 0], traj_fr2['position'][:, 1], 
                       'r--', linewidth=2, label='Global Frame')
        axes[2, 2].scatter(traj_fr1['position'][0, 0], traj_fr1['position'][0, 1], 
                          c='green', s=100, label='Start', zorder=5)
        axes[2, 2].scatter(traj_fr1['position'][-1, 0], traj_fr1['position'][-1, 1], 
                          c='red', s=100, label='End', zorder=5)
        axes[2, 2].set_title('2D Ankle Trajectory')
        axes[2, 2].set_xlabel('X Position (m)')
        axes[2, 2].set_ylabel('Y Position (m)')
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)
        axes[2, 2].axis('equal')
        
        # Joint velocities
        axes[3, 0].plot(time_vector, np.rad2deg(joint_data['hip_velocity']), 'darkviolet', linewidth=2)
        axes[3, 0].set_title('Hip Joint Velocity')
        axes[3, 0].set_ylabel('Angular Velocity (deg/s)')
        axes[3, 0].grid(True, alpha=0.3)
        
        axes[3, 1].plot(time_vector, np.rad2deg(joint_data['knee_velocity']), 'darkorange', linewidth=2)
        axes[3, 1].set_title('Knee Joint Velocity')
        axes[3, 1].set_ylabel('Angular Velocity (deg/s)')
        axes[3, 1].grid(True, alpha=0.3)
        
        # Joint phase plot (angle vs velocity)
        axes[3, 2].plot(np.rad2deg(joint_data['hip_angle']), np.rad2deg(joint_data['hip_velocity']), 
                       'purple', linewidth=2, label='Hip', alpha=0.7)
        axes[3, 2].plot(np.rad2deg(joint_data['knee_angle']), np.rad2deg(joint_data['knee_velocity']), 
                       'orange', linewidth=2, label='Knee', alpha=0.7)
        axes[3, 2].set_title('Joint Phase Plot (Angle vs Velocity)')
        axes[3, 2].set_xlabel('Joint Angle (deg)')
        axes[3, 2].set_ylabel('Joint Velocity (deg/s)')
        axes[3, 2].legend()
        axes[3, 2].grid(True, alpha=0.3)
        
        # NEW: Joint accelerations (5th row)
        axes[4, 0].plot(time_vector, np.rad2deg(joint_data['hip_acceleration']), 'darkmagenta', linewidth=2)
        axes[4, 0].set_title('Hip Joint Acceleration')
        axes[4, 0].set_xlabel('Normalized Time')
        axes[4, 0].set_ylabel('Angular Acceleration (deg/s²)')
        axes[4, 0].grid(True, alpha=0.3)
        
        axes[4, 1].plot(time_vector, np.rad2deg(joint_data['knee_acceleration']), 'darkorange', linewidth=2, alpha=0.8)
        axes[4, 1].set_title('Knee Joint Acceleration')
        axes[4, 1].set_xlabel('Normalized Time')
        axes[4, 1].set_ylabel('Angular Acceleration (deg/s²)')
        axes[4, 1].grid(True, alpha=0.3)
        
        # Acceleration phase plot (velocity vs acceleration)
        axes[4, 2].plot(np.rad2deg(joint_data['hip_velocity']), np.rad2deg(joint_data['hip_acceleration']), 
                       'purple', linewidth=2, label='Hip', alpha=0.7)
        axes[4, 2].plot(np.rad2deg(joint_data['knee_velocity']), np.rad2deg(joint_data['knee_acceleration']), 
                       'orange', linewidth=2, label='Knee', alpha=0.7)
        axes[4, 2].set_title('Joint Phase Plot (Velocity vs Acceleration)')
        axes[4, 2].set_xlabel('Joint Velocity (deg/s)')
        axes[4, 2].set_ylabel('Joint Acceleration (deg/s²)')
        axes[4, 2].legend()
        axes[4, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_acceleration_analysis(self, time_vector: np.ndarray, save_path: Optional[str] = None):
        """NEW: Plot detailed acceleration analysis"""
        
        # Get predictions for both frames
        traj_fr1 = self.predict_ankle_trajectory(time_vector, 'fr1')
        traj_fr2 = self.predict_ankle_trajectory(time_vector, 'fr2')
        joint_data = self.predict_joint_angles(time_vector)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Acceleration Analysis from TP-GMM GMR', fontsize=16, fontweight='bold')
        
        # Ankle acceleration time series
        axes[0, 0].plot(time_vector, traj_fr1['acceleration'][:, 0], 'b-', linewidth=2, label='Ax FR1')
        axes[0, 0].plot(time_vector, traj_fr1['acceleration'][:, 1], 'r-', linewidth=2, label='Ay FR1')
        axes[0, 0].plot(time_vector, traj_fr2['acceleration'][:, 0], 'b--', linewidth=2, label='Ax FR2', alpha=0.7)
        axes[0, 0].plot(time_vector, traj_fr2['acceleration'][:, 1], 'r--', linewidth=2, label='Ay FR2', alpha=0.7)
        axes[0, 0].set_title('Ankle Acceleration vs Time')
        axes[0, 0].set_xlabel('Normalized Time')
        axes[0, 0].set_ylabel('Acceleration (m/s²)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Ankle acceleration magnitude
        acc_mag_fr1 = np.sqrt(traj_fr1['acceleration'][:, 0]**2 + traj_fr1['acceleration'][:, 1]**2)
        acc_mag_fr2 = np.sqrt(traj_fr2['acceleration'][:, 0]**2 + traj_fr2['acceleration'][:, 1]**2)
        axes[0, 1].plot(time_vector, acc_mag_fr1, 'g-', linewidth=2, label='FR1')
        axes[0, 1].plot(time_vector, acc_mag_fr2, 'g--', linewidth=2, label='FR2', alpha=0.7)
        axes[0, 1].set_title('Ankle Acceleration Magnitude')
        axes[0, 1].set_xlabel('Normalized Time')
        axes[0, 1].set_ylabel('|Acceleration| (m/s²)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 2D acceleration phase plot
        axes[0, 2].plot(traj_fr1['acceleration'][:, 0], traj_fr1['acceleration'][:, 1], 
                       'b-', linewidth=2, label='FR1')
        axes[0, 2].plot(traj_fr2['acceleration'][:, 0], traj_fr2['acceleration'][:, 1], 
                       'r--', linewidth=2, label='FR2', alpha=0.7)
        axes[0, 2].scatter(traj_fr1['acceleration'][0, 0], traj_fr1['acceleration'][0, 1], 
                          c='green', s=100, label='Start', zorder=5)
        axes[0, 2].scatter(traj_fr1['acceleration'][-1, 0], traj_fr1['acceleration'][-1, 1], 
                          c='red', s=100, label='End', zorder=5)
        axes[0, 2].set_title('2D Ankle Acceleration Phase Plot')
        axes[0, 2].set_xlabel('Ax (m/s²)')
        axes[0, 2].set_ylabel('Ay (m/s²)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].axis('equal')
        
        # Joint acceleration time series
        axes[1, 0].plot(time_vector, np.rad2deg(joint_data['hip_acceleration']), 'purple', linewidth=2, label='Hip')
        axes[1, 0].plot(time_vector, np.rad2deg(joint_data['knee_acceleration']), 'orange', linewidth=2, label='Knee')
        axes[1, 0].set_title('Joint Accelerations vs Time')
        axes[1, 0].set_xlabel('Normalized Time')
        axes[1, 0].set_ylabel('Angular Acceleration (deg/s²)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Joint acceleration magnitude
        joint_acc_mag_hip = np.abs(joint_data['hip_acceleration'])
        joint_acc_mag_knee = np.abs(joint_data['knee_acceleration'])
        axes[1, 1].plot(time_vector, np.rad2deg(joint_acc_mag_hip), 'purple', linewidth=2, label='Hip')
        axes[1, 1].plot(time_vector, np.rad2deg(joint_acc_mag_knee), 'orange', linewidth=2, label='Knee')
        axes[1, 1].set_title('Joint Acceleration Magnitude')
        axes[1, 1].set_xlabel('Normalized Time')
        axes[1, 1].set_ylabel('|Angular Acceleration| (deg/s²)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Joint acceleration phase plot
        axes[1, 2].plot(np.rad2deg(joint_data['hip_acceleration']), np.rad2deg(joint_data['knee_acceleration']), 
                       'darkgreen', linewidth=2, alpha=0.8)
        axes[1, 2].scatter(np.rad2deg(joint_data['hip_acceleration'][0]), np.rad2deg(joint_data['knee_acceleration'][0]), 
                          c='green', s=100, label='Start', zorder=5)
        axes[1, 2].scatter(np.rad2deg(joint_data['hip_acceleration'][-1]), np.rad2deg(joint_data['knee_acceleration'][-1]), 
                          c='red', s=100, label='End', zorder=5)
        axes[1, 2].set_title('Joint Acceleration Coupling')
        axes[1, 2].set_xlabel('Hip Acceleration (deg/s²)')
        axes[1, 2].set_ylabel('Knee Acceleration (deg/s²)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Acceleration analysis plot saved to: {save_path}")
        
        plt.show()
    
    def real_time_prediction(self, current_time: float) -> Dict[str, float]:
        """
        Real-time prediction for exoskeleton control with acceleration
        
        Args:
            current_time: Current normalized time (0.0 to 1.0)
            
        Returns:
            Dictionary with current predictions including joint velocities and accelerations
        """
        joint_data = self.predict_joint_angles(np.array([current_time]))
        traj_fr1 = self.predict_ankle_trajectory(np.array([current_time]), 'fr1')
        
        return {
            'hip_angle_deg': np.rad2deg(joint_data['hip_angle'][0]),
            'knee_angle_deg': np.rad2deg(joint_data['knee_angle'][0]),
            'hip_velocity_deg_s': np.rad2deg(joint_data['hip_velocity'][0]),
            'knee_velocity_deg_s': np.rad2deg(joint_data['knee_velocity'][0]),
            'hip_acceleration_deg_s2': np.rad2deg(joint_data['hip_acceleration'][0]),     # NEW
            'knee_acceleration_deg_s2': np.rad2deg(joint_data['knee_acceleration'][0]),   # NEW
            'ankle_pos_x': traj_fr1['position'][0, 0],
            'ankle_pos_y': traj_fr1['position'][0, 1],
            'ankle_vel_x': traj_fr1['velocity'][0, 0],
            'ankle_vel_y': traj_fr1['velocity'][0, 1],
            'ankle_acc_x': traj_fr1['acceleration'][0, 0],  # NEW
            'ankle_acc_y': traj_fr1['acceleration'][0, 1],  # NEW
            'ankle_orientation': traj_fr1['orientation'][0]
        }


def main():
    """Example usage of TP-GMM GMR for exoskeleton control with acceleration"""
    
    model_path = 'models/tpgmm_gait_model_with_acceleration_#39-24_Wacc.pkl'
    info_path = 'models/tpgmm_gait_model_with_acceleration_#39-24_Wacc_info.txt'

    # Initialize GMR
    gmr = TPGMMGaussianMixtureRegression(
        model_path,
        info_path
    )
    
    # Generate gait cycle
    print("\n=== Generating Gait Cycle (with Acceleration) ===")
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
    
    # NEW: Display acceleration ranges
    print(f"Hip acceleration range: {np.rad2deg(gait_cycle['joint_angles']['hip_acceleration'].min()):.1f}°/s² to "
          f"{np.rad2deg(gait_cycle['joint_angles']['hip_acceleration'].max()):.1f}°/s²")
    print(f"Knee acceleration range: {np.rad2deg(gait_cycle['joint_angles']['knee_acceleration'].min()):.1f}°/s² to "
          f"{np.rad2deg(gait_cycle['joint_angles']['knee_acceleration'].max()):.1f}°/s²")
    
    # Ankle acceleration statistics
    ankle_acc_fr1 = gait_cycle['ankle_trajectory_hip_frame']['acceleration']
    ankle_acc_mag = np.sqrt(ankle_acc_fr1[:, 0]**2 + ankle_acc_fr1[:, 1]**2)
    print(f"Ankle acceleration magnitude range: {np.min(ankle_acc_mag):.3f} to {np.max(ankle_acc_mag):.3f} m/s²")
    
    # Plot predictions
    print("\n=== Generating Plots ===")
    gmr.plot_predictions(gait_cycle['time_normalized'], 
                        show_variance=True,
                        save_path='plots/tpgmm_gmr_predictions_with_acceleration' + model_path[24:-4] + '.png')
    
    # NEW: Plot acceleration analysis
    print("Creating acceleration analysis...")
    gmr.plot_acceleration_analysis(gait_cycle['time_normalized'],
                                  save_path='plots/tpgmm_gmr_acceleration_analysis' + model_path[24:-4] + '.png')
    
    # Real-time example
    print("\n=== Real-time Prediction Example (with Acceleration) ===")
    for phase in [0.0, 0.25, 0.5, 0.75, 1.0]:
        prediction = gmr.real_time_prediction(phase)
        print(f"Phase {phase:.2f}: Hip={prediction['hip_angle_deg']:.1f}° "
              f"({prediction['hip_velocity_deg_s']:.1f}°/s, {prediction['hip_acceleration_deg_s2']:.1f}°/s²), "
              f"Knee={prediction['knee_angle_deg']:.1f}° "
              f"({prediction['knee_velocity_deg_s']:.1f}°/s, {prediction['knee_acceleration_deg_s2']:.1f}°/s²)")
    
    # Acceleration-specific analysis
    print("\n=== Acceleration Data Analysis ===")
    joint_acc_hip_max = np.max(np.abs(gait_cycle['joint_angles']['hip_acceleration']))
    joint_acc_knee_max = np.max(np.abs(gait_cycle['joint_angles']['knee_acceleration']))
    print(f"Maximum joint accelerations: Hip = {np.rad2deg(joint_acc_hip_max):.1f}°/s², Knee = {np.rad2deg(joint_acc_knee_max):.1f}°/s²")
    
    # Check for potentially problematic accelerations
    hip_acc_threshold = np.deg2rad(500)  # 500 deg/s² threshold
    knee_acc_threshold = np.deg2rad(500)
    
    hip_acc_violations = np.sum(np.abs(gait_cycle['joint_angles']['hip_acceleration']) > hip_acc_threshold)
    knee_acc_violations = np.sum(np.abs(gait_cycle['joint_angles']['knee_acceleration']) > knee_acc_threshold)
    
    if hip_acc_violations > 0 or knee_acc_violations > 0:
        print(f"⚠ High acceleration points detected: Hip={hip_acc_violations}, Knee={knee_acc_violations}")
        print("  Consider smoothing or filtering for exoskeleton control")
    else:
        print("✓ All joint accelerations within reasonable limits for exoskeleton control")
    
    # Frame coincidence check for acceleration
    traj_fr1 = gait_cycle['ankle_trajectory_hip_frame']
    traj_fr2 = gait_cycle['ankle_trajectory_global_frame']
    
    acc_diff_rms = np.sqrt(np.mean((traj_fr1['acceleration'] - traj_fr2['acceleration'])**2))
    print(f"Acceleration coincidence between frames (RMS difference): {acc_diff_rms:.6f} m/s²")
    
    if acc_diff_rms < 0.01:
        print("✓ Acceleration data is coincident between frames")
    else:
        print(f"⚠ Some acceleration divergence between frames (diff: {acc_diff_rms:.6f})")
    
    print("\n✓ GMR analysis with acceleration complete!")


if __name__ == "__main__":
    main()