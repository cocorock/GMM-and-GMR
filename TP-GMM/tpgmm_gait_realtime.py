import sys
import os
import time
import numpy as np
import scipy.io
import joblib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import threading

class ForwardKinematics:
    """
    Forward kinematics calculator for 2D gait analysis
    Based on the MATLAB implementation with FR1 and FR2 reference frames
    """
    def __init__(self, l1=0.3874, l2=0.4136, use_task_space=True):
        """
        Initialize forward kinematics with link lengths
        
        Args:
            l1: Length of first link (hip to knee) in meters - corresponds to lfemur/rfemur
            l2: Length of second link (knee to ankle) in meters - corresponds to ltibia/rtibia
            use_task_space: If True, use FR2 (hip-centered) like the MATLAB FR2 implementation
        """
        self.l1 = l1  # Hip to knee (femur)
        self.l2 = l2  # Knee to ankle (tibia)
        self.use_task_space = use_task_space
        
        # Expected angle ranges (in degrees) - from your data
        self.hip_range_deg = (-23, 34)    # Hip: -23° to 34°
        self.knee_range_deg = (15, 78)    # Knee: 15° to 78°
        
        print(f"✓ Forward Kinematics initialized (MATLAB-compatible):")
        print(f"  L1 (femur): {self.l1:.4f} m")
        print(f"  L2 (tibia): {self.l2:.4f} m")
        if self.use_task_space:
            print(f"  Using FR2 (hip-centered, like MATLAB FR2)")
        else:
            print(f"  Using FR1 (global coordinates, like MATLAB FR1)")
        print(f"  Expected hip range: {self.hip_range_deg[0]}° to {self.hip_range_deg[1]}°")
        print(f"  Expected knee range: {self.knee_range_deg[0]}° to {self.knee_range_deg[1]}°")
    
    def validate_joint_angles(self, hip_angle, knee_angle):
        """
        Validate joint angles are within expected physiological ranges
        
        Args:
            hip_angle: Hip joint angle in radians
            knee_angle: Knee joint angle in radians
            
        Returns:
            bool: True if angles are within expected ranges
        """
        hip_deg = np.rad2deg(hip_angle)
        knee_deg = np.rad2deg(knee_angle)
        
        hip_valid = self.hip_range_deg[0] <= hip_deg <= self.hip_range_deg[1]
        knee_valid = self.knee_range_deg[0] <= knee_deg <= self.knee_range_deg[1]
        
        if not hip_valid:
            print(f"  ⚠ Hip angle {hip_deg:.1f}° outside expected range {self.hip_range_deg}")
        if not knee_valid:
            print(f"  ⚠ Knee angle {knee_deg:.1f}° outside expected range {self.knee_range_deg}")
            
        return hip_valid and knee_valid
    
    def calculate_ankle_position(self, hip_angle, knee_angle, pelvis_orientation=0.0):
        """
        Calculate ankle position from joint angles using MATLAB-compatible forward kinematics
        
        Based on MATLAB code:
        - FR1: Global coordinates with root rotation
        - FR2: Hip-centered coordinates (hip_pos = 0, no root rotation)
        
        Args:
            hip_angle: Hip joint angle in radians (lfemur_rot or rfemur_rot in MATLAB)
            knee_angle: Knee joint angle in radians (ltibia_rot or rtibia_rot in MATLAB)
            pelvis_orientation: Pelvis orientation angle in radians (root rotation)
            
        Returns:
            ankle_position: [x, y] ankle position
        """
        try:
            # Validate inputs
            if np.isnan(hip_angle) or np.isnan(knee_angle) or np.isnan(pelvis_orientation):
                print(f"✗ NaN input to calculate_ankle_position: hip={hip_angle}, knee={knee_angle}, pelvis={pelvis_orientation}")
                return None
            
            # Display input angles in degrees for readability
            hip_deg = np.rad2deg(hip_angle)
            knee_deg = np.rad2deg(knee_angle)
            print(f"  Input angles: Hip={hip_deg:.1f}°, Knee={knee_deg:.1f}°")
            
            # Validate angle ranges
            self.validate_joint_angles(hip_angle, knee_angle)
            
            if self.use_task_space:
                # FR2 Implementation (hip-centered, like MATLAB FR2)
                # hip_pos = 0, no root rotation (rx=ry=rz=0)
                hip_pos = np.array([0.0, 0.0])  # 2D version
                print(f"  Using FR2 (hip-centered): hip at origin")
            else:
                # FR1 Implementation (global coordinates, like MATLAB FR1)
                # Would include root position and rotation
                hip_pos = np.array([0.0, 0.0])  # For now, simplified
                print(f"  Using FR1 (global coordinates)")
            
            # MATLAB-style forward kinematics (adapted to 2D)
            # From MATLAB: knee_vec_l = bone_lengths.lfemur * [0, -cosd(lfemur_rot(i)), sind(lfemur_rot(i))]';
            # In 2D: knee_vec = l1 * [-cos(hip_angle), sin(hip_angle)]  (flipping coordinate system)
            
            # Calculate knee position relative to hip
            # MATLAB uses [0, -cos(angle), sin(angle)] for 3D, we adapt to 2D [x, y]
            knee_offset = self.l1 * np.array([-np.cos(hip_angle), np.sin(hip_angle)])
            knee_pos = hip_pos + knee_offset
            
            # Calculate ankle position relative to knee
            # MATLAB: ankle_vec_l = bone_lengths.ltibia * [0, -cosd(lfemur_rot(i) + ltibia_rot(i)), sind(lfemur_rot(i) + ltibia_rot(i))]';
            total_angle = hip_angle + knee_angle
            ankle_offset = self.l2 * np.array([-np.cos(total_angle), np.sin(total_angle)])
            ankle_pos = knee_pos + ankle_offset
            
            print(f"  Hip position: [{hip_pos[0]:.3f}, {hip_pos[1]:.3f}]")
            print(f"  Knee position: [{knee_pos[0]:.3f}, {knee_pos[1]:.3f}]")
            print(f"  Total leg angle: {total_angle:.3f} rad ({np.rad2deg(total_angle):.1f}°)")
            print(f"  Ankle position (before pelvis rotation): [{ankle_pos[0]:.3f}, {ankle_pos[1]:.3f}]")
            
            # Apply pelvis orientation if needed (like MATLAB root rotation)
            if pelvis_orientation != 0.0:
                # 2D rotation matrix
                cos_pelvis = np.cos(pelvis_orientation)
                sin_pelvis = np.sin(pelvis_orientation)
                rotation_matrix = np.array([[cos_pelvis, -sin_pelvis],
                                          [sin_pelvis, cos_pelvis]])
                
                ankle_pos = rotation_matrix @ ankle_pos
                print(f"  After pelvis rotation ({np.rad2deg(pelvis_orientation):.1f}°): [{ankle_pos[0]:.3f}, {ankle_pos[1]:.3f}]")
            
            # Validate output
            if np.any(np.isnan(ankle_pos)) or np.any(np.isinf(ankle_pos)):
                print(f"✗ Invalid output from calculate_ankle_position: {ankle_pos}")
                return None
            
            print(f"  ✓ Final ankle position: [{ankle_pos[0]:.3f}, {ankle_pos[1]:.3f}]")
            return ankle_pos
            
        except Exception as e:
            print(f"✗ Error in calculate_ankle_position: {e}")
            print(f"   Inputs: hip={hip_angle}, knee={knee_angle}, pelvis={pelvis_orientation}")
            import traceback
            traceback.print_exc()
            return None
    
    def calculate_ankle_velocity(self, hip_angle, knee_angle, hip_velocity, knee_velocity, 
                               pelvis_orientation=0.0, pelvis_velocity=0.0):
        """
        Calculate ankle velocity from joint angles and velocities
        
        Args:
            hip_angle: Hip joint angle in radians
            knee_angle: Knee joint angle in radians
            hip_velocity: Hip joint velocity in rad/s
            knee_velocity: Knee joint velocity in rad/s
            pelvis_orientation: Pelvis orientation angle in radians
            pelvis_velocity: Pelvis velocity in rad/s
            
        Returns:
            ankle_velocity: [vx, vy] ankle velocity
        """
        return [0,0]
        # # Adjust hip angle for vertical downward reference
        # adjusted_hip_angle = hip_angle - np.pi/2
        
        # # Jacobian calculation for 2-link system
        # total_angle = adjusted_hip_angle + knee_angle
        
        # # Partial derivatives of ankle position w.r.t. joint angles
        # dx_dhip = -self.l1 * np.sin(adjusted_hip_angle) - self.l2 * np.sin(total_angle)
        # dy_dhip = self.l1 * np.cos(adjusted_hip_angle) + self.l2 * np.cos(total_angle)
        
        # dx_dknee = -self.l2 * np.sin(total_angle)
        # dy_dknee = self.l2 * np.cos(total_angle)
        
        # # Calculate velocities
        # ankle_vx = dx_dhip * hip_velocity + dx_dknee * knee_velocity
        # ankle_vy = dy_dhip * hip_velocity + dy_dknee * knee_velocity
        
        # # Apply pelvis rotation effect if needed
        # if pelvis_orientation != 0.0 or pelvis_velocity != 0.0:
        #     cos_pelvis = np.cos(pelvis_orientation)
        #     sin_pelvis = np.sin(pelvis_orientation)
            
        #     # Rotate velocity by pelvis orientation
        #     ankle_vx_rot = ankle_vx * cos_pelvis - ankle_vy * sin_pelvis
        #     ankle_vy_rot = ankle_vx * sin_pelvis + ankle_vy * cos_pelvis
            
        #     # Add effect of pelvis rotation on position
        #     ankle_pos = self.calculate_ankle_position(hip_angle, knee_angle, pelvis_orientation)
        #     pelvis_effect_vx = -ankle_pos[1] * pelvis_velocity
        #     pelvis_effect_vy = ankle_pos[0] * pelvis_velocity
            
        #     ankle_vx = ankle_vx_rot + pelvis_effect_vx
        #     ankle_vy = ankle_vy_rot + pelvis_effect_vy
        # else:
        #     ankle_vx = ankle_vx
        #     ankle_vy = ankle_vy
        
    def calculate_ankle_orientation(self, hip_angle, knee_angle, pelvis_orientation=0.0):
        """
        Calculate ankle orientation from joint angles
        Formula: pelvis_orientation + hip_angle - knee_angle + 90°
        
        Args:
            hip_angle: Hip joint angle in radians
            knee_angle: Knee joint angle in radians
            pelvis_orientation: Pelvis orientation angle in radians
            
        Returns:
            ankle_orientation: Ankle orientation in radians
        """
        try:
            # Validate inputs
            if np.isnan(hip_angle) or np.isnan(knee_angle) or np.isnan(pelvis_orientation):
                print(f"✗ NaN input to calculate_ankle_orientation: hip={hip_angle}, knee={knee_angle}, pelvis={pelvis_orientation}")
                return None
            
            # Convert 90° to radians
            ankle_orientation = pelvis_orientation + hip_angle - knee_angle + np.pi/2
            
            # Normalize angle to [-π, π] range
            ankle_orientation = np.arctan2(np.sin(ankle_orientation), np.cos(ankle_orientation))
            
            # Validate output
            if np.isnan(ankle_orientation) or np.isinf(ankle_orientation):
                print(f"✗ Invalid output from calculate_ankle_orientation: {ankle_orientation}")
                return None
            
            return ankle_orientation
            
        except Exception as e:
            print(f"✗ Error in calculate_ankle_orientation: {e}")
            print(f"   Inputs: hip={hip_angle}, knee={knee_angle}, pelvis={pelvis_orientation}")
            return None

class GaitDataSimulator:
    """
    Simulates real-time gait data reception from MAT file
    """
    def __init__(self, mat_file_path):
        """
        Initialize gait data simulator
        
        Args:
            mat_file_path: Path to MAT file with gait data
        """
        self.mat_file_path = mat_file_path
        self.gait_data = None
        self.current_trial = 0
        self.current_step = 0
        self.total_trials = 0
        self.current_trial_data = None
        
        # Load gait data
        self._load_gait_data()
        
        print(f"✓ Gait Data Simulator initialized:")
        print(f"  Total trials: {self.total_trials}")
        print(f"  Press SPACE to get next data point")
        print(f"  Press 'n' to go to next trial")
        print(f"  Press 'q' to quit")
    
    def _load_gait_data(self):
        """Load gait data from MAT file"""
        try:
            mat_data = scipy.io.loadmat(self.mat_file_path)
            self.gait_data = mat_data['output_struct_array']
            self.total_trials = self.gait_data.shape[1]
            
            # Load first trial
            self._load_trial(0)
            
            print(f"✓ Loaded gait data from: {self.mat_file_path}")
            
        except Exception as e:
            print(f"✗ Error loading gait data: {e}")
            raise
    
    def _load_trial(self, trial_index):
        """Load specific trial data"""
        if trial_index >= self.total_trials:
            print(f"Trial {trial_index} not available. Max: {self.total_trials-1}")
            return False
        
        try:
            trial_data = self.gait_data[0, trial_index][0, 0]
            
            # Extract joint data - handle different possible structures
            if hasattr(trial_data, 'dtype') and trial_data.dtype.names:
                # Structured array
                hip_pos = trial_data['hip_pos'][0] if trial_data['hip_pos'].ndim > 1 else trial_data['hip_pos']
                knee_pos = trial_data['knee_pos'][0] if trial_data['knee_pos'].ndim > 1 else trial_data['knee_pos']
                hip_vel = trial_data['hip_vel'][0] if trial_data['hip_vel'].ndim > 1 else trial_data['hip_vel']
                knee_vel = trial_data['knee_vel'][0] if trial_data['knee_vel'].ndim > 1 else trial_data['knee_vel']
                
                # Check if time exists
                if 'time' in trial_data.dtype.names:
                    time_data = trial_data['time'][0] if trial_data['time'].ndim > 1 else trial_data['time']
                else:
                    time_data = np.arange(len(hip_pos.flatten()))
            else:
                # Try direct attribute access
                hip_pos = trial_data.hip_pos
                knee_pos = trial_data.knee_pos
                hip_vel = trial_data.hip_vel
                knee_vel = trial_data.knee_vel
                time_data = getattr(trial_data, 'time', np.arange(len(hip_pos.flatten())))
            
            self.current_trial_data = {
                'hip_pos': hip_pos.flatten(),
                'knee_pos': knee_pos.flatten(),
                'hip_vel': hip_vel.flatten(),
                'knee_vel': knee_vel.flatten(),
                'time': time_data.flatten()
            }
            
            self.current_trial = trial_index
            self.current_step = 0
            
            print(f"✓ Loaded trial {trial_index}: {len(self.current_trial_data['hip_pos'])} data points")
            return True
            
        except Exception as e:
            print(f"✗ Error loading trial {trial_index}: {e}")
            print(f"   Trial data type: {type(trial_data)}")
            print(f"   Trial data structure: {trial_data}")
            return False
    
    def get_current_joint_data(self):
        """Get current joint angles and velocities"""
        if self.current_trial_data is None:
            print("✗ Error: No trial data loaded")
            return None
        
        if self.current_step >= len(self.current_trial_data['hip_pos']):
            print("End of trial reached")
            return None
        
        try:
            # Extract data with validation
            hip_angle = self.current_trial_data['hip_pos'][self.current_step]
            knee_angle = self.current_trial_data['knee_pos'][self.current_step]
            hip_velocity = self.current_trial_data['hip_vel'][self.current_step]
            knee_velocity = self.current_trial_data['knee_vel'][self.current_step]
            time_val = self.current_trial_data['time'][self.current_step]
            
            # Validate data
            if np.isnan(hip_angle) or np.isnan(knee_angle) or np.isnan(hip_velocity) or np.isnan(knee_velocity):
                print(f"✗ Warning: NaN values detected at step {self.current_step}")
                print(f"   hip_angle: {hip_angle}, knee_angle: {knee_angle}")
                print(f"   hip_velocity: {hip_velocity}, knee_velocity: {knee_velocity}")
                return None
            
            data = {
                'hip_angle': np.deg2rad(float(hip_angle)),      # Convert degrees to radians
                'knee_angle': np.deg2rad(float(knee_angle)),    # Convert degrees to radians
                'hip_velocity': np.deg2rad(float(hip_velocity)), # Convert degrees/s to radians/s
                'knee_velocity': np.deg2rad(float(knee_velocity)), # Convert degrees/s to radians/s
                'time': float(time_val),
                'step': self.current_step,
                'trial': self.current_trial
            }
            
            return data
            
        except Exception as e:
            print(f"✗ Error getting joint data at step {self.current_step}: {e}")
            print(f"   Data lengths: hip_pos={len(self.current_trial_data['hip_pos'])}")
            return None
    
    def next_step(self):
        """Advance to next step in current trial"""
        self.current_step += 1
        return self.current_step < len(self.current_trial_data['hip_pos'])
    
    def next_trial(self):
        """Advance to next trial"""
        if self.current_trial + 1 < self.total_trials:
            return self._load_trial(self.current_trial + 1)
        else:
            print("No more trials available")
            return False
    
    def reset_trial(self):
        """Reset current trial to beginning"""
        self.current_step = 0
        print(f"Reset trial {self.current_trial} to beginning")

class TPGMMGaitPredictor:
    """
    TP-GMM predictor for gait data
    """
    def __init__(self, model_path):
        """
        Initialize TP-GMM gait predictor
        
        Args:
            model_path: Path to trained TP-GMM model
        """
        # Load model
        self.model_data = self._load_model(model_path)
        self.gmm_model = self.model_data['gmm_model']
        
        # Control parameters
        self.max_velocity = 0.5  # m/s
        self.smoothing_factor = 0.7
        
        # History for smoothing
        self.velocity_history = deque(maxlen=5)
        self.position_history = deque(maxlen=10)
        
        # Initialize forward kinematics with task space support
        self.fk = ForwardKinematics(use_task_space=True)  # Use task space (FR2) conventions
        
        # Frame simulation (for TP-GMM)
        self.reference_frame_origin = np.array([0.0, 0.0])  # Reference frame
        self.target_frame_origin = np.array([0.0, -0.5])    # Target frame (example)
        
        print(f"✓ TP-GMM Gait Predictor loaded:")
        print(f"  Components: {self.gmm_model.n_components}")
        print(f"  Model dimension: {self.model_data['data_structure']['total_dim']}")
    
    def _load_model(self, model_path):
        """Load trained TP-GMM model"""
        try:
            model_data = joblib.load(model_path)
            print(f"✓ Model loaded from: {model_path}")
            return model_data
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    def process_joint_data(self, joint_data, pelvis_orientation=0.0, pelvis_velocity=0.0):
        """
        Process joint data to get ankle position and velocity
        
        Args:
            joint_data: Dictionary with joint angles and velocities
            pelvis_orientation: Pelvis orientation angle
            pelvis_velocity: Pelvis velocity
            
        Returns:
            Dictionary with ankle position, velocity, and orientation
        """
        try:
            # Validate input data
            if joint_data is None:
                print("✗ Error: joint_data is None")
                return None
            
            required_keys = ['hip_angle', 'knee_angle', 'hip_velocity', 'knee_velocity', 'time']
            for key in required_keys:
                if key not in joint_data:
                    print(f"✗ Error: Missing key '{key}' in joint_data")
                    return None
                if joint_data[key] is None or np.isnan(joint_data[key]):
                    print(f"✗ Error: Invalid value for '{key}': {joint_data[key]}")
                    return None
            
            # Calculate ankle position using forward kinematics
            ankle_pos = self.fk.calculate_ankle_position(
                joint_data['hip_angle'], 
                joint_data['knee_angle'], 
                pelvis_orientation
            )
            
            if ankle_pos is None:
                print("✗ calculate_ankle_position returned None")
                return None
            else:
                print(f"✓ ankle_pos: {ankle_pos}")
            
            # Calculate ankle velocity using forward kinematics
            ankle_vel = self.fk.calculate_ankle_velocity(
                joint_data['hip_angle'], 
                joint_data['knee_angle'],
                joint_data['hip_velocity'], 
                joint_data['knee_velocity'],
                pelvis_orientation, 
                pelvis_velocity
            )
            
            if ankle_vel is None:
                print("✗ calculate_ankle_velocity returned None")
                return None
            else:
                print(f"✓ ankle_vel: {ankle_vel}")
            
            # Calculate ankle orientation using your specified formula
            ankle_orientation = self.fk.calculate_ankle_orientation(
                joint_data['hip_angle'], 
                joint_data['knee_angle'], 
                pelvis_orientation
            )
            
            if ankle_orientation is None:
                print("✗ calculate_ankle_orientation returned None")
                return None
            else:
                print(f"✓ ankle_orientation: {ankle_orientation}")
            
            # Validate outputs
            if ankle_pos is None or ankle_vel is None or ankle_orientation is None:
                print("✗ Error: Forward kinematics returned None values")
                return None
            
            if np.any(np.isnan(ankle_pos)) or np.any(np.isnan(ankle_vel)) or np.isnan(ankle_orientation):
                print("✗ Error: Forward kinematics returned NaN values")
                print(f"   ankle_pos: {ankle_pos}")
                print(f"   ankle_vel: {ankle_vel}")
                print(f"   ankle_orientation: {ankle_orientation}")
                return None
            
            return {
                'position': ankle_pos,
                'velocity': ankle_vel,
                'orientation': ankle_orientation,
                'time': joint_data['time']
            }
            
        except Exception as e:
            print(f"✗ Error in process_joint_data: {e}")
            print(f"   joint_data: {joint_data}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_tpgmm_input(self, ankle_data):
        """
        Create TP-GMM input from ankle data
        
        Args:
            ankle_data: Ankle position, velocity, orientation data
            
        Returns:
            TP-GMM input array [10D] = [Frame1_data | Frame2_data]
        """
        # Frame 1: Reference frame (absolute coordinates)
        frame1_pos = ankle_data['position']
        frame1_vel = ankle_data['velocity'] 
        frame1_orient = ankle_data['orientation']
        
        # Frame 2: Relative to target frame
        frame2_pos = ankle_data['position'] - self.target_frame_origin
        frame2_vel = ankle_data['velocity']  # Velocity should be the same in both frames
        frame2_orient = ankle_data['orientation']
        
        # Create TP-GMM input: [x1, y1, vx1, vy1, o1, x2, y2, vx2, vy2, o2]
        tpgmm_input = np.concatenate([
            frame1_pos,      # [x1, y1]
            frame1_vel,      # [vx1, vy1]
            [frame1_orient], # [o1]
            frame2_pos,      # [x2, y2]
            frame2_vel,      # [vx2, vy2]
            [frame2_orient]  # [o2]
        ])
        
        return tpgmm_input
    
    def predict_next_state(self, current_ankle_data, time_step=0.1):
        """
        Predict next ankle state using TP-GMM
        
        Args:
            current_ankle_data: Current ankle position, velocity, orientation
            time_step: Time step for prediction
            
        Returns:
            Predicted ankle state
        """
        # Create TP-GMM input
        tpgmm_input = self.create_tpgmm_input(current_ankle_data)
        
        # Predict using GMR (simplified version)
        predicted_velocity = self._gmr_predict(tpgmm_input)
        
        # Apply smoothing
        if len(self.velocity_history) > 0:
            alpha = self.smoothing_factor
            prev_vel = self.velocity_history[-1]
            predicted_velocity = alpha * predicted_velocity + (1 - alpha) * prev_vel
        
        # Limit velocity
        velocity_norm = np.linalg.norm(predicted_velocity)
        if velocity_norm > self.max_velocity:
            predicted_velocity = (predicted_velocity / velocity_norm) * self.max_velocity
        
        # Update history
        self.velocity_history.append(predicted_velocity.copy())
        
        # Predict next position
        predicted_position = current_ankle_data['position'] + predicted_velocity * time_step
        
        return {
            'position': predicted_position,
            'velocity': predicted_velocity,
            'orientation': current_ankle_data['orientation']  # For simplicity
        }
    
    def _gmr_predict(self, tpgmm_input):
        """
        Gaussian Mixture Regression for velocity prediction
        
        Args:
            tpgmm_input: Current state in TP-GMM format
            
        Returns:
            Predicted velocity [vx, vy]
        """
        # Use Frame 1 position to predict Frame 1 velocity
        input_dims = [0, 1]      # Frame 1 position [x1, y1]
        output_dims = [2, 3]     # Frame 1 velocity [vx1, vy1]
        
        input_data = tpgmm_input[input_dims]
        
        # GMM parameters
        n_components = self.gmm_model.n_components
        means = self.gmm_model.means_
        covariances = self.gmm_model.covariances_
        weights = self.gmm_model.weights_
        
        # Calculate responsibilities
        responsibilities = np.zeros(n_components)
        
        for i in range(n_components):
            # Extract mean and covariance for input dimensions
            mean_input = means[i, input_dims]
            cov_input = covariances[i][np.ix_(input_dims, input_dims)]
            
            try:
                # Add regularization
                cov_input += np.eye(len(input_dims)) * 1e-6
                
                # Calculate responsibility
                diff = input_data - mean_input
                inv_cov = np.linalg.inv(cov_input)
                mahalanobis = diff.T @ inv_cov @ diff
                
                # Probability
                det_cov = np.linalg.det(cov_input)
                if det_cov > 1e-10:
                    prob = np.exp(-0.5 * mahalanobis) / np.sqrt((2 * np.pi)**len(input_dims) * det_cov)
                    responsibilities[i] = weights[i] * prob
                else:
                    responsibilities[i] = 0.0
                    
            except np.linalg.LinAlgError:
                responsibilities[i] = 0.0
        
        # Normalize responsibilities
        total_resp = np.sum(responsibilities)
        if total_resp > 1e-10:
            responsibilities /= total_resp
        else:
            responsibilities = weights.copy()
        
        # Predict output
        predicted_output = np.zeros(len(output_dims))
        
        for i in range(n_components):
            if responsibilities[i] > 1e-10:
                mean_input = means[i, input_dims]
                mean_output = means[i, output_dims]
                
                cov_input = covariances[i][np.ix_(input_dims, input_dims)]
                cov_cross = covariances[i][np.ix_(input_dims, output_dims)]
                
                try:
                    cov_input += np.eye(len(input_dims)) * 1e-6
                    inv_cov_input = np.linalg.inv(cov_input)
                    
                    # Conditional output
                    conditional_output = mean_output + cov_cross.T @ inv_cov_input @ (input_data - mean_input)
                    predicted_output += responsibilities[i] * conditional_output
                    
                except np.linalg.LinAlgError:
                    predicted_output += responsibilities[i] * mean_output
        
class GaitVisualizer:
    """
    Real-time visualization for gait analysis
    Shows actual trajectory and TP-GMM predictions
    """
    def __init__(self, max_points=200):
        """
        Initialize gait visualizer
        
        Args:
            max_points: Maximum number of points to display
        """
        self.max_points = max_points
        
        # Data storage
        self.actual_positions = deque(maxlen=max_points)
        self.predicted_positions = deque(maxlen=max_points)
        self.actual_velocities = deque(maxlen=max_points)
        self.predicted_velocities = deque(maxlen=max_points)
        self.time_stamps = deque(maxlen=max_points)
        
        # Initialize matplotlib
        plt.ion()  # Turn on interactive mode
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 10))
        self.fig.suptitle('Real-time Gait Analysis with TP-GMM', fontsize=14)
        
        # Configure subplots
        self._setup_plots()
        
        # Plot objects for updating
        self.actual_traj_line, = self.axes[0, 0].plot([], [], 'b-', linewidth=2, label='Actual', alpha=0.8)
        self.predicted_traj_line, = self.axes[0, 0].plot([], [], 'r--', linewidth=2, label='Predicted', alpha=0.8)
        self.actual_pos_scatter = self.axes[0, 0].scatter([], [], c='blue', s=50, alpha=0.6, zorder=5)
        self.predicted_pos_scatter = self.axes[0, 0].scatter([], [], c='red', s=30, alpha=0.8, marker='x', zorder=6)
        
        self.actual_vel_line, = self.axes[0, 1].plot([], [], 'b-', linewidth=2, label='Actual')
        self.predicted_vel_line, = self.axes[0, 1].plot([], [], 'r--', linewidth=2, label='Predicted')
        
        self.pos_x_actual, = self.axes[1, 0].plot([], [], 'b-', linewidth=2, label='Actual X')
        self.pos_x_predicted, = self.axes[1, 0].plot([], [], 'r--', linewidth=2, label='Predicted X')
        self.pos_y_actual, = self.axes[1, 1].plot([], [], 'b-', linewidth=2, label='Actual Y')
        self.pos_y_predicted, = self.axes[1, 1].plot([], [], 'r--', linewidth=2, label='Predicted Y')
        
        print("✓ Gait Visualizer initialized")
    
    def _setup_plots(self):
        """Setup subplot configurations"""
        # 2D Trajectory
        self.axes[0, 0].set_title('2D Ankle Trajectory')
        self.axes[0, 0].set_xlabel('X Position (m)')
        self.axes[0, 0].set_ylabel('Y Position (m)')
        self.axes[0, 0].grid(True, alpha=0.3)
        self.axes[0, 0].legend()
        self.axes[0, 0].set_aspect('equal')
        
        # Velocity Magnitude
        self.axes[0, 1].set_title('Velocity Magnitude')
        self.axes[0, 1].set_xlabel('Time Step')
        self.axes[0, 1].set_ylabel('Velocity (m/s)')
        self.axes[0, 1].grid(True, alpha=0.3)
        self.axes[0, 1].legend()
        
        # Position X vs Time
        self.axes[1, 0].set_title('X Position vs Time')
        self.axes[1, 0].set_xlabel('Time Step')
        self.axes[1, 0].set_ylabel('X Position (m)')
        self.axes[1, 0].grid(True, alpha=0.3)
        self.axes[1, 0].legend()
        
        # Position Y vs Time
        self.axes[1, 1].set_title('Y Position vs Time')
        self.axes[1, 1].set_xlabel('Time Step')
        self.axes[1, 1].set_ylabel('Y Position (m)')
        self.axes[1, 1].grid(True, alpha=0.3)
        self.axes[1, 1].legend()
        
        plt.tight_layout()
    
    def add_data_point(self, actual_pos, predicted_pos, actual_vel, predicted_vel, time_step):
        """
        Add new data point to visualization
        
        Args:
            actual_pos: Actual ankle position [x, y]
            predicted_pos: Predicted ankle position [x, y]
            actual_vel: Actual ankle velocity [vx, vy]
            predicted_vel: Predicted ankle velocity [vx, vy]
            time_step: Current time step
        """
        # Store data
        self.actual_positions.append(actual_pos.copy())
        self.predicted_positions.append(predicted_pos.copy())
        self.actual_velocities.append(np.linalg.norm(actual_vel))
        self.predicted_velocities.append(np.linalg.norm(predicted_vel))
        self.time_stamps.append(time_step)
        
        # Update plots
        self._update_plots()
    
    def _update_plots(self):
        """Update all plots with current data"""
        if len(self.actual_positions) < 2:
            return
        
        # Convert to arrays for plotting
        actual_pos_array = np.array(self.actual_positions)
        predicted_pos_array = np.array(self.predicted_positions)
        time_array = np.array(self.time_stamps)
        actual_vel_array = np.array(self.actual_velocities)
        predicted_vel_array = np.array(self.predicted_velocities)
        
        # Update 2D trajectory
        self.actual_traj_line.set_data(actual_pos_array[:, 0], actual_pos_array[:, 1])
        self.predicted_traj_line.set_data(predicted_pos_array[:, 0], predicted_pos_array[:, 1])
        
        # Update scatter points (last few points)
        n_recent = min(10, len(actual_pos_array))
        if n_recent > 0:
            self.actual_pos_scatter.set_offsets(actual_pos_array[-n_recent:])
            self.predicted_pos_scatter.set_offsets(predicted_pos_array[-n_recent:])
        
        # Update velocity plot
        self.actual_vel_line.set_data(time_array, actual_vel_array)
        self.predicted_vel_line.set_data(time_array, predicted_vel_array)
        
        # Update position vs time plots
        self.pos_x_actual.set_data(time_array, actual_pos_array[:, 0])
        self.pos_x_predicted.set_data(time_array, predicted_pos_array[:, 0])
        self.pos_y_actual.set_data(time_array, actual_pos_array[:, 1])
        self.pos_y_predicted.set_data(time_array, predicted_pos_array[:, 1])
        
        # Auto-scale axes
        self._auto_scale_axes(actual_pos_array, predicted_pos_array, time_array, 
                            actual_vel_array, predicted_vel_array)
        
        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def _auto_scale_axes(self, actual_pos, predicted_pos, time_array, actual_vel, predicted_vel):
        """Auto-scale all axes based on current data"""
        if len(actual_pos) == 0:
            return
        
        # 2D trajectory limits
        all_x = np.concatenate([actual_pos[:, 0], predicted_pos[:, 0]])
        all_y = np.concatenate([actual_pos[:, 1], predicted_pos[:, 1]])
        
        x_margin = (np.max(all_x) - np.min(all_x)) * 0.1
        y_margin = (np.max(all_y) - np.min(all_y)) * 0.1
        
        self.axes[0, 0].set_xlim(np.min(all_x) - x_margin, np.max(all_x) + x_margin)
        self.axes[0, 0].set_ylim(np.min(all_y) - y_margin, np.max(all_y) + y_margin)
        
        # Velocity limits
        all_vel = np.concatenate([actual_vel, predicted_vel])
        vel_margin = (np.max(all_vel) - np.min(all_vel)) * 0.1
        
        self.axes[0, 1].set_xlim(np.min(time_array), np.max(time_array))
        self.axes[0, 1].set_ylim(np.min(all_vel) - vel_margin, np.max(all_vel) + vel_margin)
        
        # Position vs time limits
        self.axes[1, 0].set_xlim(np.min(time_array), np.max(time_array))
        self.axes[1, 0].set_ylim(np.min(all_x) - x_margin, np.max(all_x) + x_margin)
        
        self.axes[1, 1].set_xlim(np.min(time_array), np.max(time_array))
        self.axes[1, 1].set_ylim(np.min(all_y) - y_margin, np.max(all_y) + y_margin)
    
    def clear_data(self):
        """Clear all stored data and reset plots"""
        self.actual_positions.clear()
        self.predicted_positions.clear()
        self.actual_velocities.clear()
        self.predicted_velocities.clear()
        self.time_stamps.clear()
        
        # Clear all plot data
        self.actual_traj_line.set_data([], [])
        self.predicted_traj_line.set_data([], [])
        self.actual_pos_scatter.set_offsets(np.empty((0, 2)))
        self.predicted_pos_scatter.set_offsets(np.empty((0, 2)))
        self.actual_vel_line.set_data([], [])
        self.predicted_vel_line.set_data([], [])
        self.pos_x_actual.set_data([], [])
        self.pos_x_predicted.set_data([], [])
        self.pos_y_actual.set_data([], [])
        self.pos_y_predicted.set_data([], [])
        
        self.fig.canvas.draw()
        print("✓ Visualization data cleared")
    
    def save_plot(self, filename=None):
        """Save current plot to file"""
        if filename is None:
            filename = f"gait_analysis_{int(time.time())}.png"
        
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to: {filename}")
    
    def close(self):
        """Close the visualization"""
        plt.close(self.fig)
        print("✓ Visualization closed")

def wait_for_user_input():
    """Wait for user input (space, n, or q) - Windows compatible"""
    print("\nPress SPACE for next step, 'n' for next trial, 'q' to quit: ", end='', flush=True)
    
    try:
        # Windows compatible input method
        import msvcrt
        
        while True:
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8').lower()
                print(key)  # Echo the key pressed
                
                if key == ' ':  # Space
                    return 'space'
                elif key == 'n':
                    return 'next_trial'
                elif key == 'q':
                    return 'quit'
                elif key == 's':
                    return 'save'
                elif key == 'c':
                    return 'clear'
                elif key == '\r':  # Enter key
                    return 'space'
            time.sleep(0.1)
            
    except ImportError:
        # Fallback for non-Windows systems
        print("\n(Press Enter after typing your choice)")
        user_input = input().strip().lower()
        
        if user_input == '' or user_input == ' ':  # Space or Enter
            return 'space'
        elif user_input == 'n':
            return 'next_trial'
        elif user_input == 'q':
            return 'quit'
        elif user_input == 's':
            return 'save'
        elif user_input == 'c':
            return 'clear'
        else:
            return 'space'  # Default action

def run_gait_simulation():
    """Main simulation loop"""
    
    # Parameters
    model_path = 'tpgmm_gait_model.pkl'  # Your trained model
    mat_file_path = 'demo_gait_data_angular_10_samples.mat'  # Your gait data
    
    try:
        # Initialize components
        print("=== Initializing Gait TP-GMM Simulation ===")
        simulator = GaitDataSimulator(mat_file_path)
        predictor = TPGMMGaitPredictor(model_path)
        visualizer = GaitVisualizer(max_points=300)
        
        print("\n=== Starting Simulation ===")
        print("Commands:")
        print("  SPACE/ENTER: Next data point")
        print("  'n': Next trial")
        print("  'q': Quit")
        print("  's': Save current plot")
        print("  'c': Clear visualization")
        
        # Simulation loop
        step_count = 0
        
        while True:
            # Get current joint data
            joint_data = simulator.get_current_joint_data()
            
            if joint_data is None:
                print("\nEnd of trial. Press 'n' for next trial or 'q' to quit.")
                user_input = wait_for_user_input()
                
                if user_input == 'next_trial':
                    if simulator.next_trial():
                        step_count = 0
                        visualizer.clear_data()  # Clear visualization for new trial
                        continue
                    else:
                        print("No more trials. Exiting.")
                        break
                elif user_input == 'quit':
                    break
                elif user_input == 'save':
                    visualizer.save_plot()
                    continue
                elif user_input == 'clear':
                    visualizer.clear_data()
                    continue
                else:
                    continue
            
            # Process joint data to ankle data
            ankle_data = predictor.process_joint_data(joint_data)
            
            # Check if ankle_data is valid
            if ankle_data is None:
                print(f"✗ Failed to process joint data at step {step_count}")
                print(f"   Joint data: {joint_data}")
                
                # Wait for user input to continue or quit
                user_input = wait_for_user_input()
                if user_input == 'quit':
                    break
                elif user_input == 'next_trial':
                    if simulator.next_trial():
                        step_count = 0
                        visualizer.clear_data()
                        continue
                    else:
                        print("No more trials. Exiting.")
                        break
                elif user_input == 'save':
                    visualizer.save_plot()
                    continue
                elif user_input == 'clear':
                    visualizer.clear_data()
                    continue
                else:
                    simulator.next_step()
                    step_count += 1
                    continue
            
            # Display current state
            print(f"\n--- Step {step_count} (Trial {joint_data['trial']}, Point {joint_data['step']}) ---")
            print(f"Joint angles (input): Hip={np.rad2deg(joint_data['hip_angle']):.1f}°, Knee={np.rad2deg(joint_data['knee_angle']):.1f}°")
            print(f"Joint velocities: Hip={np.rad2deg(joint_data['hip_velocity']):.1f}°/s, Knee={np.rad2deg(joint_data['knee_velocity']):.1f}°/s")
            print(f"Ankle position: [{ankle_data['position'][0]:.3f}, {ankle_data['position'][1]:.3f}] m")
            print(f"Ankle velocity: [{ankle_data['velocity'][0]:.3f}, {ankle_data['velocity'][1]:.3f}] m/s")
            print(f"Ankle orientation: {ankle_data['orientation']:.3f} rad ({np.rad2deg(ankle_data['orientation']):.1f}°)")
            
            # Predict next state
            predicted_state = predictor.predict_next_state(ankle_data)
            print(f"Predicted velocity: [{predicted_state['velocity'][0]:.3f}, {predicted_state['velocity'][1]:.3f}]")
            print(f"Predicted position: [{predicted_state['position'][0]:.3f}, {predicted_state['position'][1]:.3f}]")
            
            # Update visualization
            visualizer.add_data_point(
                actual_pos=ankle_data['position'],
                predicted_pos=predicted_state['position'],
                actual_vel=ankle_data['velocity'],
                predicted_vel=predicted_state['velocity'],
                time_step=step_count
            )
            
            # Wait for user input
            user_input = wait_for_user_input()
            
            if user_input == 'space':
                simulator.next_step()
                step_count += 1
            elif user_input == 'next_trial':
                if simulator.next_trial():
                    step_count = 0
                    visualizer.clear_data()  # Clear visualization for new trial
                else:
                    print("No more trials. Exiting.")
                    break
            elif user_input == 'quit':
                break
            elif user_input == 'save':
                visualizer.save_plot()
            elif user_input == 'clear':
                visualizer.clear_data()
        
        print("\n✓ Simulation completed!")
        
        # Close visualization
        print("Closing visualization...")
        visualizer.close()
        
    except FileNotFoundError as e:
        print(f"✗ File not found: {e}")
        print("Please check that your model and data files exist.")
    except Exception as e:
        print(f"✗ Error during simulation: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    print("=== TP-GMM Gait Real-time Simulation ===")
    print("This program simulates real-time gait analysis using TP-GMM")
    print("Make sure you have:")
    print("  1. Trained TP-GMM model (tpgmm_gait_model.pkl)")
    print("  2. Gait data file (demo_gait_data_angular_10_samples.mat)")
    print()
    
    run_gait_simulation()

if __name__ == "__main__":
    main()