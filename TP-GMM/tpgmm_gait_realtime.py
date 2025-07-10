import sys
import os
import time
import numpy as np
import scipy.io
import joblib
import matplotlib.pyplot as plt
from collections import deque
import threading

class ForwardKinematics:
    """
    Forward kinematics calculator for 2D gait analysis
    """
    def __init__(self, l1=0.3874, l2=0.4136):
        """
        Initialize forward kinematics with link lengths
        
        Args:
            l1: Length of first link (hip to knee) in meters
            l2: Length of second link (knee to ankle) in meters
        """
        self.l1 = l1  # Hip to knee
        self.l2 = l2  # Knee to ankle
        print(f"✓ Forward Kinematics initialized:")
        print(f"  L1 (hip-knee): {self.l1:.4f} m")
        print(f"  L2 (knee-ankle): {self.l2:.4f} m")
    
    def calculate_ankle_position(self, hip_angle, knee_angle, pelvis_orientation=0.0):
        """
        Calculate ankle position from joint angles
        
        Args:
            hip_angle: Hip joint angle in radians
            knee_angle: Knee joint angle in radians  
            pelvis_orientation: Pelvis orientation angle in radians
            
        Returns:
            ankle_position: [x, y] ankle position relative to hip
        """
        # For vertical downward leg when hip_angle=0, knee_angle=0:
        # Adjust angles so that 0,0 means straight down
        # Hip angle: 0 = straight down, positive = forward
        adjusted_hip_angle = hip_angle - np.pi/2  # -90 degrees to make 0 = downward
        
        # Calculate knee position relative to hip
        knee_x = self.l1 * np.cos(adjusted_hip_angle)
        knee_y = self.l1 * np.sin(adjusted_hip_angle)
        
        # Calculate ankle position relative to knee
        # Total angle for second link
        total_angle = adjusted_hip_angle + knee_angle
        ankle_x = knee_x + self.l2 * np.cos(total_angle)
        ankle_y = knee_y + self.l2 * np.sin(total_angle)
        
        # Apply pelvis orientation if needed
        if pelvis_orientation != 0.0:
            cos_pelvis = np.cos(pelvis_orientation)
            sin_pelvis = np.sin(pelvis_orientation)
            
            # Rotate ankle position by pelvis orientation
            ankle_x_rot = ankle_x * cos_pelvis - ankle_y * sin_pelvis
            ankle_y_rot = ankle_x * sin_pelvis + ankle_y * cos_pelvis
            
            ankle_x, ankle_y = ankle_x_rot, ankle_y_rot
        
        return np.array([ankle_x, ankle_y])
    
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
        # Adjust hip angle for vertical downward reference
        adjusted_hip_angle = hip_angle - np.pi/2
        
        # Jacobian calculation for 2-link system
        total_angle = adjusted_hip_angle + knee_angle
        
        # Partial derivatives of ankle position w.r.t. joint angles
        dx_dhip = -self.l1 * np.sin(adjusted_hip_angle) - self.l2 * np.sin(total_angle)
        dy_dhip = self.l1 * np.cos(adjusted_hip_angle) + self.l2 * np.cos(total_angle)
        
        dx_dknee = -self.l2 * np.sin(total_angle)
        dy_dknee = self.l2 * np.cos(total_angle)
        
        # Calculate velocities
        ankle_vx = dx_dhip * hip_velocity + dx_dknee * knee_velocity
        ankle_vy = dy_dhip * hip_velocity + dy_dknee * knee_velocity
        
        # Apply pelvis rotation effect if needed
        if pelvis_orientation != 0.0 or pelvis_velocity != 0.0:
            cos_pelvis = np.cos(pelvis_orientation)
            sin_pelvis = np.sin(pelvis_orientation)
            
            # Rotate velocity by pelvis orientation
            ankle_vx_rot = ankle_vx * cos_pelvis - ankle_vy * sin_pelvis
            ankle_vy_rot = ankle_vx * sin_pelvis + ankle_vy * cos_pelvis
            
            # Add effect of pelvis rotation on position
            ankle_pos = self.calculate_ankle_position(hip_angle, knee_angle, pelvis_orientation)
            pelvis_effect_vx = -ankle_pos[1] * pelvis_velocity
            pelvis_effect_vy = ankle_pos[0] * pelvis_velocity
            
            ankle_vx = ankle_vx_rot + pelvis_effect_vx
            ankle_vy = ankle_vy_rot + pelvis_effect_vy
        else:
            ankle_vx = ankle_vx
            ankle_vy = ankle_vy
        
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
        # Convert 90° to radians
        ankle_orientation = pelvis_orientation + hip_angle - knee_angle + np.pi/2
        
        # Normalize angle to [-π, π] range
        ankle_orientation = np.arctan2(np.sin(ankle_orientation), np.cos(ankle_orientation))
        
        return ankle_orientation

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
            return None
        
        if self.current_step >= len(self.current_trial_data['hip_pos']):
            print("End of trial reached")
            return None
        
        data = {
            'hip_angle': self.current_trial_data['hip_pos'][self.current_step],
            'knee_angle': self.current_trial_data['knee_pos'][self.current_step],
            'hip_velocity': self.current_trial_data['hip_vel'][self.current_step],
            'knee_velocity': self.current_trial_data['knee_vel'][self.current_step],
            'time': self.current_trial_data['time'][self.current_step],
            'step': self.current_step,
            'trial': self.current_trial
        }
        
        return data
    
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
        
        # Initialize forward kinematics
        self.fk = ForwardKinematics()
        
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
        # Calculate ankle position using forward kinematics
        ankle_pos = self.fk.calculate_ankle_position(
            joint_data['hip_angle'], 
            joint_data['knee_angle'], 
            pelvis_orientation
        )
        
        # Calculate ankle velocity using forward kinematics
        ankle_vel = self.fk.calculate_ankle_velocity(
            joint_data['hip_angle'], 
            joint_data['knee_angle'],
            joint_data['hip_velocity'], 
            joint_data['knee_velocity'],
            pelvis_orientation, 
            pelvis_velocity
        )
        
        # Calculate ankle orientation using your specified formula
        ankle_orientation = self.fk.calculate_ankle_orientation(
            joint_data['hip_angle'], 
            joint_data['knee_angle'], 
            pelvis_orientation
        )
        
        return {
            'position': ankle_pos,
            'velocity': ankle_vel,
            'orientation': ankle_orientation,
            'time': joint_data['time']
        }
    
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
        
        return predicted_output

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
        
        print("\\n=== Starting Simulation ===")
        print("Commands:")
        print("  SPACE/ENTER: Next data point")
        print("  'n': Next trial")
        print("  'q': Quit")
        
        # Simulation loop
        step_count = 0
        
        while True:
            # Get current joint data
            joint_data = simulator.get_current_joint_data()
            
            if joint_data is None:
                print("\\nEnd of trial. Press 'n' for next trial or 'q' to quit.")
                user_input = wait_for_user_input()
                
                if user_input == 'next_trial':
                    if simulator.next_trial():
                        step_count = 0
                        continue
                    else:
                        print("No more trials. Exiting.")
                        break
                elif user_input == 'quit':
                    break
                else:
                    continue
            
            # Process joint data to ankle data
            ankle_data = predictor.process_joint_data(joint_data)
            
            # Display current state
            print(f"\\n--- Step {step_count} (Trial {joint_data['trial']}, Point {joint_data['step']}) ---")
            print(f"Joint angles: Hip={joint_data['hip_angle']:.3f}, Knee={joint_data['knee_angle']:.3f}")
            print(f"Ankle position: [{ankle_data['position'][0]:.3f}, {ankle_data['position'][1]:.3f}]")
            print(f"Ankle velocity: [{ankle_data['velocity'][0]:.3f}, {ankle_data['velocity'][1]:.3f}]")
            print(f"Ankle orientation: {ankle_data['orientation']:.3f}")
            
            # Predict next state
            predicted_state = predictor.predict_next_state(ankle_data)
            print(f"Predicted velocity: [{predicted_state['velocity'][0]:.3f}, {predicted_state['velocity'][1]:.3f}]")
            print(f"Predicted position: [{predicted_state['position'][0]:.3f}, {predicted_state['position'][1]:.3f}]")
            
            # Wait for user input
            user_input = wait_for_user_input()
            
            if user_input == 'space':
                simulator.next_step()
                step_count += 1
            elif user_input == 'next_trial':
                if simulator.next_trial():
                    step_count = 0
                else:
                    print("No more trials. Exiting.")
                    break
            elif user_input == 'quit':
                break
        
        print("\\n✓ Simulation completed!")
        
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