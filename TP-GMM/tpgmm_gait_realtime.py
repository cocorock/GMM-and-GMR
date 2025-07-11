import numpy as np
import scipy.io
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time
import threading
import sys
from typing import Dict, List, Tuple
from collections import deque

class InputHandler:
    """
    Cross-platform input handler that works with Python 3.11
    """
    def __init__(self):
        self.command = None
        self.running = True
        
    def get_input(self):
        """Get user input in a cross-platform way"""
        while self.running:
            try:
                user_input = input().strip().lower()
                if user_input == '' or user_input == 'space':
                    self.command = 'space'
                elif user_input == 'q' or user_input == 'quit':
                    self.command = 'q'
                elif user_input == 'r' or user_input == 'reset':
                    self.command = 'r'
                elif user_input == 'h' or user_input == 'help':
                    self.command = 'h'
                else:
                    print("Commands: ENTER/space=next, q=quit, r=reset, h=help")
            except (EOFError, KeyboardInterrupt):
                self.command = 'q'
                break
    
    def start_input_thread(self):
        """Start input handling in separate thread"""
        input_thread = threading.Thread(target=self.get_input, daemon=True)
        input_thread.start()
    
    def get_command(self):
        """Get and clear current command"""
        cmd = self.command
        self.command = None
        return cmd
    
    def stop(self):
        """Stop input handler"""
        self.running = False

class GaitKinematics:
    """
    Forward kinematics for 2-link leg with pelvis orientation
    """
    def __init__(self, l1=0.4135, l2=0.39):
        """
        Initialize gait kinematics
        
        Args:
            l1: Femur length (hip to knee) in meters
            l2: Tibia length (knee to ankle) in meters
        """
        self.l1 = l1  # Femur length
        self.l2 = l2  # Tibia length
        
    def process_joint_angles(self, hip_angle_deg, knee_angle_deg):
        """
        Process joint angles according to gait_kinematics.py processing
        
        Args:
            hip_angle_deg: Raw hip angle in degrees
            knee_angle_deg: Raw knee angle in degrees
            
        Returns:
            Processed hip and knee angles
        """
        # Process hip position: invert signal and subtract 90 degrees
        hip_processed = -hip_angle_deg - 90
        
        # Process knee position: invert signal
        knee_processed = -knee_angle_deg
        
        return hip_processed, knee_processed
    
    def forward_kinematics(self, hip_angle_deg, knee_angle_deg, pelvis_orientation_deg=0):
        """
        Calculate forward kinematics for 2-link leg with pelvis orientation
        
        Args:
            hip_angle_deg: Hip joint angle (degrees, processed)
            knee_angle_deg: Knee joint angle (degrees, processed)
            pelvis_orientation_deg: Pelvis orientation angle (degrees)
            
        Returns:
            hip_pos: Hip position [x, y]
            knee_pos: Knee position [x, y]
            ankle_pos: Ankle position [x, y]
            ankle_orientation: Ankle orientation (degrees)
        """
        # Convert to radians
        hip_angle = np.radians(hip_angle_deg)
        knee_angle = np.radians(knee_angle_deg)
        pelvis_angle = np.radians(pelvis_orientation_deg)
        
        # Hip position (considering pelvis orientation)
        hip_pos = np.array([0, 0])  # Hip at origin
        
        # Knee position (hip angle relative to pelvis)
        knee_x = self.l1 * np.cos(hip_angle + pelvis_angle)
        knee_y = self.l1 * np.sin(hip_angle + pelvis_angle)
        knee_pos = np.array([knee_x, knee_y])
        
        # Ankle position (knee angle relative to femur)
        ankle_x = knee_x + self.l2 * np.cos(hip_angle + knee_angle + pelvis_angle)
        ankle_y = knee_y + self.l2 * np.sin(hip_angle + knee_angle + pelvis_angle)
        ankle_pos = np.array([ankle_x, ankle_y])
        
        # Ankle orientation (+90 degrees from tibia)
        tibia_angle = hip_angle + knee_angle + pelvis_angle
        ankle_orientation_rad = tibia_angle + np.pi/2
        ankle_orientation_deg = np.degrees(ankle_orientation_rad)
        
        return hip_pos, knee_pos, ankle_pos, ankle_orientation_deg
    """
    Forward kinematics for 2-link leg with pelvis orientation
    """
    def __init__(self, l1=0.4135, l2=0.39):
        """
        Initialize gait kinematics
        
        Args:
            l1: Femur length (hip to knee) in meters
            l2: Tibia length (knee to ankle) in meters
        """
        self.l1 = l1  # Femur length
        self.l2 = l2  # Tibia length
        
    def process_joint_angles(self, hip_angle_deg, knee_angle_deg):
        """
        Process joint angles according to gait_kinematics.py processing
        
        Args:
            hip_angle_deg: Raw hip angle in degrees
            knee_angle_deg: Raw knee angle in degrees
            
        Returns:
            Processed hip and knee angles
        """
        # Process hip position: invert signal and subtract 90 degrees
        hip_processed = -hip_angle_deg - 90
        
        # Process knee position: invert signal
        knee_processed = -knee_angle_deg
        
        return hip_processed, knee_processed
    
    def forward_kinematics(self, hip_angle_deg, knee_angle_deg, pelvis_orientation_deg=0):
        """
        Calculate forward kinematics for 2-link leg with pelvis orientation
        
        Args:
            hip_angle_deg: Hip joint angle (degrees, processed)
            knee_angle_deg: Knee joint angle (degrees, processed)
            pelvis_orientation_deg: Pelvis orientation angle (degrees)
            
        Returns:
            hip_pos: Hip position [x, y]
            knee_pos: Knee position [x, y]
            ankle_pos: Ankle position [x, y]
            ankle_orientation: Ankle orientation (degrees)
        """
        # Convert to radians
        hip_angle = np.radians(hip_angle_deg)
        knee_angle = np.radians(knee_angle_deg)
        pelvis_angle = np.radians(pelvis_orientation_deg)
        
        # Hip position (considering pelvis orientation)
        hip_pos = np.array([0, 0])  # Hip at origin
        
        # Knee position (hip angle relative to pelvis)
        knee_x = self.l1 * np.cos(hip_angle + pelvis_angle)
        knee_y = self.l1 * np.sin(hip_angle + pelvis_angle)
        knee_pos = np.array([knee_x, knee_y])
        
        # Ankle position (knee angle relative to femur)
        ankle_x = knee_x + self.l2 * np.cos(hip_angle + knee_angle + pelvis_angle)
        ankle_y = knee_y + self.l2 * np.sin(hip_angle + knee_angle + pelvis_angle)
        ankle_pos = np.array([ankle_x, ankle_y])
        
        # Ankle orientation (+90 degrees from tibia)
        tibia_angle = hip_angle + knee_angle + pelvis_angle
        ankle_orientation_rad = tibia_angle + np.pi/2
        ankle_orientation_deg = np.degrees(ankle_orientation_rad)
        
        return hip_pos, knee_pos, ankle_pos, ankle_orientation_deg

class TPGMMGaitPredictor:
    """
    TP-GMM Gait Predictor with frame transformation using transformation matrices
    """
    def __init__(self, model_path):
        """
        Initialize TP-GMM predictor
        
        Args:
            model_path: Path to trained TP-GMM model
        """
        self.model_data = self._load_model(model_path)
        self.gmm_model = self.model_data['gmm_model']
        self.data_structure = self.model_data['data_structure']
        
        # Control parameters
        self.smoothing_factor = 0.7
        self.velocity_history = deque(maxlen=5)
        
        print(f"✓ TP-GMM Gait Predictor loaded:")
        print(f"  Components: {self.gmm_model.n_components}")
        print(f"  Dimensions: {self.data_structure['total_dim']}")
        print(f"  Frame 1: Hip coordinates")
        print(f"  Frame 2: Global coordinates (will be transformed to hip)")
        
    def _load_model(self, model_path):
        """Load trained TP-GMM model"""
        try:
            model_data = joblib.load(model_path)
            print(f"✓ Model loaded from: {model_path}")
            return model_data
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    def apply_inverse_transformation(self, position, velocity, orientation, A_matrix, b_vector):
        """
        Apply inverse transformation to convert from global frame to hip frame
        
        Args:
            position: [x, y] position in global frame
            velocity: [vx, vy] velocity in global frame  
            orientation: orientation angle in global frame (radians)
            A_matrix: 2x2 rotation matrix from hip to global
            b_vector: 2x1 translation vector from hip to global
            
        Returns:
            tuple: (transformed_position, transformed_velocity, transformed_orientation)
        """
        try:
            # Compute inverse transformation
            A_inv = np.linalg.inv(A_matrix)
            
            # Transform position: p_hip = A_inv * (p_global - b)
            pos_transformed = A_inv @ (position - b_vector)
            
            # Transform velocity: v_hip = A_inv * v_global
            vel_transformed = A_inv @ velocity
            
            # Transform orientation: theta_hip = theta_global - rotation_angle
            # Extract rotation angle from A matrix
            rotation_angle = np.arctan2(A_matrix[1, 0], A_matrix[0, 0])
            orientation_transformed = orientation - rotation_angle
            
            # Normalize orientation to [-pi, pi]
            orientation_transformed = np.arctan2(np.sin(orientation_transformed), 
                                               np.cos(orientation_transformed))
            
            return pos_transformed, vel_transformed, orientation_transformed
            
        except np.linalg.LinAlgError:
            # If matrix is not invertible, return original data
            print("Warning: Transformation matrix not invertible, using original data")
            return position, velocity, orientation
    
    def process_gait_state(self, gait_state):
        """
        Process gait state and apply transformations for TP-GMM prediction
        
        Args:
            gait_state: Dictionary with current gait data
            
        Returns:
            frame1_state: State in Frame 1 (hip coordinates) [x, y, vx, vy, orientation]
            frame2_state: State in Frame 2 (global->hip transformed) [x, y, vx, vy, orientation]
        """
        # Frame 1 data (already in hip coordinates)
        pos_fr1 = gait_state['ankle_pos_FR1']
        vel_fr1 = gait_state['ankle_vel_FR1'] 
        orient_fr1 = np.radians(gait_state['ankle_orient_FR1'])
        
        frame1_state = np.array([
            pos_fr1[0], pos_fr1[1],
            vel_fr1[0], vel_fr1[1],
            orient_fr1
        ])
        
        # Frame 2 data (global coordinates, need transformation)
        pos_fr2 = gait_state['ankle_pos_FR2']
        vel_fr2 = gait_state['ankle_vel_FR2']
        orient_fr2 = np.radians(gait_state['ankle_orient_FR2'])
        
        # Apply inverse transformation using A_FR2 and b_FR2
        A_FR2 = gait_state['A_FR2']
        b_FR2 = gait_state['b_FR2']
        
        pos_fr2_transformed, vel_fr2_transformed, orient_fr2_transformed = self.apply_inverse_transformation(
            pos_fr2, vel_fr2, orient_fr2, A_FR2, b_FR2
        )
        
        frame2_state = np.array([
            pos_fr2_transformed[0], pos_fr2_transformed[1],
            vel_fr2_transformed[0], vel_fr2_transformed[1],
            orient_fr2_transformed
        ])
        
        return frame1_state, frame2_state
    
    def predict_next_state(self, current_frame1_state, current_frame2_state):
        """
        Predict next state using TP-GMM
        
        Args:
            current_frame1_state: [x, y, vx, vy, orientation] in Frame 1
            current_frame2_state: [x, y, vx, vy, orientation] in Frame 2 (transformed)
            
        Returns:
            predicted_frame1_state: Predicted state in Frame 1
            predicted_frame2_state: Predicted state in Frame 2
        """
        # Combine current states for TP-GMM input
        current_tp_state = np.concatenate([current_frame1_state, current_frame2_state])
        
        # Use GMR to predict next state
        predicted_state = self._gmr_prediction(current_tp_state)
        
        # Split prediction into frames
        predicted_frame1_state = predicted_state[:5]
        predicted_frame2_state = predicted_state[5:]
        
        return predicted_frame1_state, predicted_frame2_state
    
    def _gmr_prediction(self, current_state):
        """
        Gaussian Mixture Regression prediction
        
        Args:
            current_state: Current TP-GMM state [10D]
            
        Returns:
            predicted_state: Predicted next state [10D]
        """
        n_components = self.gmm_model.n_components
        means = self.gmm_model.means_
        covariances = self.gmm_model.covariances_
        weights = self.gmm_model.weights_
        
        # Calculate responsibilities
        responsibilities = np.zeros(n_components)
        
        for i in range(n_components):
            try:
                # Use position dimensions for responsibility calculation
                pos_dims = [1, 2, 6, 7]  # x, y positions from both frames (skip time dim)
                current_pos = current_state[[0, 1, 5, 6]]  # Positions from input state
                mean_pos = means[i, pos_dims]
                cov_pos = covariances[i][np.ix_(pos_dims, pos_dims)]
                
                # Add regularization
                cov_pos += np.eye(len(pos_dims)) * 1e-6
                
                # Calculate responsibility
                diff = current_pos - mean_pos
                inv_cov = np.linalg.inv(cov_pos)
                mahalanobis = diff.T @ inv_cov @ diff
                
                det_cov = np.linalg.det(cov_pos)
                if det_cov > 1e-10:
                    prob = np.exp(-0.5 * mahalanobis) / np.sqrt((2 * np.pi)**len(pos_dims) * det_cov)
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
        
        # Weighted prediction
        predicted_state = np.zeros(10)
        for i in range(n_components):
            if responsibilities[i] > 1e-10:
                predicted_state += responsibilities[i] * means[i, 1:11]  # Skip time dimension
        
        return predicted_state

class GaitDataLoader:
    """
    Load and manage gait data from MAT file with new structure
    """
    def __init__(self, mat_file_path):
        """
        Initialize gait data loader
        
        Args:
            mat_file_path: Path to MAT file with gait data
        """
        self.mat_file_path = mat_file_path
        self.gait_data = self._load_data()
        self.current_sample = 0
        self.current_point = 0
        
    def _load_data(self):
        """Load gait data from MAT file"""
        try:
            mat_data = scipy.io.loadmat(self.mat_file_path)
            # Try different possible field names
            if 'processed_gait_data' in mat_data:
                gait_data = mat_data['processed_gait_data']
            elif 'output_struct_array' in mat_data:
                gait_data = mat_data['output_struct_array']
            else:
                # Print available fields to help debug
                print("Available fields in MAT file:")
                for key in mat_data.keys():
                    if not key.startswith('__'):
                        print(f"  - {key}")
                raise KeyError("Could not find gait data field")
            
            print(f"✓ Gait data loaded from: {self.mat_file_path}")
            print(f"  Number of samples: {gait_data.shape[1]}")
            
            # Print structure of first sample
            if gait_data.shape[1] > 0:
                sample = gait_data[0, 0][0, 0] if gait_data[0, 0].ndim > 0 else gait_data[0, 0]
                print("  Data structure:")
                for field in sample.dtype.names:
                    field_data = sample[field]
                    if hasattr(field_data, 'shape'):
                        print(f"    {field}: {field_data.shape}")
                    else:
                        print(f"    {field}: {type(field_data)}")
            
            return gait_data
        except Exception as e:
            print(f"✗ Error loading gait data: {e}")
            raise
    
    def get_current_gait_state(self):
        """
        Get current gait state with ankle data and transformations
        
        Returns:
            gait_state: Dictionary with current gait data
            is_end: Boolean indicating end of trajectory
        """
        if self.current_sample >= self.gait_data.shape[1]:
            return None, True
        
        # Get current sample
        sample = self.gait_data[0, self.current_sample]
        if sample.ndim > 0:
            sample = sample[0, 0]
        
        # Extract time vector to check length
        time = sample['time'].flatten()
        
        if self.current_point >= len(time):
            # Move to next sample
            self.current_sample += 1
            self.current_point = 0
            
            if self.current_sample >= self.gait_data.shape[1]:
                return None, True
            
            # Get new sample
            sample = self.gait_data[0, self.current_sample]
            if sample.ndim > 0:
                sample = sample[0, 0]
            time = sample['time'].flatten()
        
        # Extract all data for current time point
        gait_state = {
            'time': time[self.current_point],
            'pelvis_orientation': sample['pelvis_orientation'].flatten()[self.current_point],
            
            # Ankle data (Frame 1 - hip frame)
            'ankle_pos_FR1': sample['ankle_pos_FR1'][self.current_point, :],
            'ankle_vel_FR1': sample['ankle_pos_FR1_velocity'][self.current_point, :],
            'ankle_orient_FR1': sample['ankle_orientation_FR1'].flatten()[self.current_point],
            
            # Ankle data (Frame 2 - global frame)
            'ankle_pos_FR2': sample['ankle_pos_FR2'][self.current_point, :],
            'ankle_vel_FR2': sample['ankle_pos_FR2_velocity'][self.current_point, :],
            'ankle_orient_FR2': sample['ankle_orientation_FR2'].flatten()[self.current_point],
            
            # Transformation matrices
            'A_FR1': sample['ankle_A_FR1'][self.current_point, :, :],
            'b_FR1': sample['ankle_b_FR1'][self.current_point, :],
            'A_FR2': sample['ankle_A_FR2'][self.current_point, :, :],
            'b_FR2': sample['ankle_b_FR2'][self.current_point, :],
        }
        
        # Advance point
        self.current_point += 1
        
        return gait_state, False
    
    def reset(self):
        """Reset to beginning of data"""
        self.current_sample = 0
        self.current_point = 0

class GaitVisualizer:
    """
    Real-time visualization of gait prediction
    """
    def __init__(self):
        """Initialize visualizer"""
        plt.ion()  # Interactive mode
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Setup leg visualization
        self.ax1.set_xlim(-0.8, 0.2)
        self.ax1.set_ylim(-0.8, 0.2)
        self.ax1.set_aspect('equal')
        self.ax1.set_title('Leg Configuration')
        self.ax1.grid(True, alpha=0.3)
        
        # Setup trajectory visualization
        self.ax2.set_xlim(-0.6, 0.6)
        self.ax2.set_ylim(-0.6, 0.6)
        self.ax2.set_aspect('equal')
        self.ax2.set_title('Ankle Trajectory (Frame 1: Heel Strike Origin)')
        self.ax2.grid(True, alpha=0.3)
        
        # Initialize plot elements
        self.leg_line, = self.ax1.plot([], [], 'b-', linewidth=3, label='Current')
        self.predicted_leg_line, = self.ax1.plot([], [], 'r--', linewidth=2, label='Predicted')
        self.hip_point, = self.ax1.plot([], [], 'ko', markersize=8)
        self.knee_point, = self.ax1.plot([], [], 'ro', markersize=6)
        self.ankle_point, = self.ax1.plot([], [], 'go', markersize=6)
        
        # Trajectory plots
        self.actual_traj, = self.ax2.plot([], [], 'b-', linewidth=2, label='Actual')
        self.predicted_traj, = self.ax2.plot([], [], 'r--', linewidth=2, label='Predicted')
        self.current_pos, = self.ax2.plot([], [], 'go', markersize=8, label='Current')
        self.predicted_pos, = self.ax2.plot([], [], 'ro', markersize=8, label='Predicted Next')
        
        # Add legends
        self.ax1.legend()
        self.ax2.legend()
        
        # Trajectory storage
        self.actual_trajectory = []
        self.predicted_trajectory = []
        
        plt.tight_layout()
        plt.show()
    
    def update_visualization(self, gait_state, predicted_frame1_state, predicted_frame2_state):
        """
        Update real-time visualization
        
        Args:
            gait_state: Current gait state dictionary
            predicted_frame1_state: Predicted state in Frame 1
            predicted_frame2_state: Predicted state in Frame 2
        """
        # Current ankle positions
        ankle_pos_fr1 = gait_state['ankle_pos_FR1']
        ankle_pos_fr2 = gait_state['ankle_pos_FR2']
        
        # For leg visualization, create simple leg structure
        # Using Frame 1 data (hip coordinates)
        hip_pos = np.array([0, 0])  # Hip at origin
        
        # Estimate knee position (simplified - halfway to ankle)
        knee_pos = ankle_pos_fr1 / 2
        
        # Current leg configuration
        self.leg_line.set_data([hip_pos[0], knee_pos[0], ankle_pos_fr1[0]], 
                              [hip_pos[1], knee_pos[1], ankle_pos_fr1[1]])
        self.hip_point.set_data([hip_pos[0]], [hip_pos[1]])
        self.knee_point.set_data([knee_pos[0]], [knee_pos[1]])
        self.ankle_point.set_data([ankle_pos_fr1[0]], [ankle_pos_fr1[1]])
        
        # Predicted leg (using predicted Frame 1 state)
        predicted_ankle_pos = predicted_frame1_state[:2]
        predicted_knee_pos = predicted_ankle_pos / 2  # Simplified
        
        self.predicted_leg_line.set_data([hip_pos[0], predicted_knee_pos[0], predicted_ankle_pos[0]], 
                                        [hip_pos[1], predicted_knee_pos[1], predicted_ankle_pos[1]])
        
        # Update trajectories (using Frame 1 coordinates)
        self.actual_trajectory.append(ankle_pos_fr1.copy())
        self.predicted_trajectory.append(predicted_frame1_state[:2].copy())
        
        if len(self.actual_trajectory) > 200:  # Limit trajectory length
            self.actual_trajectory.pop(0)
            self.predicted_trajectory.pop(0)
        
        # Plot trajectories
        if len(self.actual_trajectory) > 1:
            actual_array = np.array(self.actual_trajectory)
            predicted_array = np.array(self.predicted_trajectory)
            
            self.actual_traj.set_data(actual_array[:, 0], actual_array[:, 1])
            self.predicted_traj.set_data(predicted_array[:, 0], predicted_array[:, 1])
        
        # Current positions
        self.current_pos.set_data([ankle_pos_fr1[0]], [ankle_pos_fr1[1]])
        self.predicted_pos.set_data([predicted_frame1_state[0]], [predicted_frame1_state[1]])
        
        # Update titles with current information
        self.ax1.set_title(f'Leg Configuration (Hip Frame)\nPelvis: {np.degrees(gait_state["pelvis_orientation"]):.1f}°')
        self.ax2.set_title(f'Ankle Trajectory (Frame 1: Hip Coordinates)\nSample {self.current_sample+1}')
        
        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def set_current_sample(self, sample_num):
        """Set current sample number for display"""
        self.current_sample = sample_num
    
    def close(self):
        """Close visualization"""
        plt.close(self.fig)

def main():
    """
    Main function for TP-GMM gait real-time regression
    """
    print("=== TP-GMM Gait Real-time Regression ===")
    print("Commands:")
    print("  ENTER or 'space' = Next data point")
    print("  'q' or 'quit' = Quit")
    print("  'r' or 'reset' = Reset data to beginning")
    print("  'h' or 'help' = Show help")
    
    # Configuration
    model_path = 'tpgmm_gait_model_updated.pkl'
    gait_data_path = 'new_processed_gait_data.mat'  # Updated for new data structure
    
    # Initialize input handler
    input_handler = InputHandler()
    
    try:
        # Initialize components
        print("\n--- Initializing components ---")
        predictor = TPGMMGaitPredictor(model_path)
        data_loader = GaitDataLoader(gait_data_path)
        visualizer = GaitVisualizer()
        
        # Start input handling
        input_handler.start_input_thread()
        
        print("\n--- Starting real-time prediction ---")
        print("Ready! Type commands and press ENTER:")
        
        running = True
        while running:
            try:
                # Display current status
                print(f"\nSample {data_loader.current_sample + 1}, Point {data_loader.current_point + 1}")
                print("Command (ENTER=next, q=quit, r=reset, h=help): ", end='', flush=True)
                
                # Wait for command
                command = None
                while command is None and running:
                    command = input_handler.get_command()
                    if command is None:
                        time.sleep(0.1)
                
                if command == 'q':
                    running = False
                    break
                elif command == 'r':
                    data_loader.reset()
                    visualizer.actual_trajectory.clear()
                    visualizer.predicted_trajectory.clear()
                    print("\n✓ Data reset to beginning")
                    continue
                elif command == 'h':
                    print("\nCommands:")
                    print("  ENTER or 'space' = Next data point")
                    print("  'q' or 'quit' = Quit")
                    print("  'r' or 'reset' = Reset data to beginning")
                    print("  'h' or 'help' = Show help")
                    continue
                elif command == 'space':
                    pass  # Continue to next data point
                else:
                    continue  # Unknown command, ask again
                
                # Get current gait state
                gait_state, is_end = data_loader.get_current_gait_state()
                
                if is_end:
                    print("\n✓ End of data reached")
                    print("Type 'r' to reset or 'q' to quit")
                    continue
                
                # Process gait state and apply transformations
                frame1_state, frame2_state = predictor.process_gait_state(gait_state)
                
                print(f"\nTime: {gait_state['time']:.3f}")
                print(f"Pelvis orientation: {np.degrees(gait_state['pelvis_orientation']):.1f}°")
                print(f"Ankle FR1: [{gait_state['ankle_pos_FR1'][0]:.3f}, {gait_state['ankle_pos_FR1'][1]:.3f}]")
                print(f"Ankle FR2: [{gait_state['ankle_pos_FR2'][0]:.3f}, {gait_state['ankle_pos_FR2'][1]:.3f}]")
                print(f"Frame2 transformed: [{frame2_state[0]:.3f}, {frame2_state[1]:.3f}]")
                
                # Predict next state using TP-GMM
                predicted_frame1_state, predicted_frame2_state = predictor.predict_next_state(
                    frame1_state, frame2_state
                )
                
                print(f"Predicted Frame1: [{predicted_frame1_state[0]:.3f}, {predicted_frame1_state[1]:.3f}]")
                print(f"Predicted Frame2: [{predicted_frame2_state[0]:.3f}, {predicted_frame2_state[1]:.3f}]")
                
                # Update visualization
                visualizer.set_current_sample(data_loader.current_sample)
                visualizer.update_visualization(gait_state, predicted_frame1_state, predicted_frame2_state)
                
                # Show transformation matrix info (optional debug)
                A_FR2 = gait_state['A_FR2']
                det_A = np.linalg.det(A_FR2)
                print(f"Transformation matrix determinant: {det_A:.6f}")
                
                # Small delay for visualization
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                print("\n⚠ Interrupted by user")
                break
            except Exception as e:
                print(f"✗ Error in main loop: {e}")
                import traceback
                traceback.print_exc()
                break
        
    except Exception as e:
        print(f"✗ Error in initialization: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n--- Cleaning up ---")
        input_handler.stop()
        if 'visualizer' in locals():
            visualizer.close()
        print("✓ Program ended")

if __name__ == "__main__":
    main()