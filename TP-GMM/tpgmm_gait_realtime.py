import numpy as np
import joblib
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import time
from collections import deque
from sklearn.mixture import GaussianMixture

class GaitForwardKinematics:
    """
    Forward kinematics for gait analysis
    """
    def __init__(self, l1=0.4135, l2=0.39):
        """
        Initialize forward kinematics
        
        Args:
            l1: Length of first link (hip to knee) in meters
            l2: Length of second link (knee to ankle) in meters
        """
        self.l1 = l1
        self.l2 = l2
        
    def calculate_forward_kinematics(self, hip_angle_deg, knee_angle_deg, pelvis_angle_deg=0):
        """
        Calculate forward kinematics for 2-link leg with pelvis orientation
        
        Args:
            hip_angle_deg: Hip joint angle (degrees)
            knee_angle_deg: Knee joint angle (degrees) 
            pelvis_angle_deg: Pelvis orientation angle (degrees)
            
        Returns:
            dict: positions and orientations
                - hip_pos: Hip position (origin at 0,0)
                - knee_pos: Knee position
                - ankle_pos: Ankle position
                - foot_orientation: Foot orientation angle (degrees)
        """
        # Convert degrees to radians
        hip_angle = np.radians(hip_angle_deg)
        knee_angle = np.radians(knee_angle_deg)
        pelvis_angle = np.radians(pelvis_angle_deg)
        
        # Hip is at origin (considering pelvis orientation)
        hip_pos = np.array([0, 0])
        
        # Knee position (hip angle relative to pelvis)
        knee_x = self.l1 * np.cos(hip_angle + pelvis_angle)
        knee_y = self.l1 * np.sin(hip_angle + pelvis_angle)
        knee_pos = np.array([knee_x, knee_y])
        
        # Ankle position (knee angle is relative to femur)
        ankle_x = knee_x + self.l2 * np.cos(hip_angle + knee_angle + pelvis_angle)
        ankle_y = knee_y + self.l2 * np.sin(hip_angle + knee_angle + pelvis_angle)
        ankle_pos = np.array([ankle_x, ankle_y])
        
        # Foot orientation (+90 degrees from tibia direction)
        tibia_angle = hip_angle + knee_angle + pelvis_angle
        foot_orientation_deg = np.degrees(tibia_angle) + 90
        
        return {
            'hip_pos': hip_pos,
            'knee_pos': knee_pos,
            'ankle_pos': ankle_pos,
            'foot_orientation': foot_orientation_deg
        }

class TPGMMGaitPredictor:
    """
    TP-GMM Gait Predictor - simplified for real-time gait analysis
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
        self.smoothing_factor = 0.7
        
        # History for smoothing
        self.prediction_history = deque(maxlen=5)
        
        # Forward kinematics
        self.fk = GaitForwardKinematics()
        
        # Frame information from model
        self.data_structure = self.model_data['data_structure']
        self.frame_info = self.model_data['frame_info']
        
        # Diagnostics
        self.diagnostics = {
            'likelihoods': [],
            'predictions': [],
            'inputs': []
        }
        
        print(f"✓ TP-GMM Gait Predictor loaded:")
        print(f"  Components: {self.gmm_model.n_components}")
        print(f"  Data structure: {self.data_structure}")
        
    def _load_model(self, model_path):
        """Load TP-GMM model"""
        try:
            model_data = joblib.load(model_path)
            print(f"✓ Model loaded from: {model_path}")
            return model_data
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    def preprocess_joint_angles(self, hip_angle_raw, knee_angle_raw):
        """
        Preprocess joint angles (same as gait_kinematics.py)
        
        Args:
            hip_angle_raw: Raw hip angle (degrees)
            knee_angle_raw: Raw knee angle (degrees)
            
        Returns:
            tuple: (processed_hip_angle, processed_knee_angle)
        """
        # Process hip position: subtract 90 degrees  
        hip_angle = hip_angle_raw - 90
        
        # Process knee position: invert signal
        knee_angle = -knee_angle_raw
        
        return hip_angle, knee_angle
    
    def calculate_ankle_trajectory_frames(self, hip_angle, knee_angle, pelvis_angle=0):
        """
        Calculate ankle position in both reference frames
        
        Args:
            hip_angle: Hip angle (degrees, processed)
            knee_angle: Knee angle (degrees, processed)
            pelvis_angle: Pelvis orientation (degrees)
            
        Returns:
            dict: ankle positions in both frames
        """
        # Calculate forward kinematics
        fk_result = self.fk.calculate_forward_kinematics(hip_angle, knee_angle, pelvis_angle)
        ankle_pos_global = fk_result['ankle_pos']
        foot_orientation = fk_result['foot_orientation']
        
        # Frame 1: Euler coordinates with origin at heel strike (final point)
        # For now, we'll use the current position as Frame 1
        # In a complete implementation, you'd need the heel strike reference
        frame1_pos = ankle_pos_global.copy()
        frame1_orientation = foot_orientation
        
        # Frame 2: Hip-centered coordinates  
        # Transform to hip-centered frame
        hip_pos = fk_result['hip_pos']
        frame2_pos = ankle_pos_global - hip_pos  # Relative to hip
        frame2_orientation = foot_orientation
        
        return {
            'frame1': {
                'position': frame1_pos,
                'orientation': frame1_orientation
            },
            'frame2': {
                'position': frame2_pos,
                'orientation': frame2_orientation
            },
            'global_ankle': ankle_pos_global,
            'foot_orientation': foot_orientation
        }
    
    def predict_from_current_state(self, hip_angle_raw, knee_angle_raw, pelvis_angle=0):
        """
        Predict next state using TP-GMM
        
        Args:
            hip_angle_raw: Raw hip angle (degrees)
            knee_angle_raw: Raw knee angle (degrees)
            pelvis_angle: Pelvis orientation (degrees)
            
        Returns:
            dict: prediction results
        """
        # Preprocess angles
        hip_angle, knee_angle = self.preprocess_joint_angles(hip_angle_raw, knee_angle_raw)
        
        # Calculate current state in both frames
        current_frames = self.calculate_ankle_trajectory_frames(hip_angle, knee_angle, pelvis_angle)
        
        # Create input vector for GMM (using Frame 2 data as primary input)
        # Based on model structure: [x, y, vx, vy, orientation] for each frame
        frame2_pos = current_frames['frame2']['position']
        frame2_orient = np.radians(current_frames['frame2']['orientation'])
        
        # For velocity, use previous prediction if available, otherwise zero
        if len(self.prediction_history) > 0:
            prev_frame2_pos = self.prediction_history[-1]['frame2']['position']
            frame2_vel = frame2_pos - prev_frame2_pos  # Simple velocity estimation
        else:
            frame2_vel = np.zeros(2)
        
        # Construct input for Frame 2: [x, y, vx, vy, orientation]
        frame2_input = np.array([
            frame2_pos[0], frame2_pos[1],
            frame2_vel[0], frame2_vel[1],
            frame2_orient
        ])
        
        # For Frame 1, use similar approach
        frame1_pos = current_frames['frame1']['position']
        frame1_orient = np.radians(current_frames['frame1']['orientation'])
        
        if len(self.prediction_history) > 0:
            prev_frame1_pos = self.prediction_history[-1]['frame1']['position']
            frame1_vel = frame1_pos - prev_frame1_pos
        else:
            frame1_vel = np.zeros(2)
            
        frame1_input = np.array([
            frame1_pos[0], frame1_pos[1],
            frame1_vel[0], frame1_vel[1], 
            frame1_orient
        ])
        
        # Full input vector: [Frame1_data | Frame2_data] (10D)
        full_input = np.concatenate([frame1_input, frame2_input])
        
        # Use GMM for prediction (without time dimension for now)
        predicted_output = self._gmr_prediction(full_input)
        
        # Apply smoothing
        if len(self.prediction_history) > 0:
            alpha = self.smoothing_factor
            prev_pred = self.prediction_history[-1]
            
            # Smooth Frame 1 prediction
            predicted_output['frame1']['position'] = (
                alpha * predicted_output['frame1']['position'] + 
                (1 - alpha) * prev_pred['frame1']['position']
            )
            
            # Smooth Frame 2 prediction
            predicted_output['frame2']['position'] = (
                alpha * predicted_output['frame2']['position'] + 
                (1 - alpha) * prev_pred['frame2']['position']
            )
        
        # Store in history
        self.prediction_history.append(predicted_output)
        
        # Store diagnostics
        self.diagnostics['inputs'].append(full_input)
        self.diagnostics['predictions'].append(predicted_output)
        
        return {
            'current_state': current_frames,
            'prediction': predicted_output,
            'input_vector': full_input
        }
    
    def _gmr_prediction(self, input_vector):
        """
        Gaussian Mixture Regression for prediction
        
        Args:
            input_vector: Current state vector (10D)
            
        Returns:
            dict: predicted next state
        """
        # For simplicity, we'll predict the next position based on current state
        # In a full implementation, you'd use proper GMR with conditional probabilities
        
        n_components = self.gmm_model.n_components
        means = self.gmm_model.means_
        covariances = self.gmm_model.covariances_
        weights = self.gmm_model.weights_
        
        # Calculate responsibilities (simplified approach)
        responsibilities = np.zeros(n_components)
        
        for i in range(n_components):
            try:
                # Use only position dimensions for responsibility calculation
                pos_dims = [0, 1, 5, 6]  # x,y positions from both frames
                mean_pos = means[i, 1:][pos_dims]  # Skip time dimension
                input_pos = input_vector[pos_dims]
                
                cov_pos = covariances[i][1:, 1:][np.ix_(pos_dims, pos_dims)]  # Skip time
                cov_pos += np.eye(len(pos_dims)) * 1e-6
                
                diff = input_pos - mean_pos
                inv_cov = np.linalg.inv(cov_pos)
                mahalanobis = diff.T @ inv_cov @ diff
                
                det_cov = np.linalg.det(cov_pos)
                if det_cov > 1e-10:
                    prob = np.exp(-0.5 * mahalanobis) / np.sqrt((2 * np.pi)**len(pos_dims) * det_cov)
                    responsibilities[i] = weights[i] * prob
                    
            except np.linalg.LinAlgError:
                responsibilities[i] = 0.0
        
        # Normalize responsibilities
        total_resp = np.sum(responsibilities)
        if total_resp > 1e-10:
            responsibilities /= total_resp
        else:
            responsibilities = weights.copy()
        
        # Predict next state (simplified - using weighted average of means)
        predicted_state = np.zeros(10)  # 10D state vector
        
        for i in range(n_components):
            if responsibilities[i] > 1e-10:
                # Use mean of component (skip time dimension)
                component_mean = means[i, 1:]  # Skip time dimension
                predicted_state += responsibilities[i] * component_mean
        
        # Structure prediction result
        prediction = {
            'frame1': {
                'position': predicted_state[0:2],
                'velocity': predicted_state[2:4],
                'orientation': predicted_state[4]
            },
            'frame2': {
                'position': predicted_state[5:7],
                'velocity': predicted_state[7:9],
                'orientation': predicted_state[9]
            }
        }
        
        return prediction

class GaitDataLoader:
    """
    Load and manage gait data from MAT file
    """
    def __init__(self, mat_file_path):
        """
        Initialize gait data loader
        
        Args:
            mat_file_path: Path to MAT file with gait data
        """
        self.mat_file_path = mat_file_path
        self.current_trial = 0
        self.current_step = 0
        self.gait_data = None
        self.load_data()
        
    def load_data(self):
        """Load gait data from MAT file"""
        try:
            mat_data = scipy.io.loadmat(self.mat_file_path)
            # Assuming the data structure is similar to demo_gait_data_angular_10_samples.mat
            self.gait_data = mat_data['output_struct_array']
            print(f"✓ Loaded gait data from: {self.mat_file_path}")
            print(f"  Number of trials: {self.gait_data.shape[1]}")
            return True
        except Exception as e:
            print(f"✗ Error loading gait data: {e}")
            return False
    
    def get_current_angles(self):
        """
        Get current joint angles
        
        Returns:
            dict: current angles or None if end of data
        """
        if self.gait_data is None:
            return None
            
        if self.current_trial >= self.gait_data.shape[1]:
            print("End of all trials")
            return None
            
        # Get current trial data
        trial = self.gait_data[0, self.current_trial]
        hip_pos_raw = trial['hip_pos'][0, 0].flatten()
        knee_pos_raw = trial['knee_pos'][0, 0].flatten()
        
        if self.current_step >= len(hip_pos_raw):
            print(f"End of trial {self.current_trial + 1}")
            return None
            
        # Get current step data
        hip_angle = hip_pos_raw[self.current_step]
        knee_angle = knee_pos_raw[self.current_step]
        
        return {
            'hip_angle': hip_angle,
            'knee_angle': knee_angle,
            'trial': self.current_trial,
            'step': self.current_step,
            'total_steps': len(hip_pos_raw)
        }
    
    def next_step(self):
        """Move to next step"""
        self.current_step += 1
        
    def next_trial(self):
        """Move to next trial"""
        self.current_trial += 1
        self.current_step = 0
    
    def reset(self):
        """Reset to beginning"""
        self.current_trial = 0
        self.current_step = 0

class GaitVisualizer:
    """
    Real-time visualization of gait analysis
    """
    def __init__(self, predictor):
        """
        Initialize visualizer
        
        Args:
            predictor: TP-GMM gait predictor
        """
        self.predictor = predictor
        self.fig = None
        self.axes = None
        
        # History for trajectory plotting
        self.input_trajectory = []
        self.predicted_trajectory_f1 = []
        self.predicted_trajectory_f2 = []
        
        # Control state
        self.paused = True
        self.step_requested = False
        self.next_trial_requested = False
        self.reset_requested = False
        self.quit_requested = False
        
        self.init_plot()
        
    def init_plot(self):
        """Initialize matplotlib plots"""
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 10))
        self.fig.suptitle('TP-GMM Gait Real-time Analysis\nPress SPACE: next step, N: next trial, R: reset, Q: quit', fontsize=14)
        
        # Configure subplots
        self.axes[0, 0].set_title('Current Leg Configuration')
        self.axes[0, 0].set_xlabel('X Position (m)')
        self.axes[0, 0].set_ylabel('Y Position (m)')
        self.axes[0, 0].grid(True)
        self.axes[0, 0].set_aspect('equal')
        
        self.axes[0, 1].set_title('Ankle Trajectory - Input vs Prediction')
        self.axes[0, 1].set_xlabel('X Position (m)')
        self.axes[0, 1].set_ylabel('Y Position (m)')
        self.axes[0, 1].grid(True)
        self.axes[0, 1].set_aspect('equal')
        
        self.axes[1, 0].set_title('Frame 1 Trajectory')
        self.axes[1, 0].set_xlabel('X Position (m)')
        self.axes[1, 0].set_ylabel('Y Position (m)')
        self.axes[1, 0].grid(True)
        self.axes[1, 0].set_aspect('equal')
        
        self.axes[1, 1].set_title('Frame 2 Trajectory')
        self.axes[1, 1].set_xlabel('X Position (m)')
        self.axes[1, 1].set_ylabel('Y Position (m)')
        self.axes[1, 1].grid(True)
        self.axes[1, 1].set_aspect('equal')
        
        # Connect key press events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        plt.tight_layout()
        plt.ion()  # Interactive mode
        plt.show()
    
    def on_key_press(self, event):
        """Handle key press events"""
        if event.key == ' ':  # Space bar
            self.step_requested = True
        elif event.key == 'n':
            self.next_trial_requested = True
        elif event.key == 'r':
            self.reset_requested = True
        elif event.key == 'q':
            self.quit_requested = True
        
        print(f"Key pressed: {event.key}")
    
    def update_visualization(self, current_angles, prediction_result):
        """
        Update real-time visualization
        
        Args:
            current_angles: Current joint angles
            prediction_result: Prediction result from TP-GMM
        """
        # Clear previous plots
        for ax in self.axes.flat:
            ax.clear()
        
        # Re-configure subplots
        self.init_plot_config()
        
        # Get data
        current_state = prediction_result['current_state']
        prediction = prediction_result['prediction']
        
        # Update trajectory histories
        self.input_trajectory.append(current_state['global_ankle'])
        self.predicted_trajectory_f1.append(prediction['frame1']['position'])
        self.predicted_trajectory_f2.append(prediction['frame2']['position'])
        
        # Plot 1: Current leg configuration
        self.plot_leg_configuration(current_angles, current_state)
        
        # Plot 2: Input vs predicted trajectory
        self.plot_trajectory_comparison(current_state, prediction)
        
        # Plot 3: Frame 1 trajectory
        self.plot_frame_trajectory(1, self.predicted_trajectory_f1)
        
        # Plot 4: Frame 2 trajectory
        self.plot_frame_trajectory(2, self.predicted_trajectory_f2)
        
        # Update display
        plt.draw()
        plt.pause(0.01)
    
    def init_plot_config(self):
        """Re-initialize plot configuration"""
        titles = [
            'Current Leg Configuration',
            'Ankle Trajectory - Input vs Prediction', 
            'Frame 1 Trajectory',
            'Frame 2 Trajectory'
        ]
        
        for i, ax in enumerate(self.axes.flat):
            ax.set_title(titles[i])
            ax.set_xlabel('X Position (m)')
            ax.set_ylabel('Y Position (m)')
            ax.grid(True)
            ax.set_aspect('equal')
    
    def plot_leg_configuration(self, current_angles, current_state):
        """Plot current leg configuration"""
        ax = self.axes[0, 0]
        
        # Calculate leg segments
        hip_angle, knee_angle = self.predictor.preprocess_joint_angles(
            current_angles['hip_angle'], current_angles['knee_angle']
        )
        
        fk_result = self.predictor.fk.calculate_forward_kinematics(hip_angle, knee_angle)
        
        # Plot leg segments
        hip_pos = fk_result['hip_pos']
        knee_pos = fk_result['knee_pos'] 
        ankle_pos = fk_result['ankle_pos']
        
        # Femur (hip to knee)
        ax.plot([hip_pos[0], knee_pos[0]], [hip_pos[1], knee_pos[1]], 
                'b-', linewidth=4, label='Femur')
        
        # Tibia (knee to ankle)
        ax.plot([knee_pos[0], ankle_pos[0]], [knee_pos[1], ankle_pos[1]], 
                'r-', linewidth=4, label='Tibia')
        
        # Joints
        ax.plot(hip_pos[0], hip_pos[1], 'ko', markersize=10, label='Hip')
        ax.plot(knee_pos[0], knee_pos[1], 'go', markersize=8, label='Knee')
        ax.plot(ankle_pos[0], ankle_pos[1], 'ro', markersize=6, label='Ankle')
        
        ax.legend()
        ax.set_title(f'Leg Config - Hip: {current_angles["hip_angle"]:.1f}°, Knee: {current_angles["knee_angle"]:.1f}°')
    
    def plot_trajectory_comparison(self, current_state, prediction):
        """Plot input vs predicted trajectory"""
        ax = self.axes[0, 1]
        
        if len(self.input_trajectory) > 1:
            # Plot input trajectory
            input_traj = np.array(self.input_trajectory)
            ax.plot(input_traj[:, 0], input_traj[:, 1], 'b-', linewidth=2, label='Input Trajectory')
            ax.plot(input_traj[-1, 0], input_traj[-1, 1], 'bo', markersize=8, label='Current')
        
        # Plot predicted next point
        pred_ankle = prediction['frame1']['position']  # Use frame 1 as main prediction
        ax.plot(pred_ankle[0], pred_ankle[1], 'ro', markersize=8, label='Predicted Next')
        
        ax.legend()
    
    def plot_frame_trajectory(self, frame_num, trajectory_data):
        """Plot trajectory for specific frame"""
        if frame_num == 1:
            ax = self.axes[1, 0]
            color = 'blue'
        else:
            ax = self.axes[1, 1]
            color = 'red'
        
        if len(trajectory_data) > 1:
            traj = np.array(trajectory_data)
            ax.plot(traj[:, 0], traj[:, 1], f'{color[0]}-', linewidth=2, alpha=0.7)
            ax.plot(traj[-1, 0], traj[-1, 1], f'{color[0]}o', markersize=6)

def main():
    """
    Main function for TP-GMM gait real-time analysis
    """
    print("=== TP-GMM Gait Real-time Analysis ===")
    
    # Paths
    model_path = 'tpgmm_gait_model.pkl'  # Your trained model
    gait_data_path = 'demo_gait_data_angular_10_samples.mat'  # Your gait data
    
    try:
        # Initialize components
        print("Initializing TP-GMM predictor...")
        predictor = TPGMMGaitPredictor(model_path)
        
        print("Loading gait data...")
        data_loader = GaitDataLoader(gait_data_path)
        
        print("Initializing visualizer...")
        visualizer = GaitVisualizer(predictor)
        
        print("\n🚀 Real-time gait analysis started!")
        print("📋 Controls (click on plot window first):")
        print("  SPACE: Next step")
        print("  N: Next trial") 
        print("  R: Reset to beginning")
        print("  Q: Quit")
        print("\nClick on the plot window and press SPACE to begin...")
        
        # Main loop
        while True:
            # Check for user input through matplotlib events
            plt.pause(0.1)  # Allow GUI to process events
            
            if visualizer.quit_requested:
                print("Quitting...")
                break
            
            if visualizer.step_requested:
                visualizer.step_requested = False
                
                # Get current angles
                current_angles = data_loader.get_current_angles()
                
                if current_angles is None:
                    print("No more data. Press 'N' for next trial or 'R' to reset.")
                    continue
                
                print(f"\nTrial {current_angles['trial']+1}, Step {current_angles['step']+1}/{current_angles['total_steps']}")
                print(f"Hip: {current_angles['hip_angle']:.1f}°, Knee: {current_angles['knee_angle']:.1f}°")
                
                # Predict using TP-GMM
                prediction_result = predictor.predict_from_current_state(
                    current_angles['hip_angle'],
                    current_angles['knee_angle'],
                    pelvis_angle=0  # Can be extended later
                )
                
                # Update visualization
                visualizer.update_visualization(current_angles, prediction_result)
                
                # Move to next step
                data_loader.next_step()
            
            if visualizer.next_trial_requested:
                visualizer.next_trial_requested = False
                data_loader.next_trial()
                print(f"Moved to trial {data_loader.current_trial + 1}")
            
            if visualizer.reset_requested:
                visualizer.reset_requested = False
                data_loader.reset()
                visualizer.input_trajectory.clear()
                visualizer.predicted_trajectory_f1.clear()
                visualizer.predicted_trajectory_f2.clear()
                print("Reset to beginning")
        
        plt.close('all')
        print("✓ Program ended")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()