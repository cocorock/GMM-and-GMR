import numpy as np
import scipy.io
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import joblib
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

class TPGMMGaitTrainer:
    def __init__(self, reference_frame_id=2, target_frame_id=1):
        """
        TP-GMM Trainer for gait data
        
        Args:
            reference_frame_id: ID of the reference frame (FR2 - global frame)
            target_frame_id: ID of the target frame (FR1 - hip's frame)
        """
        self.reference_frame_id = reference_frame_id  # FR2 (global)
        self.target_frame_id = target_frame_id        # FR1 (hip)
        self.num_frames = 2  # Frame 1 + Frame 2
        
        # Dimensions for 2D gait data
        self.dims = {
            'position': 2,      # x, y
            'velocity': 2,      # vx, vy
            'orientation': 1    # orientation angle
        }
        self.point_dim = sum(self.dims.values())  # 5D per frame
        self.total_dim = self.point_dim * self.num_frames  # 10D total
        
    def load_gait_data(self, mat_file_path):
        """
        Loads and preprocesses the complete gait data from the specified .mat file for TP-GMM.
        Includes position, velocity, and orientation for both frames.
        
        Updated for new data structure where:
        - FR2 is the global frame of reference
        - FR1 is the hip's frame of reference (no rotation/translation needed)

        Args:
            mat_file_path (str): The path to the .mat file.

        Returns:
            tuple: (trajectories_fr1, trajectories_fr2, time_data, frame_transforms)
                   - trajectories_fr1: list of trajectories in frame 1 (hip frame) [time, x, y, vx, vy, orientation]
                   - trajectories_fr2: list of trajectories in frame 2 (global frame) [time, x, y, vx, vy, orientation]  
                   - time_data: list of time vectors for each trajectory
                   - frame_transforms: list of transformation data for each trajectory
        """
        try:
            mat_data = scipy.io.loadmat(mat_file_path)
            processed_gait_data = mat_data['processed_gait_data']
            
            trajectories_fr1 = []
            trajectories_fr2 = []
            time_data = []
            frame_transforms = []
            
            # The data is in a cell array, so we iterate through it
            for i in range(processed_gait_data.shape[1]):
                # The struct is often nested inside a 1x1 array within the cell
                trial_data = processed_gait_data[0, i][0, 0]
                
                # Extract all data fields
                time = trial_data['time'].flatten().astype(np.float64)
                
                # Frame 1 data (Hip's frame of reference)
                ankle_pos_fr1 = trial_data['ankle_pos_FR1'].astype(np.float64)  # [200x2]
                ankle_vel_fr1 = trial_data['ankle_pos_FR1_velocity'].astype(np.float64)  # [200x2]
                ankle_orient_fr1 = trial_data['ankle_orientation_FR1'].flatten().astype(np.float64)  # [200x1]
                
                # Frame 2 data (Global frame of reference)
                ankle_pos_fr2 = trial_data['ankle_pos_FR2'].astype(np.float64)  # [200x2]
                ankle_vel_fr2 = trial_data['ankle_pos_FR2_velocity'].astype(np.float64)  # [200x2]
                ankle_orient_fr2 = trial_data['ankle_orientation_FR2'].flatten().astype(np.float64)  # [200x1]
                
                # Extract transformation matrices (for analysis purposes)
                ankle_A_fr1 = trial_data['ankle_A_FR1'].astype(np.float64)  # [200x2x2] rotation matrices
                ankle_b_fr1 = trial_data['ankle_b_FR1'].astype(np.float64)  # [200x2] translation vectors
                ankle_A_fr2 = trial_data['ankle_A_FR2'].astype(np.float64)  # [200x2x2] rotation matrices
                ankle_b_fr2 = trial_data['ankle_b_FR2'].astype(np.float64)  # [200x2] translation vectors
                
                # Store transformation data for this trajectory
                transform_data = {
                    'A_FR1': ankle_A_fr1,  # Should be identity since FR1 is hip frame
                    'b_FR1': ankle_b_fr1,  # Should be zero since FR1 is hip frame
                    'A_FR2': ankle_A_fr2,  # Rotation from global to hip frame
                    'b_FR2': ankle_b_fr2   # Translation from global to hip frame
                }
                frame_transforms.append(transform_data)
                
                # Create complete trajectories for both frames
                # Frame 1 (Hip's frame): [time, x, y, vx, vy, orientation] = 6 dimensions
                trajectory_fr1 = np.column_stack([
                    time,
                    ankle_pos_fr1[:, 0],     # x position
                    ankle_pos_fr1[:, 1],     # y position  
                    ankle_vel_fr1[:, 0],     # x velocity
                    ankle_vel_fr1[:, 1],     # y velocity
                    ankle_orient_fr1         # orientation
                ])
                
                # Frame 2 (Global frame): [time, x, y, vx, vy, orientation] = 6 dimensions
                trajectory_fr2 = np.column_stack([
                    time,
                    ankle_pos_fr2[:, 0],     # x position
                    ankle_pos_fr2[:, 1],     # y position
                    ankle_vel_fr2[:, 0],     # x velocity
                    ankle_vel_fr2[:, 1],     # y velocity
                    ankle_orient_fr2         # orientation
                ])
                
                trajectories_fr1.append(trajectory_fr1)
                trajectories_fr2.append(trajectory_fr2)
                time_data.append(time)
                
            print(f"Loaded {len(trajectories_fr1)} trajectories")
            print(f"Each trajectory has {trajectory_fr1.shape[1]} dimensions: [time, x, y, vx, vy, orientation]")
            print(f"Trajectory length: {trajectory_fr1.shape[0]} time steps")
            print(f"Frame setup: FR2 (global) -> FR1 (hip)")
            print(f"Transformation matrices available for each trajectory")
                
            return trajectories_fr1, trajectories_fr2, time_data, frame_transforms
        except FileNotFoundError:
            print(f"Error: The file {mat_file_path} was not found.")
            return None, None, None, None
        except KeyError as e:
            print(f"Error: Could not find required field in the .mat file: {e}")
            print("Available fields:")
            if 'processed_gait_data' in mat_data:
                sample_data = mat_data['processed_gait_data'][0, 0][0, 0]
                for field in sample_data.dtype.names:
                    print(f"  - {field}")
            return None, None, None, None
    
    def load_demonstrations_from_mat(self, mat_file_path: str) -> List[np.ndarray]:
        """
        Load demonstrations from MAT file and process for TP-GMM
        
        Args:
            mat_file_path: Path to the .mat file
            
        Returns:
            List of numpy arrays with demonstrations
        """
        print(f"Loading demonstrations from: {mat_file_path}")
        
        # Load gait data
        trajectories_fr1, trajectories_fr2, time_data, frame_transforms = self.load_gait_data(mat_file_path)
        
        if trajectories_fr1 is None:
            print("Failed to load gait data")
            return []
        
        demonstrations = []
        
        # Process each trajectory
        for i, (traj_fr1, traj_fr2, transform_data) in enumerate(zip(trajectories_fr1, trajectories_fr2, frame_transforms)):
            print(f"Processing trajectory {i+1}/{len(trajectories_fr1)}")
            
            # Process trajectory for TP-GMM with transformations
            demo_array = self.process_gait_trajectory(traj_fr1, traj_fr2, transform_data)
            
            if demo_array is not None and len(demo_array) > 0:
                demonstrations.append(demo_array)
                print(f"✓ Trajectory processed: {len(demo_array)} points")
            else:
                print(f"✗ Invalid or empty trajectory")
                
        return demonstrations
    
    def apply_inverse_transformation(self, position: np.ndarray, velocity: np.ndarray, orientation: float, 
                                    A_matrix: np.ndarray, b_vector: np.ndarray) -> tuple:
        """
        Apply inverse transformation to convert from global frame to hip frame
        
        Args:
            position: [x, y] position in global frame
            velocity: [vx, vy] velocity in global frame  
            orientation: orientation angle in global frame
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
    
    def process_gait_trajectory(self, traj_fr1: np.ndarray, traj_fr2: np.ndarray, 
                              transform_data: dict = None) -> np.ndarray:
        """
        Process gait trajectory for TP-GMM format with inverse transformations
        
        Args:
            traj_fr1: Trajectory in frame 1 (hip frame) [time, x, y, vx, vy, orientation]
            traj_fr2: Trajectory in frame 2 (global frame) [time, x, y, vx, vy, orientation]
            transform_data: Dictionary with transformation matrices A_FR2 and b_FR2
            
        Returns:
            Array with TP-GMM data [N x 10] = [Frame1_data | Frame2_data_transformed]
        """
        valid_points = []
        
        # Check if both trajectories have the same length
        if len(traj_fr1) != len(traj_fr2):
            print(f"Warning: Trajectory lengths don't match: {len(traj_fr1)} vs {len(traj_fr2)}")
            min_len = min(len(traj_fr1), len(traj_fr2))
            traj_fr1 = traj_fr1[:min_len]
            traj_fr2 = traj_fr2[:min_len]
        
        # Extract transformation matrices if available
        use_transformation = transform_data is not None
        if use_transformation:
            A_FR2 = transform_data['A_FR2']  # [200x2x2]
            b_FR2 = transform_data['b_FR2']  # [200x2]
            print("Applying inverse transformations from global to hip frame")
        else:
            print("No transformation data provided, using raw coordinates")
        
        for i in range(len(traj_fr1)):
            try:
                # FRAME 1 (Hip's frame): Extract data (skip time column)
                # [x, y, vx, vy, orientation] = 5D
                frame1_data = traj_fr1[i, 1:6]  # Skip time column
                
                # FRAME 2 (Global frame): Extract data (skip time column)
                # [x, y, vx, vy, orientation] = 5D
                frame2_raw = traj_fr2[i, 1:6]  # Skip time column
                
                # Apply inverse transformation to Frame 2 data if available
                if use_transformation and i < len(A_FR2):
                    # Get transformation matrices for this time step
                    A_i = A_FR2[i]  # 2x2 rotation matrix
                    b_i = b_FR2[i]  # 2x1 translation vector
                    
                    # Extract position, velocity, orientation
                    pos_global = frame2_raw[0:2]    # [x, y]
                    vel_global = frame2_raw[2:4]    # [vx, vy]
                    orient_global = frame2_raw[4]   # orientation
                    
                    # Apply inverse transformation
                    pos_hip, vel_hip, orient_hip = self.apply_inverse_transformation(
                        pos_global, vel_global, orient_global, A_i, b_i
                    )
                    
                    # Reconstruct frame 2 data in hip coordinates
                    frame2_data = np.array([pos_hip[0], pos_hip[1], vel_hip[0], vel_hip[1], orient_hip])
                else:
                    frame2_data = frame2_raw
                
                # Check for valid data (no NaN or infinite values)
                if np.any(np.isnan(frame1_data)) or np.any(np.isnan(frame2_data)):
                    continue
                if np.any(np.isinf(frame1_data)) or np.any(np.isinf(frame2_data)):
                    continue
                
                # TP-GMM DATA: [Frame1 (Hip original) | Frame2 (Global->Hip transformed)]
                # Dimension: [5D | 5D] = 10D total
                tp_point = np.concatenate([frame1_data, frame2_data])
                valid_points.append(tp_point)
                
            except Exception as e:
                print(f"Error processing point {i}: {e}")
                continue
        
        if len(valid_points) == 0:
            print("No valid points found in trajectory")
            return None
            
        return np.array(valid_points)
    
    def add_time_dimension(self, demonstrations: List[np.ndarray]) -> List[np.ndarray]:
        """
        Add time dimension to TP-GMM data
        
        Args:
            demonstrations: List of demonstrations without time
            
        Returns:
            List of demonstrations with time [N x 11] = [t | 10D]
        """
        timed_demonstrations = []
        
        for demo in demonstrations:
            n_points = len(demo)
            
            # Normalize time from 0 to 1
            time_values = np.linspace(0, 1, n_points).reshape(-1, 1)
            
            # Add time as first dimension
            demo_with_time = np.hstack([time_values, demo])
            timed_demonstrations.append(demo_with_time)
            
        return timed_demonstrations
    
    def optimize_n_components(self, data: np.ndarray, max_components: int = 20) -> int:
        """
        Optimize number of components using BIC/AIC
        
        Args:
            data: Demonstration data
            max_components: Maximum number of components to test
            
        Returns:
            Optimal number of components
        """
        n_components_range = range(1, min(max_components, len(data)//5) + 1)
        bic_scores = []
        aic_scores = []
        best_bic = np.inf
        best_n_components = 1
        
        print(f"Optimizing number of components (1 to {max(n_components_range)})...")
        
        for n in n_components_range:
            try:
                gmm = GaussianMixture(
                    n_components=n, 
                    covariance_type='full',
                    reg_covar=1e-6,
                    random_state=42,
                    max_iter=100
                )
                gmm.fit(data)
                
                bic = gmm.bic(data)
                aic = gmm.aic(data)
                
                bic_scores.append(bic)
                aic_scores.append(aic)
                
                if bic < best_bic:
                    best_bic = bic
                    best_n_components = n
                    
                print(f"  n={n}: BIC={bic:.1f}, AIC={aic:.1f}")
                    
            except Exception as e:
                print(f"  n={n}: Error - {e}")
                break
        
        print(f"✓ Optimal number of components: {best_n_components} (BIC: {best_bic:.1f})")
        return best_n_components
    
    def train_tpgmm_model(self, demonstrations: List[np.ndarray]) -> Dict:
        """
        Train TP-GMM model
        
        Args:
            demonstrations: List of processed demonstrations
            
        Returns:
            Dictionary with trained model and metadata
        """
        print(f"\n=== Training TP-GMM ===")
        print(f"Demonstrations: {len(demonstrations)}")
        
        # Add time dimension
        timed_demos = self.add_time_dimension(demonstrations)
        
        # Combine all demonstrations
        all_data = np.vstack(timed_demos)
        print(f"Total points: {len(all_data)}")
        print(f"Data dimension: {all_data.shape[1]} (expected: 11 = 1 time + 10 TP-GMM)")
        
        # Optimize number of components
        n_components = self.optimize_n_components(all_data, max_components=20)
        
        # Train final model
        print(f"\nTraining GMM with {n_components} components...")
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            reg_covar=1e-6,
            random_state=42,
            max_iter=300,
            init_params='kmeans'
        )
        
        gmm.fit(all_data)
        
        # Calculate metrics
        log_likelihood = gmm.score(all_data)
        bic = gmm.bic(all_data)
        aic = gmm.aic(all_data)
        
        print(f"✓ Model trained:")
        print(f"  Log-likelihood: {log_likelihood:.2f}")
        print(f"  BIC: {bic:.1f}")
        print(f"  AIC: {aic:.1f}")
        print(f"  Components: {n_components}")
        
        # Structure result
        model_data = {
            'gmm_model': gmm,
            'training_data': all_data,
            'individual_demos': timed_demos,
            'n_components': n_components,
            'metrics': {
                'log_likelihood': log_likelihood,
                'bic': bic,
                'aic': aic
            },
            'data_structure': {
                'total_dim': self.total_dim + 1,  # +1 for time
                'time_dim': 0,
                'frame1_dims': list(range(1, self.point_dim + 1)),      # Hip frame (FR1)
                'frame2_dims': list(range(self.point_dim + 1, self.total_dim + 1)),  # Global frame (FR2)
                'position_dims': {
                    'frame1_hip': [1, 2],      # x, y in hip frame (original)
                    'frame2_hip': [6, 7]       # x, y in hip frame (transformed from global)
                },
                'velocity_dims': {
                    'frame1_hip': [3, 4],      # vx, vy in hip frame (original)
                    'frame2_hip': [8, 9]       # vx, vy in hip frame (transformed from global)
                },
                'orientation_dims': {
                    'frame1_hip': [5],         # orientation in hip frame (original)
                    'frame2_hip': [10]         # orientation in hip frame (transformed from global)
                }
            },
            'frame_info': {
                'reference_frame_id': self.reference_frame_id,  # FR2 (global)
                'target_frame_id': self.target_frame_id,        # FR1 (hip)
                'num_frames': self.num_frames,
                'frame_description': {
                    'FR1': 'Hip frame of reference (target)',
                    'FR2': 'Global frame of reference (reference)'
                }
            }
        }
        
        return model_data
    
    def save_tpgmm_model(self, model_data: Dict, filename: str):
        """
        Save TP-GMM model
        
        Args:
            model_data: Model data
            filename: Filename to save
        """
        try:
            joblib.dump(model_data, filename)
            print(f"✓ TP-GMM model saved to: {filename}")
            
            # Save readable information
            info_file = filename.replace('.pkl', '_info.txt')
            with open(info_file, 'w') as f:
                f.write("=== TP-GMM Gait Model Information ===\n\n")
                f.write(f"GMM Components: {model_data['n_components']}\n")
                f.write(f"Total Dimension: {model_data['data_structure']['total_dim']}\n")
                f.write(f"Frames: {model_data['frame_info']['num_frames']}\n")
                f.write(f"Reference Frame (FR2): Global - ID {model_data['frame_info']['reference_frame_id']}\n")
                f.write(f"Target Frame (FR1): Hip - ID {model_data['frame_info']['target_frame_id']}\n\n")
                f.write("Frame Descriptions:\n")
                for frame, desc in model_data['frame_info']['frame_description'].items():
                    f.write(f"  {frame}: {desc}\n")
                f.write("Data Structure (2D) - Both frames in hip coordinates:\n")
                f.write(f"  Position dims - Hip frame (original): {model_data['data_structure']['position_dims']['frame1_hip']}\n")
                f.write(f"  Position dims - Hip frame (transformed): {model_data['data_structure']['position_dims']['frame2_hip']}\n")
                f.write(f"  Velocity dims - Hip frame (original): {model_data['data_structure']['velocity_dims']['frame1_hip']}\n")
                f.write(f"  Velocity dims - Hip frame (transformed): {model_data['data_structure']['velocity_dims']['frame2_hip']}\n")
                f.write(f"  Orientation dims - Hip frame (original): {model_data['data_structure']['orientation_dims']['frame1_hip']}\n")
                f.write(f"  Orientation dims - Hip frame (transformed): {model_data['data_structure']['orientation_dims']['frame2_hip']}\n\n")
                f.write("Metrics:\n")
                for metric, value in model_data['metrics'].items():
                    f.write(f"  {metric}: {value:.2f}\n")
                f.write(f"\nDemonstrations: {len(model_data['individual_demos'])}\n")
                f.write(f"Total points: {len(model_data['training_data'])}\n")
            
            print(f"✓ Information saved to: {info_file}")
            
        except Exception as e:
            print(f"✗ Error saving model: {e}")
    
    def load_tpgmm_model(self, filename: str) -> Dict:
        """
        Load TP-GMM model
        
        Args:
            filename: Filename to load
            
        Returns:
            Model data
        """
        try:
            model_data = joblib.load(filename)
            print(f"✓ TP-GMM model loaded from: {filename}")
            return model_data
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return None
    
    def visualize_training_data(self, model_data: Dict):
        """
        Visualize training data
        
        Args:
            model_data: Model data
        """
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('TP-GMM Gait Training Data (Both frames in Hip coordinates)', fontsize=16)
            
            data = model_data['training_data']
            pos_dims = model_data['data_structure']['position_dims']
            vel_dims = model_data['data_structure']['velocity_dims']
            orient_dims = model_data['data_structure']['orientation_dims']
            
            # Frame 1 data (Hip frame - original)
            axes[0, 0].plot(data[:, 0], data[:, pos_dims['frame1_hip'][0]], 'b-', alpha=0.7, label='X')
            axes[0, 0].plot(data[:, 0], data[:, pos_dims['frame1_hip'][1]], 'r-', alpha=0.7, label='Y')
            axes[0, 0].set_title('Frame 1 (Hip Original) - Position')
            axes[0, 0].set_xlabel('Normalized Time')
            axes[0, 0].set_ylabel('Position (m)')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            axes[0, 1].plot(data[:, 0], data[:, vel_dims['frame1_hip'][0]], 'b-', alpha=0.7, label='Vx')
            axes[0, 1].plot(data[:, 0], data[:, vel_dims['frame1_hip'][1]], 'r-', alpha=0.7, label='Vy')
            axes[0, 1].set_title('Frame 1 (Hip Original) - Velocity')
            axes[0, 1].set_xlabel('Normalized Time')
            axes[0, 1].set_ylabel('Velocity (m/s)')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            axes[0, 2].plot(data[:, 0], data[:, orient_dims['frame1_hip'][0]], 'g-', alpha=0.7)
            axes[0, 2].set_title('Frame 1 (Hip Original) - Orientation')
            axes[0, 2].set_xlabel('Normalized Time')
            axes[0, 2].set_ylabel('Orientation (rad)')
            axes[0, 2].grid(True)
            
            # Frame 2 data (Global->Hip transformed)
            axes[1, 0].plot(data[:, 0], data[:, pos_dims['frame2_hip'][0]], 'b-', alpha=0.7, label='X')
            axes[1, 0].plot(data[:, 0], data[:, pos_dims['frame2_hip'][1]], 'r-', alpha=0.7, label='Y')
            axes[1, 0].set_title('Frame 2 (Global→Hip Transformed) - Position')
            axes[1, 0].set_xlabel('Normalized Time')
            axes[1, 0].set_ylabel('Position (m)')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            axes[1, 1].plot(data[:, 0], data[:, vel_dims['frame2_hip'][0]], 'b-', alpha=0.7, label='Vx')
            axes[1, 1].plot(data[:, 0], data[:, vel_dims['frame2_hip'][1]], 'r-', alpha=0.7, label='Vy')
            axes[1, 1].set_title('Frame 2 (Global→Hip Transformed) - Velocity')
            axes[1, 1].set_xlabel('Normalized Time')
            axes[1, 1].set_ylabel('Velocity (m/s)')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            axes[1, 2].plot(data[:, 0], data[:, orient_dims['frame2_hip'][0]], 'g-', alpha=0.7)
            axes[1, 2].set_title('Frame 2 (Global→Hip Transformed) - Orientation')
            axes[1, 2].set_xlabel('Normalized Time')
            axes[1, 2].set_ylabel('Orientation (rad)')
            axes[1, 2].grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error in visualization: {e}")
    
    def visualize_2d_trajectories(self, model_data: Dict):
        """
        Visualize 2D trajectories for both frames
        
        Args:
            model_data: Model data
        """
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle('2D Gait Trajectories (Both frames in Hip coordinates)', fontsize=16)
            
            data = model_data['training_data']
            pos_dims = model_data['data_structure']['position_dims']
            
            # Frame 1 trajectory (Hip frame - original)
            axes[0].plot(data[:, pos_dims['frame1_hip'][0]], data[:, pos_dims['frame1_hip'][1]], 'b-', alpha=0.7)
            axes[0].scatter(data[0, pos_dims['frame1_hip'][0]], data[0, pos_dims['frame1_hip'][1]], 
                          c='green', s=50, label='Start')
            axes[0].scatter(data[-1, pos_dims['frame1_hip'][0]], data[-1, pos_dims['frame1_hip'][1]], 
                          c='red', s=50, label='End')
            axes[0].set_title('Frame 1 (Hip Original) - 2D Trajectory')
            axes[0].set_xlabel('X Position (m)')
            axes[0].set_ylabel('Y Position (m)')
            axes[0].legend()
            axes[0].grid(True)
            axes[0].axis('equal')
            
            # Frame 2 trajectory (Global->Hip transformed)
            axes[1].plot(data[:, pos_dims['frame2_hip'][0]], data[:, pos_dims['frame2_hip'][1]], 'r-', alpha=0.7)
            axes[1].scatter(data[0, pos_dims['frame2_hip'][0]], data[0, pos_dims['frame2_hip'][1]], 
                          c='green', s=50, label='Start')
            axes[1].scatter(data[-1, pos_dims['frame2_hip'][0]], data[-1, pos_dims['frame2_hip'][1]], 
                          c='red', s=50, label='End')
            axes[1].set_title('Frame 2 (Global→Hip Transformed) - 2D Trajectory')
            axes[1].set_xlabel('X Position (m)')
            axes[1].set_ylabel('Y Position (m)')
            axes[1].legend()
            axes[1].grid(True)
            axes[1].axis('equal')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error in 2D visualization: {e}")
    
    def analyze_latent_space_pca(self, model_data: Dict) -> Dict:
        """
        Analyze the latent space using PCA with different numbers of components
        
        Args:
            model_data: Model data containing training data
            
        Returns:
            Dictionary with PCA analysis results
        """
        print("\n=== PCA Latent Space Analysis ===")
        
        # Get training data (excluding time dimension)
        data = model_data['training_data'][:, 1:]  # Remove time column
        
        print(f"Data shape for PCA: {data.shape}")
        print(f"Original dimensions: {data.shape[1]} (10D TP-GMM data)")
        
        # Different numbers of PCA components to analyze
        n_components_list = [2, 3, 5, 8]
        pca_results = {}
        
        # Standardize the data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        for n_comp in n_components_list:
            if n_comp <= data.shape[1]:  # Can't have more components than features
                print(f"\nAnalyzing PCA with {n_comp} components...")
                
                # Fit PCA
                pca = PCA(n_components=n_comp, random_state=42)
                data_pca = pca.fit_transform(data_scaled)
                
                # Calculate explained variance
                explained_var_ratio = pca.explained_variance_ratio_
                cumulative_var = np.cumsum(explained_var_ratio)
                
                print(f"  Explained variance ratio: {explained_var_ratio}")
                print(f"  Cumulative explained variance: {cumulative_var[-1]:.3f}")
                
                # Store results
                pca_results[f'{n_comp}D'] = {
                    'pca_model': pca,
                    'transformed_data': data_pca,
                    'explained_variance_ratio': explained_var_ratio,
                    'cumulative_variance': cumulative_var,
                    'components': pca.components_,
                    'scaler': scaler
                }
        
        # Store in model data
        model_data['pca_analysis'] = pca_results
        
        print(f"\n✓ PCA analysis completed for {len(pca_results)} different component counts")
        return pca_results
    
    def visualize_pca_latent_space(self, model_data: Dict):
        """
        Visualize PCA latent space with 4 different component variations
        
        Args:
            model_data: Model data with PCA analysis
        """
        if 'pca_analysis' not in model_data:
            print("No PCA analysis found. Running PCA analysis first...")
            self.analyze_latent_space_pca(model_data)
        
        pca_results = model_data['pca_analysis']
        
        try:
            # Create subplots for different PCA variations
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('PCA Latent Space Analysis - Different Component Variations', fontsize=16)
            
            # Time vector for coloring
            n_points = len(model_data['training_data'])
            time_colors = np.linspace(0, 1, n_points)
            
            plot_configs = [
                ('2D', 0, 0, 'scatter'),
                ('3D', 0, 1, 'scatter_3d'),
                ('5D', 1, 0, 'components'),
                ('8D', 1, 1, 'components')
            ]
            
            for pca_key, row, col, plot_type in plot_configs:
                if pca_key in pca_results:
                    ax = axes[row, col]
                    pca_data = pca_results[pca_key]
                    transformed_data = pca_data['transformed_data']
                    explained_var = pca_data['explained_variance_ratio']
                    cumulative_var = pca_data['cumulative_variance']
                    
                    if plot_type == 'scatter' and transformed_data.shape[1] >= 2:
                        # 2D scatter plot
                        scatter = ax.scatter(transformed_data[:, 0], transformed_data[:, 1], 
                                           c=time_colors, cmap='viridis', alpha=0.6, s=20)
                        ax.set_xlabel(f'PC1 ({explained_var[0]:.1%} var)')
                        ax.set_ylabel(f'PC2 ({explained_var[1]:.1%} var)')
                        ax.set_title(f'{pca_key} PCA Space\nCumulative: {cumulative_var[-1]:.1%}')
                        ax.grid(True, alpha=0.3)
                        plt.colorbar(scatter, ax=ax, label='Time progression')
                    
                    elif plot_type == 'scatter_3d' and transformed_data.shape[1] >= 3:
                        # 3D scatter plot (projected to 2D)
                        ax.scatter(transformed_data[:, 0], transformed_data[:, 1], 
                                 c=time_colors, cmap='plasma', alpha=0.6, s=20)
                        ax.set_xlabel(f'PC1 ({explained_var[0]:.1%})')
                        ax.set_ylabel(f'PC2 ({explained_var[1]:.1%})')
                        ax.set_title(f'{pca_key} PCA Space (PC1 vs PC2)\nPC3: {explained_var[2]:.1%}, Cumulative: {cumulative_var[-1]:.1%}')
                        ax.grid(True, alpha=0.3)
                    
                    elif plot_type == 'components':
                        # Plot explained variance components
                        bars = ax.bar(range(1, len(explained_var) + 1), explained_var, 
                                     alpha=0.7, color='steelblue')
                        ax.plot(range(1, len(cumulative_var) + 1), cumulative_var, 
                               'ro-', linewidth=2, markersize=4, label='Cumulative')
                        ax.set_xlabel('Principal Component')
                        ax.set_ylabel('Explained Variance Ratio')
                        ax.set_title(f'{pca_key} PCA - Explained Variance\nTotal: {cumulative_var[-1]:.1%}')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        # Add value labels on bars
                        for i, bar in enumerate(bars):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                   f'{height:.1%}', ha='center', va='bottom', fontsize=8)
                else:
                    axes[row, col].text(0.5, 0.5, f'{pca_key} PCA\nNot Available', 
                                       ha='center', va='center', transform=axes[row, col].transAxes,
                                       fontsize=12)
                    axes[row, col].set_title(f'{pca_key} PCA - Not Available')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error in PCA visualization: {e}")
    
    def visualize_pca_component_analysis(self, model_data: Dict):
        """
        Detailed analysis of PCA components and their contribution to original features
        
        Args:
            model_data: Model data with PCA analysis
        """
        if 'pca_analysis' not in model_data:
            print("No PCA analysis found. Running PCA analysis first...")
            self.analyze_latent_space_pca(model_data)
        
        pca_results = model_data['pca_analysis']
        
        try:
            # Create comprehensive component analysis
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('PCA Component Analysis - Feature Contributions', fontsize=16)
            
            # Feature names for better interpretation
            feature_names = [
                'F1_X', 'F1_Y', 'F1_Vx', 'F1_Vy', 'F1_θ',  # Frame 1 (Hip original)
                'F2_X', 'F2_Y', 'F2_Vx', 'F2_Vy', 'F2_θ'   # Frame 2 (Global→Hip transformed)
            ]
            
            pca_keys = ['2D', '3D', '5D', '8D']
            
            for idx, pca_key in enumerate(pca_keys):
                if pca_key in pca_results:
                    row, col = idx // 2, idx % 2
                    ax = axes[row, col]
                    
                    components = pca_results[pca_key]['components']
                    explained_var = pca_results[pca_key]['explained_variance_ratio']
                    
                    # Create heatmap of component loadings
                    im = ax.imshow(components, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
                    
                    # Set labels
                    ax.set_xticks(range(len(feature_names)))
                    ax.set_xticklabels(feature_names, rotation=45, ha='right')
                    ax.set_yticks(range(len(explained_var)))
                    ax.set_yticklabels([f'PC{i+1}\n({explained_var[i]:.1%})' for i in range(len(explained_var))])
                    
                    ax.set_title(f'{pca_key} PCA Components\nFeature Loadings')
                    
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label('Component Loading')
                    
                    # Add text annotations for significant loadings
                    for i in range(components.shape[0]):
                        for j in range(components.shape[1]):
                            value = components[i, j]
                            if abs(value) > 0.3:  # Only show significant loadings
                                ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                                       color='white' if abs(value) > 0.6 else 'black',
                                       fontsize=8, weight='bold')
                else:
                    row, col = idx // 2, idx % 2
                    axes[row, col].text(0.5, 0.5, f'{pca_key} PCA\nNot Available',
                                       ha='center', va='center', transform=axes[row, col].transAxes)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error in component analysis: {e}")
    
    def visualize_pca_trajectory_projection(self, model_data: Dict):
        """
        Visualize individual trajectory projections in PCA space
        
        Args:
            model_data: Model data with PCA analysis and individual demonstrations
        """
        if 'pca_analysis' not in model_data:
            print("No PCA analysis found. Running PCA analysis first...")
            self.analyze_latent_space_pca(model_data)
        
        pca_results = model_data['pca_analysis']
        individual_demos = model_data['individual_demos']
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Individual Trajectory Projections in PCA Space', fontsize=16)
            
            # Use 2D PCA for trajectory visualization
            if '2D' in pca_results:
                pca_model = pca_results['2D']['pca_model']
                scaler = pca_results['2D']['scaler']
                explained_var = pca_results['2D']['explained_variance_ratio']
                
                # Different visualization approaches
                plot_configs = [
                    (0, 0, 'all_trajectories', 'All Trajectories'),
                    (0, 1, 'first_few', 'First 5 Trajectories'),
                    (1, 0, 'trajectory_evolution', 'Trajectory Evolution'),
                    (1, 1, 'density_plot', 'Density Distribution')
                ]
                
                for row, col, plot_type, title in plot_configs:
                    ax = axes[row, col]
                    
                    if plot_type == 'all_trajectories':
                        # Plot all trajectories in PCA space
                        colors = plt.cm.tab10(np.linspace(0, 1, len(individual_demos)))
                        for i, demo in enumerate(individual_demos[:10]):  # Limit to first 10
                            demo_data = demo[:, 1:]  # Remove time
                            demo_scaled = scaler.transform(demo_data)
                            demo_pca = pca_model.transform(demo_scaled)
                            ax.plot(demo_pca[:, 0], demo_pca[:, 1], alpha=0.6, 
                                   color=colors[i], linewidth=1.5)
                        ax.set_title(f'{title}\n(First 10 demonstrations)')
                    
                    elif plot_type == 'first_few':
                        # Highlight first few trajectories
                        for i, demo in enumerate(individual_demos[:5]):
                            demo_data = demo[:, 1:]
                            demo_scaled = scaler.transform(demo_data)
                            demo_pca = pca_model.transform(demo_scaled)
                            ax.plot(demo_pca[:, 0], demo_pca[:, 1], alpha=0.8,
                                   linewidth=2, label=f'Traj {i+1}')
                            # Mark start and end points
                            ax.scatter(demo_pca[0, 0], demo_pca[0, 1], 
                                     s=50, marker='o', alpha=0.8)
                            ax.scatter(demo_pca[-1, 0], demo_pca[-1, 1], 
                                     s=50, marker='s', alpha=0.8)
                        ax.legend(fontsize=8)
                        ax.set_title(title)
                    
                    elif plot_type == 'trajectory_evolution':
                        # Show trajectory evolution with time coloring
                        demo = individual_demos[0]  # Use first demonstration
                        demo_data = demo[:, 1:]
                        demo_scaled = scaler.transform(demo_data)
                        demo_pca = pca_model.transform(demo_scaled)
                        
                        # Color by time progression
                        time_colors = np.linspace(0, 1, len(demo_pca))
                        scatter = ax.scatter(demo_pca[:, 0], demo_pca[:, 1], 
                                           c=time_colors, cmap='viridis', s=30, alpha=0.7)
                        ax.plot(demo_pca[:, 0], demo_pca[:, 1], alpha=0.3, color='gray')
                        plt.colorbar(scatter, ax=ax, label='Time progression')
                        ax.set_title(f'{title}\n(First demonstration)')
                    
                    elif plot_type == 'density_plot':
                        # 2D histogram/density plot
                        all_pca_data = pca_results['2D']['transformed_data']
                        hist, xedges, yedges = np.histogram2d(all_pca_data[:, 0], all_pca_data[:, 1], 
                                                            bins=30, density=True)
                        im = ax.imshow(hist.T, origin='lower', extent=[xedges[0], xedges[-1], 
                                                                     yedges[0], yedges[-1]],
                                     cmap='Blues', alpha=0.8)
                        plt.colorbar(im, ax=ax, label='Density')
                        ax.set_title(title)
                    
                    ax.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
                    ax.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error in trajectory projection visualization: {e}")

def main():
    """
    Example usage of the TP-GMM Gait Trainer
    """
    # Initialize trainer with updated frame assignments
    # FR2 (reference_frame_id=2) = Global frame
    # FR1 (target_frame_id=1) = Hip frame
    trainer = TPGMMGaitTrainer(reference_frame_id=2, target_frame_id=1)
    
    # Path to your updated .mat file
    mat_file_path = 'new_processed_gait_data.mat'  # Replace with your actual file path
    
    try:
        # 1. Load demonstrations from MAT file
        print("=== Loading demonstrations from MAT file ===")
        demonstrations = trainer.load_demonstrations_from_mat(mat_file_path)
        
        if len(demonstrations) == 0:
            print("✗ No valid demonstrations found!")
            return
        
        # 2. Train TP-GMM model
        print("\n=== Training TP-GMM model ===")
        model_data = trainer.train_tpgmm_model(demonstrations)
        
        # 3. Save model
        print("\n=== Saving model ===")
        trainer.save_tpgmm_model(model_data, 'tpgmm_gait_model_updated.pkl')
        
        # 4. Visualize data (optional)
        print("\n=== Visualizing data ===")
        trainer.visualize_training_data(model_data)
        trainer.visualize_2d_trajectories(model_data)
        
        # 5. PCA Latent Space Analysis
        print("\n=== PCA Latent Space Analysis ===")
        trainer.analyze_latent_space_pca(model_data)
        trainer.visualize_pca_latent_space(model_data)
        trainer.visualize_pca_component_analysis(model_data)
        trainer.visualize_pca_trajectory_projection(model_data)
        
        print("\n✓ TP-GMM gait model trained successfully!")
        print(f"  Demonstrations: {len(demonstrations)}")
        print(f"  Components: {model_data['n_components']}")
        print(f"  Dimension: {model_data['data_structure']['total_dim']}")
        print(f"  Frame setup: FR2 (Global) -> FR1 (Hip)")
        
    except Exception as e:
        print(f"✗ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()