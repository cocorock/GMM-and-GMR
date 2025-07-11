import numpy as np
import scipy.io
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

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
        
    def load_gait_data(self, mat_file_path: str) -> Tuple[Optional[List[np.ndarray]], 
                                                         Optional[List[np.ndarray]], 
                                                         Optional[List[np.ndarray]], 
                                                         Optional[List[dict]]]:
        """
        Loads and preprocesses the complete gait data from the specified .mat file for TP-GMM.
        
        Args:
            mat_file_path: The path to the .mat file.

        Returns:
            tuple: (trajectories_fr1, trajectories_fr2, time_data, frame_transforms)
        """
        try:
            print(f"Loading MATLAB file: {mat_file_path}")
            mat_data = scipy.io.loadmat(mat_file_path)
            
            if 'processed_gait_data' not in mat_data:
                print("Error: 'processed_gait_data' not found in .mat file")
                print("Available keys:", list(mat_data.keys()))
                return None, None, None, None
            
            processed_gait_data = mat_data['processed_gait_data']
            print(f"Data shape: {processed_gait_data.shape}")
            print(f"Number of trials: {processed_gait_data.shape[1]}")
            
            trajectories_fr1 = []
            trajectories_fr2 = []
            time_data = []
            frame_transforms = []
            
            # Process each trial
            for i in range(processed_gait_data.shape[1]):
                try:
                    # Extract trial data - handle nested structure
                    trial_data = processed_gait_data[0, i]
                    if hasattr(trial_data, 'item'):
                        trial_data = trial_data.item()
                    
                    # Check if it's a structured array
                    if hasattr(trial_data, 'dtype') and trial_data.dtype.names:
                        # Structured array - extract fields by name
                        time = self._extract_field(trial_data, 'time')
                        
                        # Frame 1 data (Hip's frame of reference)
                        ankle_pos_fr1 = self._extract_field(trial_data, 'ankle_pos_FR1')
                        ankle_vel_fr1 = self._extract_field(trial_data, 'ankle_pos_FR1_velocity')
                        ankle_orient_fr1 = self._extract_field(trial_data, 'ankle_orientation_FR1')
                        
                        # Frame 2 data (Global frame of reference)
                        ankle_pos_fr2 = self._extract_field(trial_data, 'ankle_pos_FR2')
                        ankle_vel_fr2 = self._extract_field(trial_data, 'ankle_pos_FR2_velocity')
                        ankle_orient_fr2 = self._extract_field(trial_data, 'ankle_orientation_FR2')
                        
                        # Transformation matrices
                        ankle_A_fr1 = self._extract_field(trial_data, 'ankle_A_FR1')
                        ankle_b_fr1 = self._extract_field(trial_data, 'ankle_b_FR1')
                        ankle_A_fr2 = self._extract_field(trial_data, 'ankle_A_FR2')
                        ankle_b_fr2 = self._extract_field(trial_data, 'ankle_b_FR2')
                        
                    else:
                        # Tuple/array structure - use positional indexing
                        print(f"Warning: Trial {i} using positional indexing (less reliable)")
                        time = trial_data[0].flatten()
                        ankle_pos_fr1 = trial_data[3]
                        ankle_vel_fr1 = trial_data[2]
                        ankle_orient_fr1 = trial_data[6].flatten()
                        ankle_pos_fr2 = trial_data[5]
                        ankle_vel_fr2 = trial_data[4]
                        ankle_orient_fr2 = trial_data[7].flatten()
                        ankle_A_fr1 = trial_data[8]
                        ankle_b_fr1 = trial_data[9].astype(np.float64)
                        ankle_A_fr2 = trial_data[10]
                        ankle_b_fr2 = trial_data[11]
                    
                    # Validate data shapes
                    if not self._validate_data_shapes(time, ankle_pos_fr1, ankle_pos_fr2, 
                                                    ankle_vel_fr1, ankle_vel_fr2, 
                                                    ankle_orient_fr1, ankle_orient_fr2,
                                                    ankle_A_fr1, ankle_b_fr1, 
                                                    ankle_A_fr2, ankle_b_fr2, i):
                        continue
                    
                    # Store transformation data
                    transform_data = {
                        'A_FR1': ankle_A_fr1,
                        'b_FR1': ankle_b_fr1,
                        'A_FR2': ankle_A_fr2,
                        'b_FR2': ankle_b_fr2
                    }
                    frame_transforms.append(transform_data)
                    
                    # Create trajectories
                    trajectory_fr1 = np.column_stack([
                        time,
                        ankle_pos_fr1[:, 0], ankle_pos_fr1[:, 1],
                        ankle_vel_fr1[:, 0], ankle_vel_fr1[:, 1],
                        ankle_orient_fr1
                    ])
                    
                    trajectory_fr2 = np.column_stack([
                        time,
                        ankle_pos_fr2[:, 0], ankle_pos_fr2[:, 1],
                        ankle_vel_fr2[:, 0], ankle_vel_fr2[:, 1],
                        ankle_orient_fr2
                    ])
                    
                    trajectories_fr1.append(trajectory_fr1)
                    trajectories_fr2.append(trajectory_fr2)
                    time_data.append(time)
                    
                except Exception as e:
                    print(f"Error processing trial {i}: {e}")
                    continue
            
            print(f"Successfully loaded {len(trajectories_fr1)} trajectories")
            if len(trajectories_fr1) > 0:
                print(f"Each trajectory has {trajectories_fr1[0].shape[1]} dimensions")
                print(f"Trajectory length: {trajectories_fr1[0].shape[0]} time steps")
            
            return trajectories_fr1, trajectories_fr2, time_data, frame_transforms
            
        except Exception as e:
            print(f"Error loading file {mat_file_path}: {e}")
            return None, None, None, None
    
    def _extract_field(self, trial_data, field_name: str) -> np.ndarray:
        """Extract and clean field data from structured array"""
        try:
            field_data = trial_data[field_name]
            
            # Handle nested arrays
            if hasattr(field_data, 'item') and field_data.shape == (1, 1):
                field_data = field_data.item()
            
            # Ensure it's a numpy array
            field_data = np.asarray(field_data, dtype=np.float64)
            
            # Flatten if needed for 1D data
            if field_name in ['time', 'ankle_orientation_FR1', 'ankle_orientation_FR2']:
                field_data = field_data.flatten()
            
            return field_data
            
        except Exception as e:
            raise ValueError(f"Could not extract field '{field_name}': {e}")
    
    def _validate_data_shapes(self, time, pos_fr1, pos_fr2, vel_fr1, vel_fr2, 
                            orient_fr1, orient_fr2, A_fr1, b_fr1, A_fr2, b_fr2, trial_idx) -> bool:
        """Validate that all data arrays have consistent shapes"""
        try:
            n_points = len(time)
            
            # Check position data
            if pos_fr1.shape != (n_points, 2) or pos_fr2.shape != (n_points, 2):
                print(f"Trial {trial_idx}: Invalid position shapes - FR1: {pos_fr1.shape}, FR2: {pos_fr2.shape}")
                return False
            
            # Check velocity data
            if vel_fr1.shape != (n_points, 2) or vel_fr2.shape != (n_points, 2):
                print(f"Trial {trial_idx}: Invalid velocity shapes - FR1: {vel_fr1.shape}, FR2: {vel_fr2.shape}")
                return False
            
            # Check orientation data
            if len(orient_fr1) != n_points or len(orient_fr2) != n_points:
                print(f"Trial {trial_idx}: Invalid orientation lengths - FR1: {len(orient_fr1)}, FR2: {len(orient_fr2)}")
                return False
            
            # Check transformation matrices
            if A_fr1.shape != (n_points, 2, 2) or A_fr2.shape != (n_points, 2, 2):
                print(f"Trial {trial_idx}: Invalid A matrix shapes - FR1: {A_fr1.shape}, FR2: {A_fr2.shape}")
                return False
            
            if b_fr1.shape != (n_points, 2) or b_fr2.shape != (n_points, 2):
                print(f"Trial {trial_idx}: Invalid b vector shapes - FR1: {b_fr1.shape}, FR2: {b_fr2.shape}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Trial {trial_idx}: Shape validation error: {e}")
            return False
    
    def apply_inverse_transformation(self, position: np.ndarray, velocity: np.ndarray, 
                                   orientation: float, A_matrix: np.ndarray, 
                                   b_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
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
            # Ensure proper shapes
            position = np.asarray(position).reshape(-1)
            velocity = np.asarray(velocity).reshape(-1)
            b_vector = np.asarray(b_vector).reshape(-1)
            
            # Compute inverse transformation
            A_inv = np.linalg.inv(A_matrix)
            
            # Transform position: p_hip = A_inv * (p_global - b)
            pos_transformed = A_inv @ (position - b_vector)
            
            # Transform velocity: v_hip = A_inv * v_global
            vel_transformed = A_inv @ velocity
            
            # Transform orientation: theta_hip = theta_global - rotation_angle
            rotation_angle = np.arctan2(A_matrix[1, 0], A_matrix[0, 0])
            orientation_transformed = orientation - rotation_angle
            
            # Normalize orientation to [-pi, pi]
            orientation_transformed = np.arctan2(np.sin(orientation_transformed), 
                                               np.cos(orientation_transformed))
            
            return pos_transformed, vel_transformed, orientation_transformed
            
        except np.linalg.LinAlgError:
            warnings.warn("Transformation matrix not invertible, using pseudo-inverse")
            A_pinv = np.linalg.pinv(A_matrix)
            pos_transformed = A_pinv @ (position - b_vector)
            vel_transformed = A_pinv @ velocity
            orientation_transformed = orientation
            return pos_transformed, vel_transformed, orientation_transformed
        except Exception as e:
            warnings.warn(f"Transformation failed: {e}, using original data")
            return position, velocity, orientation
    
    def process_gait_trajectory(self, traj_fr1: np.ndarray, traj_fr2: np.ndarray, 
                              transform_data: dict) -> Optional[np.ndarray]:
        """
        Process gait trajectory for TP-GMM format
        
        CORRECTED APPROACH: Both FR1 and FR2 contain transformed data and need inverse 
        transformations to get back to the true ankle positions.
        
        Args:
            traj_fr1: Trajectory in frame 1 (transformed ankle data) [time, x, y, vx, vy, orientation]
            traj_fr2: Trajectory in frame 2 (transformed ankle data) [time, x, y, vx, vy, orientation]
            transform_data: Dictionary with transformation matrices for both frames
            
        Returns:
            Array with TP-GMM data [N x 10] = [Frame1_true | Frame2_true]
        """
        valid_points = []
        
        # Ensure both trajectories have same length
        min_len = min(len(traj_fr1), len(traj_fr2))
        if min_len == 0:
            print("Empty trajectories")
            return None
        
        # Extract transformation matrices for BOTH frames
        A_FR1 = transform_data['A_FR1']  # [N x 2 x 2] - FR1 transformations
        b_FR1 = transform_data['b_FR1']  # [N x 2] - FR1 translations
        A_FR2 = transform_data['A_FR2']  # [N x 2 x 2] - FR2 transformations  
        b_FR2 = transform_data['b_FR2']  # [N x 2] - FR2 translations
        
        print(f"Processing {min_len} time points with inverse transformations for BOTH frames...")
        
        for i in range(min_len):
            try:
                # FRAME 1: Apply inverse transformation to get true ankle position
                frame1_raw = traj_fr1[i, 1:6]  # Extract [x, y, vx, vy, orientation]
                
                # Extract FR1 components
                pos_fr1 = frame1_raw[0:2]    # [x, y]
                vel_fr1 = frame1_raw[2:4]    # [vx, vy]
                orient_fr1 = frame1_raw[4]   # orientation
                
                # Get FR1 transformation matrices for this time step
                A1_i = A_FR1[i]  # 2x2 transformation matrix
                b1_i = b_FR1[i]  # 2x1 translation vector
                
                # Apply inverse transformation to get true ankle position from FR1
                pos_true_fr1, vel_true_fr1, orient_true_fr1 = self.apply_inverse_transformation(
                    pos_fr1, vel_fr1, orient_fr1, A1_i, b1_i
                )
                
                # FRAME 2: Apply inverse transformation to get true ankle position
                frame2_raw = traj_fr2[i, 1:6]  # Extract [x, y, vx, vy, orientation]
                
                # Extract FR2 components
                pos_fr2 = frame2_raw[0:2]    # [x, y]
                vel_fr2 = frame2_raw[2:4]    # [vx, vy]
                orient_fr2 = frame2_raw[4]   # orientation
                
                # Get FR2 transformation matrices for this time step
                A2_i = A_FR2[i]  # 2x2 transformation matrix
                b2_i = b_FR2[i]  # 2x1 translation vector
                
                # Apply inverse transformation to get true ankle position from FR2
                pos_true_fr2, vel_true_fr2, orient_true_fr2 = self.apply_inverse_transformation(
                    pos_fr2, vel_fr2, orient_fr2, A2_i, b2_i
                )
                
                # Reconstruct both frames with true ankle positions
                frame1_data = np.array([pos_true_fr1[0], pos_true_fr1[1], 
                                      vel_true_fr1[0], vel_true_fr1[1], orient_true_fr1])
                frame2_data = np.array([pos_true_fr2[0], pos_true_fr2[1], 
                                      vel_true_fr2[0], vel_true_fr2[1], orient_true_fr2])
                
                # Validate data
                if np.any(np.isnan(frame1_data)) or np.any(np.isnan(frame2_data)):
                    continue
                if np.any(np.isinf(frame1_data)) or np.any(np.isinf(frame2_data)):
                    continue
                
                # Create TP-GMM point: [Frame1_true | Frame2_true]
                # Both frames now represent the same thing: true ankle positions
                # They should be coincident (or very close)
                tp_point = np.concatenate([frame1_data, frame2_data])
                valid_points.append(tp_point)
                
            except Exception as e:
                print(f"Error processing point {i}: {e}")
                continue
        
        if len(valid_points) == 0:
            print("No valid points found in trajectory")
            return None
        
        result = np.array(valid_points)
        print(f"Successfully processed {len(result)} points")
        
        # Check if trajectories are now coincident
        if len(result) > 0:
            frame1_positions = result[:, 0:2]  # First frame positions
            frame2_positions = result[:, 5:7]  # Second frame positions  
            mean_diff = np.mean(np.abs(frame1_positions - frame2_positions))
            print(f"Mean position difference between frames: {mean_diff:.6f}")
            if mean_diff < 0.001:
                print("✓ Trajectories are coincident - transformations successful!")
            else:
                print(f"⚠ Trajectories not fully coincident (diff: {mean_diff:.6f})")
        
        return result
    
    def load_demonstrations_from_mat(self, mat_file_path: str) -> List[np.ndarray]:
        """Load and process demonstrations from MAT file"""
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
            
            # Process trajectory for TP-GMM
            demo_array = self.process_gait_trajectory(traj_fr1, traj_fr2, transform_data)
            
            if demo_array is not None and len(demo_array) > 0:
                demonstrations.append(demo_array)
                print(f"✓ Trajectory {i+1} processed: {len(demo_array)} points")
            else:
                print(f"✗ Trajectory {i+1} failed or empty")
        
        print(f"Successfully processed {len(demonstrations)} demonstrations")
        return demonstrations
    
    def add_time_dimension(self, demonstrations: List[np.ndarray]) -> List[np.ndarray]:
        """Add normalized time dimension to demonstrations"""
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
        """Optimize number of GMM components using BIC"""
        max_comp = min(max_components, len(data)//10, 50)  # Reasonable limits
        n_components_range = range(1, max_comp + 1)
        
        best_bic = np.inf
        best_n_components = 1
        
        print(f"Optimizing GMM components (testing 1 to {max_comp})...")
        
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
                
                if bic < best_bic:
                    best_bic = bic
                    best_n_components = n
                
                if n <= 10 or n % 5 == 0:  # Reduce output
                    print(f"  n={n}: BIC={bic:.1f}")
                    
            except Exception as e:
                print(f"  n={n}: Error - {e}")
                break
        
        print(f"✓ Optimal components: {best_n_components} (BIC: {best_bic:.1f})")
        return best_n_components
    
    def train_tpgmm_model(self, demonstrations: List[np.ndarray]) -> Dict:
        """Train TP-GMM model"""
        print(f"\n=== Training TP-GMM ===")
        print(f"Demonstrations: {len(demonstrations)}")
        
        if len(demonstrations) == 0:
            raise ValueError("No demonstrations provided")
        
        # Add time dimension
        timed_demos = self.add_time_dimension(demonstrations)
        
        # Combine all demonstrations
        all_data = np.vstack(timed_demos)
        print(f"Total data points: {len(all_data)}")
        print(f"Data dimension: {all_data.shape[1]} (1 time + 10 TP-GMM)")
        
        # Check data statistics
        print(f"Data range: [{np.min(all_data):.3f}, {np.max(all_data):.3f}]")
        print(f"Data std: {np.std(all_data):.3f}")
        
        # Optimize number of components
        n_components = self.optimize_n_components(all_data, max_components=25)
        
        # Train final model
        print(f"\nTraining final GMM with {n_components} components...")
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            reg_covar=1e-6,
            random_state=42,
            max_iter=500,
            init_params='kmeans'
        )
        
        gmm.fit(all_data)
        
        # Calculate metrics
        log_likelihood = gmm.score(all_data)
        bic = gmm.bic(all_data)
        aic = gmm.aic(all_data)
        
        print(f"✓ Model training completed:")
        print(f"  Log-likelihood: {log_likelihood:.2f}")
        print(f"  BIC: {bic:.1f}")
        print(f"  AIC: {aic:.1f}")
        print(f"  Components: {n_components}")
        
        # Create model data structure
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
                'frame1_dims': list(range(1, self.point_dim + 1)),      # Hip frame (original)
                'frame2_dims': list(range(self.point_dim + 1, self.total_dim + 1)),  # Hip frame (transformed)
                'position_dims': {
                    'frame1_true': [1, 2],      # x, y true ankle position from FR1
                    'frame2_true': [6, 7]       # x, y true ankle position from FR2
                },
                'velocity_dims': {
                    'frame1_true': [3, 4],      # vx, vy true ankle velocity from FR1
                    'frame2_true': [8, 9]       # vx, vy true ankle velocity from FR2
                },
                'orientation_dims': {
                    'frame1_true': [5],         # true ankle orientation from FR1
                    'frame2_true': [10]         # true ankle orientation from FR2
                }
            },
            'frame_info': {
                'reference_frame_id': self.reference_frame_id,  # FR2 (global)
                'target_frame_id': self.target_frame_id,        # FR1 (hip)
                'num_frames': self.num_frames,
                'transformation_applied': 'Both_FR1_and_FR2_to_true_positions',  # Both frames transformed
                'coordinate_system': 'true_ankle_positions',     # Both frames represent true ankle positions
                'frame_description': {
                    'FR1': 'Hip frame data - inverse transformed to true ankle position',
                    'FR2': 'Global frame data - inverse transformed to true ankle position'
                }
            }
        }
        
        return model_data
    
    def save_tpgmm_model(self, model_data: Dict, filename: str):
        """Save TP-GMM model with comprehensive information"""
        try:
            joblib.dump(model_data, filename)
            print(f"✓ TP-GMM model saved to: {filename}")
            
            # Save detailed information
            info_file = filename.replace('.pkl', '_info.txt')
            with open(info_file, 'w') as f:
                f.write("=== TP-GMM Gait Model Information ===\n\n")
                f.write(f"Model Type: Task-Parameterized Gaussian Mixture Model\n")
                f.write(f"Application: Gait trajectory modeling\n")
                f.write(f"Date Created: {np.datetime64('now')}\n\n")
                
                f.write("=== Model Configuration ===\n")
                f.write(f"GMM Components: {model_data['n_components']}\n")
                f.write(f"Total Dimension: {model_data['data_structure']['total_dim']}\n")
                f.write(f"Coordinate System: {model_data['frame_info']['coordinate_system']}\n")
                f.write(f"Transformation Applied: {model_data['frame_info']['transformation_applied']}\n\n")
                
                f.write("=== Frame Configuration ===\n")
                f.write(f"Number of Frames: {model_data['frame_info']['num_frames']}\n")
                f.write(f"Reference Frame (FR2): Global frame (ID {model_data['frame_info']['reference_frame_id']})\n")
                f.write(f"Target Frame (FR1): Hip frame (ID {model_data['frame_info']['target_frame_id']})\n\n")
                
                f.write("Frame Descriptions:\n")
                for frame, desc in model_data['frame_info']['frame_description'].items():
                    f.write(f"  {frame}: {desc}\n")
                
                f.write("\n=== Data Structure ===\n")
                f.write("Both frames expressed in hip coordinate system:\n")
                f.write(f"  Time dimension: {model_data['data_structure']['time_dim']}\n")
                f.write(f"  Frame 1 dimensions: {model_data['data_structure']['frame1_dims']}\n")
                f.write(f"  Frame 2 dimensions: {model_data['data_structure']['frame2_dims']}\n\n")
                
                f.write("Feature mapping:\n")
                f.write(f"  Position (Frame 1): {model_data['data_structure']['position_dims']['frame1_hip']}\n")
                f.write(f"  Position (Frame 2): {model_data['data_structure']['position_dims']['frame2_hip']}\n")
                f.write(f"  Velocity (Frame 1): {model_data['data_structure']['velocity_dims']['frame1_hip']}\n")
                f.write(f"  Velocity (Frame 2): {model_data['data_structure']['velocity_dims']['frame2_hip']}\n")
                f.write(f"  Orientation (Frame 1): {model_data['data_structure']['orientation_dims']['frame1_hip']}\n")
                f.write(f"  Orientation (Frame 2): {model_data['data_structure']['orientation_dims']['frame2_hip']}\n\n")
                
                f.write("=== Training Statistics ===\n")
                f.write(f"Demonstrations: {len(model_data['individual_demos'])}\n")
                f.write(f"Total data points: {len(model_data['training_data'])}\n\n")
                
                f.write("=== Model Performance ===\n")
                for metric, value in model_data['metrics'].items():
                    f.write(f"  {metric}: {value:.3f}\n")
            
            print(f"✓ Model information saved to: {info_file}")
            
        except Exception as e:
            print(f"✗ Error saving model: {e}")
    
    def load_tpgmm_model(self, filename: str) -> Optional[Dict]:
        """Load TP-GMM model"""
        try:
            model_data = joblib.load(filename)
            print(f"✓ TP-GMM model loaded from: {filename}")
            
            # Validate model structure
            required_keys = ['gmm_model', 'training_data', 'data_structure', 'frame_info']
            for key in required_keys:
                if key not in model_data:
                    print(f"Warning: Missing key '{key}' in loaded model")
            
            return model_data
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return None
    
    def visualize_training_data(self, model_data: Dict):
        """Visualize training data with enhanced plots"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('TP-GMM Gait Training Data Analysis\n(Both frames as true ankle positions)', 
                        fontsize=16, fontweight='bold')
            
            data = model_data['training_data']
            pos_dims = model_data['data_structure']['position_dims']
            vel_dims = model_data['data_structure']['velocity_dims']
            orient_dims = model_data['data_structure']['orientation_dims']
            
            # Time vector
            time_col = data[:, 0]
            
            # Frame 1 data (True ankle position from FR1)
            axes[0, 0].plot(time_col, data[:, pos_dims['frame1_true'][0]], 'b-', alpha=0.7, label='X', linewidth=1.5)
            axes[0, 0].plot(time_col, data[:, pos_dims['frame1_true'][1]], 'r-', alpha=0.7, label='Y', linewidth=1.5)
            axes[0, 0].set_title('Frame 1 - True Ankle Position')
            axes[0, 0].set_xlabel('Normalized Time')
            axes[0, 0].set_ylabel('Position (m)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].plot(time_col, data[:, vel_dims['frame1_true'][0]], 'b-', alpha=0.7, label='Vx', linewidth=1.5)
            axes[0, 1].plot(time_col, data[:, vel_dims['frame1_true'][1]], 'r-', alpha=0.7, label='Vy', linewidth=1.5)
            axes[0, 1].set_title('Frame 1 - True Ankle Velocity')
            axes[0, 1].set_xlabel('Normalized Time')
            axes[0, 1].set_ylabel('Velocity (m/s)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            axes[0, 2].plot(time_col, data[:, orient_dims['frame1_true'][0]], 'g-', alpha=0.7, linewidth=1.5)
            axes[0, 2].set_title('Frame 1 - True Ankle Orientation')
            axes[0, 2].set_xlabel('Normalized Time')
            axes[0, 2].set_ylabel('Orientation (rad)')
            axes[0, 2].grid(True, alpha=0.3)
            
            # Frame 2 data (True ankle position from FR2)
            axes[1, 0].plot(time_col, data[:, pos_dims['frame2_true'][0]], 'b--', alpha=0.7, label='X', linewidth=1.5)
            axes[1, 0].plot(time_col, data[:, pos_dims['frame2_true'][1]], 'r--', alpha=0.7, label='Y', linewidth=1.5)
            axes[1, 0].set_title('Frame 2 - True Ankle Position')
            axes[1, 0].set_xlabel('Normalized Time')
            axes[1, 0].set_ylabel('Position (m)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].plot(time_col, data[:, vel_dims['frame2_true'][0]], 'b--', alpha=0.7, label='Vx', linewidth=1.5)
            axes[1, 1].plot(time_col, data[:, vel_dims['frame2_true'][1]], 'r--', alpha=0.7, label='Vy', linewidth=1.5)
            axes[1, 1].set_title('Frame 2 - True Ankle Velocity')
            axes[1, 1].set_xlabel('Normalized Time')
            axes[1, 1].set_ylabel('Velocity (m/s)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            axes[1, 2].plot(time_col, data[:, orient_dims['frame2_true'][0]], 'g--', alpha=0.7, linewidth=1.5)
            axes[1, 2].set_title('Frame 2 - True Ankle Orientation')
            axes[1, 2].set_xlabel('Normalized Time')
            axes[1, 2].set_ylabel('Orientation (rad)')
            axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('tpgmm_training_data_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error in visualization: {e}")
    
    def visualize_2d_trajectories(self, model_data: Dict):
        """Visualize 2D trajectories for both frames"""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('2D Gait Trajectories (Both frames as true ankle positions)', 
                        fontsize=16, fontweight='bold')
            
            data = model_data['training_data']
            pos_dims = model_data['data_structure']['position_dims']
            
            # Frame 1 trajectory (True ankle position from FR1)
            x1, y1 = data[:, pos_dims['frame1_true'][0]], data[:, pos_dims['frame1_true'][1]]
            axes[0].plot(x1, y1, 'b-', alpha=0.7, linewidth=2, label='Trajectory')
            axes[0].scatter(x1[0], y1[0], c='green', s=100, label='Start', zorder=5)
            axes[0].scatter(x1[-1], y1[-1], c='red', s=100, label='End', zorder=5)
            axes[0].set_title('Frame 1 - True Ankle Position')
            axes[0].set_xlabel('X Position (m)')
            axes[0].set_ylabel('Y Position (m)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].axis('equal')
            
            # Frame 2 trajectory (True ankle position from FR2)
            x2, y2 = data[:, pos_dims['frame2_true'][0]], data[:, pos_dims['frame2_true'][1]]
            axes[1].plot(x2, y2, 'r--', alpha=0.7, linewidth=2, label='Trajectory')
            axes[1].scatter(x2[0], y2[0], c='green', s=100, label='Start', zorder=5)
            axes[1].scatter(x2[-1], y2[-1], c='red', s=100, label='End', zorder=5)
            axes[1].set_title('Frame 2 - True Ankle Position')
            axes[1].set_xlabel('X Position (m)')
            axes[1].set_ylabel('Y Position (m)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].axis('equal')
            
            # Overlay comparison
            axes[2].plot(x1, y1, 'b-', alpha=0.8, linewidth=2, label='Frame 1 (True)')
            axes[2].plot(x2, y2, 'r--', alpha=0.8, linewidth=2, label='Frame 2 (True)')
            axes[2].scatter(x1[0], y1[0], c='green', s=100, zorder=5)
            axes[2].scatter(x1[-1], y1[-1], c='red', s=100, zorder=5)
            axes[2].set_title('True Ankle Trajectory Comparison')
            axes[2].set_xlabel('X Position (m)')
            axes[2].set_ylabel('Y Position (m)')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            axes[2].axis('equal')
            
            # Add coincidence statistics
            mean_diff_x = np.mean(x1 - x2)
            mean_diff_y = np.mean(y1 - y2)
            rms_diff = np.sqrt(np.mean((x1 - x2)**2 + (y1 - y2)**2))
            max_diff = np.max(np.sqrt((x1 - x2)**2 + (y1 - y2)**2))
            
            axes[2].text(0.02, 0.98, f'Mean ΔX: {mean_diff_x:.6f}m\nMean ΔY: {mean_diff_y:.6f}m\nRMS diff: {rms_diff:.6f}m\nMax diff: {max_diff:.6f}m', 
                        transform=axes[2].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgreen' if rms_diff < 0.001 else 'lightyellow', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig('tpgmm_2d_trajectories.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error in 2D visualization: {e}")
    
    def create_gait_comparison_plots(self, model_data: Dict):
        """
        Create comprehensive comparison plots similar to gait_analysis_python.py
        Shows original vs transformed data and trajectory coincidence analysis
        """
        try:
            print("Creating comprehensive gait comparison plots...")
            
            data = model_data['training_data']
            pos_dims = model_data['data_structure']['position_dims']
            
            # Extract data for both frames
            time_col = data[:, 0]
            
            # Frame 1 (True ankle position from FR1)
            pos_FR1_true = data[:, [pos_dims['frame1_true'][0], pos_dims['frame1_true'][1]]]
            
            # Frame 2 (True ankle position from FR2) 
            pos_FR2_true = data[:, [pos_dims['frame2_true'][0], pos_dims['frame2_true'][1]]]
            
            # For comparison, we'll use the first demonstration to show original vs transformed
            if 'individual_demos' in model_data and len(model_data['individual_demos']) > 0:
                demo = model_data['individual_demos'][0]
                demo_time = demo[:, 0]
                demo_pos_fr1 = demo[:, [pos_dims['frame1_true'][0], pos_dims['frame1_true'][1]]]
                demo_pos_fr2 = demo[:, [pos_dims['frame2_true'][0], pos_dims['frame2_true'][1]]]
            else:
                demo_time = time_col
                demo_pos_fr1 = pos_FR1_true
                demo_pos_fr2 = pos_FR2_true
            
            # Create main comparison figure (similar to gait_analysis_python.py)
            fig = plt.figure(figsize=(20, 14))
            fig.suptitle('TP-GMM Gait Trajectory Analysis - True Ankle Positions', fontsize=18, fontweight='bold')
            
            # Subplot 1: All training data trajectories
            ax1 = plt.subplot(2, 4, 1)
            plt.plot(pos_FR1_true[:, 0], pos_FR1_true[:, 1], 'b-', linewidth=1, alpha=0.7, label='FR1 True')
            plt.plot(pos_FR2_true[:, 0], pos_FR2_true[:, 1], 'r-', linewidth=1, alpha=0.7, label='FR2 True')
            plt.xlabel('X Position (m)')
            plt.ylabel('Y Position (m)')
            plt.title('All Training Data\n(True Ankle Positions)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            
            # Subplot 2: First demonstration overlay
            ax2 = plt.subplot(2, 4, 2)
            plt.plot(demo_pos_fr1[:, 0], demo_pos_fr1[:, 1], 'b-', linewidth=2, label='FR1 True')
            plt.plot(demo_pos_fr2[:, 0], demo_pos_fr2[:, 1], 'r--', linewidth=2, label='FR2 True')
            plt.scatter(demo_pos_fr1[0, 0], demo_pos_fr1[0, 1], c='green', s=100, label='Start', zorder=5)
            plt.scatter(demo_pos_fr1[-1, 0], demo_pos_fr1[-1, 1], c='red', s=100, label='End', zorder=5)
            plt.xlabel('X Position (m)')
            plt.ylabel('Y Position (m)')
            plt.title('Single Demo - True Positions\n(Should be Coincident)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            
            # Subplot 3: X-component time series
            ax3 = plt.subplot(2, 4, 3)
            plt.plot(demo_time, demo_pos_fr1[:, 0], 'b-', linewidth=2, label='FR1 X')
            plt.plot(demo_time, demo_pos_fr2[:, 0], 'r--', linewidth=2, label='FR2 X')
            plt.xlabel('Normalized Time')
            plt.ylabel('X Position (m)')
            plt.title('X-Component vs Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Subplot 4: Y-component time series
            ax4 = plt.subplot(2, 4, 4)
            plt.plot(demo_time, demo_pos_fr1[:, 1], 'b-', linewidth=2, label='FR1 Y')
            plt.plot(demo_time, demo_pos_fr2[:, 1], 'r--', linewidth=2, label='FR2 Y')
            plt.xlabel('Normalized Time')
            plt.ylabel('Y Position (m)')
            plt.title('Y-Component vs Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Subplot 5: Difference between trajectories
            ax5 = plt.subplot(2, 4, 5)
            diff_x = demo_pos_fr1[:, 0] - demo_pos_fr2[:, 0]
            diff_y = demo_pos_fr1[:, 1] - demo_pos_fr2[:, 1]
            plt.plot(demo_time, diff_x, 'g-', linewidth=2, label='X Difference')
            plt.plot(demo_time, diff_y, 'm-', linewidth=2, label='Y Difference')
            plt.xlabel('Normalized Time')
            plt.ylabel('Position Difference (m)')
            plt.title('Trajectory Differences (FR1 - FR2)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Subplot 6: Euclidean distance
            ax6 = plt.subplot(2, 4, 6)
            euclidean_dist = np.sqrt(diff_x**2 + diff_y**2)
            plt.plot(demo_time, euclidean_dist, 'k-', linewidth=2)
            plt.xlabel('Normalized Time')
            plt.ylabel('Euclidean Distance (m)')
            plt.title('Distance Between True Positions')
            plt.grid(True, alpha=0.3)
            
            # Calculate and display statistics
            mean_diff_x = np.mean(diff_x)
            mean_diff_y = np.mean(diff_y)
            mean_euclidean = np.mean(euclidean_dist)
            max_euclidean = np.max(euclidean_dist)
            std_euclidean = np.std(euclidean_dist)
            
            # Subplot 7: Statistics summary
            ax7 = plt.subplot(2, 4, 7)
            ax7.axis('off')
            stats_text = f"""Coincidence Statistics:
            
Mean X Difference: {mean_diff_x:.6f} m
Mean Y Difference: {mean_diff_y:.6f} m

Euclidean Distance:
  Mean: {mean_euclidean:.6f} m
  Max:  {max_euclidean:.6f} m  
  Std:  {std_euclidean:.6f} m

Trajectory Ranges:
FR1 X: [{np.min(demo_pos_fr1[:, 0]):.3f}, {np.max(demo_pos_fr1[:, 0]):.3f}]
FR1 Y: [{np.min(demo_pos_fr1[:, 1]):.3f}, {np.max(demo_pos_fr1[:, 1]):.3f}]

FR2 X: [{np.min(demo_pos_fr2[:, 0]):.3f}, {np.max(demo_pos_fr2[:, 0]):.3f}]
FR2 Y: [{np.min(demo_pos_fr2[:, 1]):.3f}, {np.max(demo_pos_fr2[:, 1]):.3f}]

Status: {'✓ COINCIDENT' if mean_euclidean < 0.001 else '⚠ NOT COINCIDENT'}"""
            
            ax7.text(0.05, 0.95, stats_text, transform=ax7.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            ax7.set_title('Transformation Statistics')
            
            # Subplot 8: 3D trajectory visualization
            ax8 = fig.add_subplot(2, 4, 8, projection='3d')
            ax8.plot(demo_pos_fr1[:, 0], demo_pos_fr1[:, 1], demo_time, 'b-', linewidth=2, label='FR1 True')
            ax8.plot(demo_pos_fr2[:, 0], demo_pos_fr2[:, 1], demo_time, 'r--', linewidth=2, label='FR2 True')
            ax8.set_xlabel('X Position (m)')
            ax8.set_ylabel('Y Position (m)')
            ax8.set_zlabel('Normalized Time')
            ax8.set_title('3D Trajectory Comparison')
            ax8.legend()
            
            plt.tight_layout()
            plt.savefig('tpgmm_gait_trajectory_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Create additional detailed analysis figure
            self.create_detailed_trajectory_analysis(demo_time, demo_pos_fr1, demo_pos_fr2)
            
            # Print statistics to console
            print(f"\n=== TRAJECTORY COINCIDENCE ANALYSIS ===")
            print(f"Mean X Difference: {mean_diff_x:.8f} m")
            print(f"Mean Y Difference: {mean_diff_y:.8f} m")
            print(f"Mean Euclidean Distance: {mean_euclidean:.8f} m")
            print(f"Max Euclidean Distance: {max_euclidean:.8f} m")
            print(f"Std Euclidean Distance: {std_euclidean:.8f} m")
            
            if mean_euclidean < 0.001:
                print("✓ SUCCESS: Trajectories are COINCIDENT - transformations working correctly!")
            else:
                print("⚠ WARNING: Trajectories are NOT coincident - check transformations")
            print("=" * 50)
            
        except Exception as e:
            print(f"Error in gait comparison plots: {e}")
    
    def create_detailed_trajectory_analysis(self, time, pos_FR1, pos_FR2):
        """
        Create detailed trajectory analysis with multiple views
        """
        try:
            fig = plt.figure(figsize=(16, 12))
            fig.suptitle('Detailed Trajectory Analysis - Coincidence Verification', fontsize=16, fontweight='bold')
            
            # Calculate differences
            diff_x = pos_FR1[:, 0] - pos_FR2[:, 0]
            diff_y = pos_FR1[:, 1] - pos_FR2[:, 1]
            euclidean_dist = np.sqrt(diff_x**2 + diff_y**2)
            
            # 1. Trajectory overlay with error bars
            ax1 = plt.subplot(2, 3, 1)
            plt.plot(pos_FR1[:, 0], pos_FR1[:, 1], 'b-', linewidth=3, alpha=0.7, label='FR1 True')
            plt.plot(pos_FR2[:, 0], pos_FR2[:, 1], 'r--', linewidth=2, alpha=0.9, label='FR2 True')
            
            # Add error visualization at certain points
            step = max(1, len(time) // 20)  # Show error bars at ~20 points
            for i in range(0, len(time), step):
                plt.plot([pos_FR1[i, 0], pos_FR2[i, 0]], [pos_FR1[i, 1], pos_FR2[i, 1]], 
                        'k-', alpha=0.3, linewidth=1)
            
            plt.scatter(pos_FR1[0, 0], pos_FR1[0, 1], c='green', s=100, label='Start', zorder=5)
            plt.scatter(pos_FR1[-1, 0], pos_FR1[-1, 1], c='red', s=100, label='End', zorder=5)
            plt.xlabel('X Position (m)')
            plt.ylabel('Y Position (m)')
            plt.title('Trajectory Overlay with Error Lines')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            
            # 2. Error magnitude heatmap
            ax2 = plt.subplot(2, 3, 2)
            scatter = plt.scatter(pos_FR1[:, 0], pos_FR1[:, 1], c=euclidean_dist, 
                                cmap='hot', s=30, alpha=0.8)
            plt.colorbar(scatter, label='Error Distance (m)')
            plt.xlabel('X Position (m)')
            plt.ylabel('Y Position (m)')
            plt.title('Error Magnitude Heatmap')
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            
            # 3. Error vs time
            ax3 = plt.subplot(2, 3, 3)
            plt.plot(time, euclidean_dist, 'k-', linewidth=2)
            plt.fill_between(time, euclidean_dist, alpha=0.3)
            plt.xlabel('Normalized Time')
            plt.ylabel('Euclidean Distance (m)')
            plt.title('Error vs Time')
            plt.grid(True, alpha=0.3)
            
            # 4. X and Y differences
            ax4 = plt.subplot(2, 3, 4)
            plt.plot(time, diff_x, 'b-', linewidth=2, label='X Difference')
            plt.plot(time, diff_y, 'r-', linewidth=2, label='Y Difference')
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            plt.xlabel('Normalized Time')
            plt.ylabel('Position Difference (m)')
            plt.title('Component-wise Differences')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 5. Error distribution histogram
            ax5 = plt.subplot(2, 3, 5)
            plt.hist(euclidean_dist, bins=30, alpha=0.7, edgecolor='black')
            plt.axvline(np.mean(euclidean_dist), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(euclidean_dist):.6f}')
            plt.axvline(np.median(euclidean_dist), color='green', linestyle='--', 
                       label=f'Median: {np.median(euclidean_dist):.6f}')
            plt.xlabel('Euclidean Distance (m)')
            plt.ylabel('Frequency')
            plt.title('Error Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 6. Cumulative error
            ax6 = plt.subplot(2, 3, 6)
            cumulative_error = np.cumsum(euclidean_dist)
            plt.plot(time, cumulative_error, 'purple', linewidth=2)
            plt.xlabel('Normalized Time')
            plt.ylabel('Cumulative Error (m)')
            plt.title('Cumulative Error Accumulation')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('tpgmm_detailed_trajectory_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error in detailed trajectory analysis: {e}")
    
    def visualize_all_demonstrations(self, model_data: Dict):
        """
        Visualize all individual demonstrations to check coincidence across all data
        """
        try:
            if 'individual_demos' not in model_data:
                print("No individual demonstrations available")
                return
            
            demos = model_data['individual_demos']
            pos_dims = model_data['data_structure']['position_dims']
            
            n_demos = len(demos)
            print(f"Visualizing {n_demos} individual demonstrations...")
            
            # Create figure for all demonstrations
            fig = plt.figure(figsize=(20, 15))
            fig.suptitle(f'All {n_demos} Demonstrations - Coincidence Check', fontsize=16, fontweight='bold')
            
            # Calculate grid size
            cols = min(4, n_demos)
            rows = (n_demos + cols - 1) // cols
            
            coincidence_stats = []
            
            for i, demo in enumerate(demos):
                ax = plt.subplot(rows, cols, i + 1)
                
                # Extract positions
                pos_fr1 = demo[:, [pos_dims['frame1_true'][0], pos_dims['frame1_true'][1]]]
                pos_fr2 = demo[:, [pos_dims['frame2_true'][0], pos_dims['frame2_true'][1]]]
                
                # Plot trajectories
                plt.plot(pos_fr1[:, 0], pos_fr1[:, 1], 'b-', linewidth=2, alpha=0.8, label='FR1')
                plt.plot(pos_fr2[:, 0], pos_fr2[:, 1], 'r--', linewidth=1.5, alpha=0.8, label='FR2')
                
                # Calculate coincidence metric
                diff = np.sqrt(np.sum((pos_fr1 - pos_fr2)**2, axis=1))
                mean_diff = np.mean(diff)
                max_diff = np.max(diff)
                
                coincidence_stats.append({
                    'demo': i+1,
                    'mean_error': mean_diff,
                    'max_error': max_diff
                })
                
                # Color-code title based on coincidence
                title_color = 'green' if mean_diff < 0.001 else 'orange' if mean_diff < 0.01 else 'red'
                
                plt.title(f'Demo {i+1}\nMean: {mean_diff:.6f}m', color=title_color, fontweight='bold')
                plt.xlabel('X Position (m)')
                plt.ylabel('Y Position (m)')
                
                if i == 0:  # Only show legend for first subplot
                    plt.legend(fontsize=8)
                
                plt.grid(True, alpha=0.3)
                plt.axis('equal')
            
            plt.tight_layout()
            plt.savefig('tpgmm_all_demonstrations_coincidence.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Create summary statistics plot
            self.create_coincidence_summary(coincidence_stats)
            
        except Exception as e:
            print(f"Error in demonstrations visualization: {e}")
    
    def create_coincidence_summary(self, coincidence_stats):
        """
        Create summary plot of coincidence statistics across all demonstrations
        """
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Coincidence Statistics Summary Across All Demonstrations', fontsize=16, fontweight='bold')
            
            demo_nums = [stat['demo'] for stat in coincidence_stats]
            mean_errors = [stat['mean_error'] for stat in coincidence_stats]
            max_errors = [stat['max_error'] for stat in coincidence_stats]
            
            # 1. Mean error per demonstration
            ax1 = axes[0]
            bars = ax1.bar(demo_nums, mean_errors, alpha=0.7)
            ax1.axhline(y=0.001, color='red', linestyle='--', label='Target threshold (0.001m)')
            ax1.set_xlabel('Demonstration Number')
            ax1.set_ylabel('Mean Error (m)')
            ax1.set_title('Mean Coincidence Error')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Color bars based on threshold
            for bar, error in zip(bars, mean_errors):
                if error < 0.001:
                    bar.set_color('green')
                elif error < 0.01:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
            
            # 2. Max error per demonstration
            ax2 = axes[1]
            bars2 = ax2.bar(demo_nums, max_errors, alpha=0.7, color='lightcoral')
            ax2.axhline(y=0.01, color='red', linestyle='--', label='Warning threshold (0.01m)')
            ax2.set_xlabel('Demonstration Number')
            ax2.set_ylabel('Max Error (m)')
            ax2.set_title('Maximum Coincidence Error')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Error distribution
            ax3 = axes[2]
            ax3.hist(mean_errors, bins=15, alpha=0.7, edgecolor='black', label='Mean Errors')
            ax3.axvline(np.mean(mean_errors), color='red', linestyle='--', 
                       label=f'Overall Mean: {np.mean(mean_errors):.6f}m')
            ax3.axvline(0.001, color='green', linestyle='--', label='Target: 0.001m')
            ax3.set_xlabel('Mean Error (m)')
            ax3.set_ylabel('Number of Demonstrations')
            ax3.set_title('Error Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('tpgmm_coincidence_summary.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Print summary statistics
            print(f"\n=== COINCIDENCE SUMMARY ACROSS ALL DEMONSTRATIONS ===")
            print(f"Total demonstrations: {len(coincidence_stats)}")
            print(f"Overall mean error: {np.mean(mean_errors):.8f} m")
            print(f"Overall std error: {np.std(mean_errors):.8f} m")
            print(f"Overall max error: {np.max(max_errors):.8f} m")
            
            excellent_count = sum(1 for e in mean_errors if e < 0.001)
            good_count = sum(1 for e in mean_errors if 0.001 <= e < 0.01)
            poor_count = sum(1 for e in mean_errors if e >= 0.01)
            
            print(f"\nQuality assessment:")
            print(f"  Excellent (< 0.001m): {excellent_count}/{len(coincidence_stats)} ({100*excellent_count/len(coincidence_stats):.1f}%)")
            print(f"  Good (< 0.01m): {good_count}/{len(coincidence_stats)} ({100*good_count/len(coincidence_stats):.1f}%)")
            print(f"  Poor (≥ 0.01m): {poor_count}/{len(coincidence_stats)} ({100*poor_count/len(coincidence_stats):.1f}%)")
            
            if excellent_count == len(coincidence_stats):
                print("✓ PERFECT: All demonstrations are perfectly coincident!")
            elif excellent_count + good_count == len(coincidence_stats):
                print("✓ GOOD: All demonstrations meet acceptable coincidence criteria")
            else:
                print("⚠ WARNING: Some demonstrations have poor coincidence - check transformations")
            
            print("=" * 60)

            fig = plt.figure(figsize=(18, 12))
            axes[0, 2].grid(True, alpha=0.3)
            
            # Frame 2 data (Global→Hip transformed)
            axes[1, 0].plot(time_col, data[:, pos_dims['frame2_hip'][0]], 'b--', alpha=0.7, label='X', linewidth=1.5)
            axes[1, 0].plot(time_col, data[:, pos_dims['frame2_hip'][1]], 'r--', alpha=0.7, label='Y', linewidth=1.5)
            axes[1, 0].set_title('Frame 2 (Global→Hip Transformed) - Position')
            axes[1, 0].set_xlabel('Normalized Time')
            axes[1, 0].set_ylabel('Position (m)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].plot(time_col, data[:, vel_dims['frame2_hip'][0]], 'b--', alpha=0.7, label='Vx', linewidth=1.5)
            axes[1, 1].plot(time_col, data[:, vel_dims['frame2_hip'][1]], 'r--', alpha=0.7, label='Vy', linewidth=1.5)
            axes[1, 1].set_title('Frame 2 (Global→Hip Transformed) - Velocity')
            axes[1, 1].set_xlabel('Normalized Time')
            axes[1, 1].set_ylabel('Velocity (m/s)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            axes[1, 2].plot(time_col, data[:, orient_dims['frame2_hip'][0]], 'g--', alpha=0.7, linewidth=1.5)
            axes[1, 2].set_title('Frame 2 (Global→Hip Transformed) - Orientation')
            axes[1, 2].set_xlabel('Normalized Time')
            axes[1, 2].set_ylabel('Orientation (rad)')
            axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('tpgmm_training_data_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error in visualization: {e}")
    
    def visualize_2d_trajectories(self, model_data: Dict):
        """Visualize 2D trajectories for both frames"""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('2D Gait Trajectories (Both frames in Hip coordinate system)', 
                        fontsize=16, fontweight='bold')
            
            data = model_data['training_data']
            pos_dims = model_data['data_structure']['position_dims']
            
            # Frame 1 trajectory (Hip frame - original)
            x1, y1 = data[:, pos_dims['frame1_hip'][0]], data[:, pos_dims['frame1_hip'][1]]
            axes[0].plot(x1, y1, 'b-', alpha=0.7, linewidth=2, label='Trajectory')
            axes[0].scatter(x1[0], y1[0], c='green', s=100, label='Start', zorder=5)
            axes[0].scatter(x1[-1], y1[-1], c='red', s=100, label='End', zorder=5)
            axes[0].set_title('Frame 1 (Hip Original)')
            axes[0].set_xlabel('X Position (m)')
            axes[0].set_ylabel('Y Position (m)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].axis('equal')
            
            # Frame 2 trajectory (Global→Hip transformed)
            x2, y2 = data[:, pos_dims['frame2_hip'][0]], data[:, pos_dims['frame2_hip'][1]]
            axes[1].plot(x2, y2, 'r--', alpha=0.7, linewidth=2, label='Trajectory')
            axes[1].scatter(x2[0], y2[0], c='green', s=100, label='Start', zorder=5)
            axes[1].scatter(x2[-1], y2[-1], c='red', s=100, label='End', zorder=5)
            axes[1].set_title('Frame 2 (Global→Hip Transformed)')
            axes[1].set_xlabel('X Position (m)')
            axes[1].set_ylabel('Y Position (m)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].axis('equal')
            
            # Overlay comparison
            axes[2].plot(x1, y1, 'b-', alpha=0.8, linewidth=2, label='Frame 1 (Hip)')
            axes[2].plot(x2, y2, 'r--', alpha=0.8, linewidth=2, label='Frame 2 (Transformed)')
            axes[2].scatter(x1[0], y1[0], c='green', s=100, zorder=5)
            axes[2].scatter(x1[-1], y1[-1], c='red', s=100, zorder=5)
            axes[2].set_title('Trajectory Comparison')
            axes[2].set_xlabel('X Position (m)')
            axes[2].set_ylabel('Y Position (m)')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            axes[2].axis('equal')
            
            # Add statistics
            mean_diff_x = np.mean(x1 - x2)
            mean_diff_y = np.mean(y1 - y2)
            rms_diff = np.sqrt(np.mean((x1 - x2)**2 + (y1 - y2)**2))
            
            axes[2].text(0.02, 0.98, f'Mean ΔX: {mean_diff_x:.4f}m\nMean ΔY: {mean_diff_y:.4f}m\nRMS diff: {rms_diff:.4f}m', 
                        transform=axes[2].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig('tpgmm_2d_trajectories.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error in 2D visualization: {e}")
    
    def analyze_latent_space_pca(self, model_data: Dict) -> Dict:
        """Enhanced PCA analysis of the latent space"""
        print("\n=== PCA Latent Space Analysis ===")
        
        # Get training data (excluding time dimension)
        data = model_data['training_data'][:, 1:]  # Remove time column
        
        print(f"Data shape for PCA: {data.shape}")
        print(f"Original dimensions: {data.shape[1]} (10D TP-GMM data)")
        
        # Standardize the data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Different numbers of PCA components to analyze
        n_components_list = [2, 3, 5, 8]
        pca_results = {}
        
        for n_comp in n_components_list:
            if n_comp <= data.shape[1]:
                print(f"\nAnalyzing PCA with {n_comp} components...")
                
                # Fit PCA
                pca = PCA(n_components=n_comp, random_state=42)
                data_pca = pca.fit_transform(data_scaled)
                
                # Calculate metrics
                explained_var_ratio = pca.explained_variance_ratio_
                cumulative_var = np.cumsum(explained_var_ratio)
                
                print(f"  Explained variance per component: {explained_var_ratio}")
                print(f"  Cumulative explained variance: {cumulative_var[-1]:.3f}")
                
                # Analyze component loadings
                components = pca.components_
                feature_names = ['F1_X', 'F1_Y', 'F1_Vx', 'F1_Vy', 'F1_θ',
                               'F2_X', 'F2_Y', 'F2_Vx', 'F2_Vy', 'F2_θ']
                
                # Store results
                pca_results[f'{n_comp}D'] = {
                    'pca_model': pca,
                    'transformed_data': data_pca,
                    'explained_variance_ratio': explained_var_ratio,
                    'cumulative_variance': cumulative_var,
                    'components': components,
                    'scaler': scaler,
                    'feature_names': feature_names
                }
        
        # Store in model data
        model_data['pca_analysis'] = pca_results
        
        print(f"\n✓ PCA analysis completed for {len(pca_results)} component configurations")
        return pca_results
    
    def visualize_pca_analysis(self, model_data: Dict):
        """Comprehensive PCA visualization"""
        if 'pca_analysis' not in model_data:
            print("Running PCA analysis first...")
            self.analyze_latent_space_pca(model_data)
        
        pca_results = model_data['pca_analysis']
        
        try:
            # Create comprehensive PCA visualization
            fig = plt.figure(figsize=(20, 15))
            fig.suptitle('TP-GMM PCA Latent Space Analysis', fontsize=18, fontweight='bold')
            
            # 1. Explained variance comparison
            ax1 = plt.subplot(3, 3, 1)
            for pca_key in ['2D', '3D', '5D', '8D']:
                if pca_key in pca_results:
                    explained_var = pca_results[pca_key]['explained_variance_ratio']
                    cumulative_var = pca_results[pca_key]['cumulative_variance']
                    x = range(1, len(explained_var) + 1)
                    ax1.plot(x, cumulative_var, 'o-', label=f'{pca_key} PCA', linewidth=2, markersize=6)
            ax1.set_xlabel('Principal Component')
            ax1.set_ylabel('Cumulative Explained Variance')
            ax1.set_title('Explained Variance Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. 2D PCA scatter plot
            if '2D' in pca_results:
                ax2 = plt.subplot(3, 3, 2)
                data_2d = pca_results['2D']['transformed_data']
                explained_var = pca_results['2D']['explained_variance_ratio']
                
                # Color by time progression
                n_points = len(data_2d)
                colors = np.linspace(0, 1, n_points)
                scatter = ax2.scatter(data_2d[:, 0], data_2d[:, 1], c=colors, cmap='viridis', alpha=0.6, s=20)
                ax2.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
                ax2.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
                ax2.set_title('2D PCA Space (Time Colored)')
                ax2.grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=ax2, label='Time progression')
            
            # 3. Component loadings heatmap (2D PCA)
            if '2D' in pca_results:
                ax3 = plt.subplot(3, 3, 3)
                components = pca_results['2D']['components']
                feature_names = pca_results['2D']['feature_names']
                
                im = ax3.imshow(components, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
                ax3.set_xticks(range(len(feature_names)))
                ax3.set_xticklabels(feature_names, rotation=45, ha='right')
                ax3.set_yticks(range(len(components)))
                ax3.set_yticklabels([f'PC{i+1}' for i in range(len(components))])
                ax3.set_title('2D PCA Component Loadings')
                plt.colorbar(im, ax=ax3, label='Loading strength')
                
                # Add text annotations for significant loadings
                for i in range(components.shape[0]):
                    for j in range(components.shape[1]):
                        value = components[i, j]
                        if abs(value) > 0.3:
                            ax3.text(j, i, f'{value:.2f}', ha='center', va='center',
                                   color='white' if abs(value) > 0.6 else 'black', fontweight='bold')
            
            # 4. Individual trajectory projections (first few)
            if '2D' in pca_results and 'individual_demos' in model_data:
                ax4 = plt.subplot(3, 3, 4)
                pca_model = pca_results['2D']['pca_model']
                scaler = pca_results['2D']['scaler']
                
                colors = plt.cm.tab10(np.linspace(0, 1, 5))
                for i, demo in enumerate(model_data['individual_demos'][:5]):
                    demo_data = demo[:, 1:]  # Remove time
                    demo_scaled = scaler.transform(demo_data)
                    demo_pca = pca_model.transform(demo_scaled)
                    ax4.plot(demo_pca[:, 0], demo_pca[:, 1], color=colors[i], alpha=0.8, 
                            linewidth=2, label=f'Demo {i+1}')
                    # Mark start and end
                    ax4.scatter(demo_pca[0, 0], demo_pca[0, 1], color=colors[i], s=60, marker='o')
                    ax4.scatter(demo_pca[-1, 0], demo_pca[-1, 1], color=colors[i], s=60, marker='s')
                
                ax4.set_xlabel('PC1')
                ax4.set_ylabel('PC2')
                ax4.set_title('Individual Trajectories in PCA Space')
                ax4.legend(fontsize=8)
                ax4.grid(True, alpha=0.3)
            
            # 5. 3D PCA (first 3 components projected to 2D views)
            if '3D' in pca_results:
                data_3d = pca_results['3D']['transformed_data']
                explained_var = pca_results['3D']['explained_variance_ratio']
                
                ax5 = plt.subplot(3, 3, 5)
                ax5.scatter(data_3d[:, 0], data_3d[:, 2], c=colors, cmap='plasma', alpha=0.6, s=20)
                ax5.set_xlabel(f'PC1 ({explained_var[0]:.1%})')
                ax5.set_ylabel(f'PC3 ({explained_var[2]:.1%})')
                ax5.set_title('3D PCA: PC1 vs PC3')
                ax5.grid(True, alpha=0.3)
            
            # 6. Reconstruction error analysis
            ax6 = plt.subplot(3, 3, 6)
            reconstruction_errors = []
            component_counts = []
            
            for pca_key in ['2D', '3D', '5D', '8D']:
                if pca_key in pca_results:
                    pca_model = pca_results[pca_key]['pca_model']
                    scaler = pca_results[pca_key]['scaler']
                    original_data = model_data['training_data'][:, 1:]
                    
                    # Transform and reconstruct
                    data_scaled = scaler.transform(original_data)
                    data_pca = pca_model.transform(data_scaled)
                    data_reconstructed = pca_model.inverse_transform(data_pca)
                    data_original_scale = scaler.inverse_transform(data_reconstructed)
                    
                    # Calculate reconstruction error
                    error = np.mean(np.sum((original_data - data_original_scale)**2, axis=1))
                    reconstruction_errors.append(error)
                    component_counts.append(int(pca_key[0]))
            
            if reconstruction_errors:
                ax6.plot(component_counts, reconstruction_errors, 'bo-', linewidth=2, markersize=8)
                ax6.set_xlabel('Number of PCA Components')
                ax6.set_ylabel('Mean Reconstruction Error')
                ax6.set_title('Reconstruction Error vs Components')
                ax6.grid(True, alpha=0.3)
                ax6.set_yscale('log')
            
            # 7. Feature importance analysis
            if '5D' in pca_results:
                ax7 = plt.subplot(3, 3, 7)
                components = pca_results['5D']['components']
                feature_names = pca_results['5D']['feature_names']
                
                # Calculate total absolute contribution per feature
                feature_importance = np.sum(np.abs(components), axis=0)
                sorted_indices = np.argsort(feature_importance)[::-1]
                
                bars = ax7.bar(range(len(feature_importance)), feature_importance[sorted_indices])
                ax7.set_xticks(range(len(feature_names)))
                ax7.set_xticklabels([feature_names[i] for i in sorted_indices], rotation=45, ha='right')
                ax7.set_ylabel('Total Absolute Loading')
                ax7.set_title('Feature Importance (5D PCA)')
                ax7.grid(True, alpha=0.3)
                
                # Color bars by frame
                for i, bar in enumerate(bars):
                    if sorted_indices[i] < 5:  # Frame 1
                        bar.set_color('blue')
                    else:  # Frame 2
                        bar.set_color('red')
            
            # 8. Trajectory density in PCA space
            if '2D' in pca_results:
                ax8 = plt.subplot(3, 3, 8)
                data_2d = pca_results['2D']['transformed_data']
                
                # Create 2D histogram
                hist, xedges, yedges = np.histogram2d(data_2d[:, 0], data_2d[:, 1], bins=25, density=True)
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                
                im = ax8.imshow(hist.T, origin='lower', extent=extent, cmap='Blues', alpha=0.8)
                ax8.scatter(data_2d[:, 0], data_2d[:, 1], c='red', alpha=0.3, s=10)
                ax8.set_xlabel('PC1')
                ax8.set_ylabel('PC2')
                ax8.set_title('Trajectory Density in PCA Space')
                plt.colorbar(im, ax=ax8, label='Density')
            
            # 9. Variance explained summary
            ax9 = plt.subplot(3, 3, 9)
            summary_data = []
            labels = []
            
            for pca_key in ['2D', '3D', '5D', '8D']:
                if pca_key in pca_results:
                    cumulative_var = pca_results[pca_key]['cumulative_variance'][-1]
                    summary_data.append(cumulative_var)
                    labels.append(pca_key)
            
            if summary_data:
                bars = ax9.bar(labels, summary_data, color=['skyblue', 'lightgreen', 'orange', 'lightcoral'])
                ax9.set_ylabel('Cumulative Explained Variance')
                ax9.set_title('PCA Summary')
                ax9.set_ylim(0, 1)
                ax9.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, value in zip(bars, summary_data):
                    height = bar.get_height()
                    ax9.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.1%}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('tpgmm_pca_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error in PCA visualization: {e}")

def main():
    """
    Enhanced main function with better error handling and validation
    """
    # Initialize trainer
    trainer = TPGMMGaitTrainer(reference_frame_id=2, target_frame_id=1)
    
    # Configuration
    mat_file_path = 'new_processed_gait_data.mat'  # Replace with your file path
    model_save_path = 'tpgmm_gait_model_fixed.pkl'
    
    try:
        print("=== TP-GMM Gait Training Pipeline ===")
        print(f"Data file: {mat_file_path}")
        print(f"Model save path: {model_save_path}")
        print(f"Frame configuration: FR2 (Global) -> FR1 (Hip)")
        
        # 1. Load demonstrations
        print("\n=== Step 1: Loading Demonstrations ===")
        demonstrations = trainer.load_demonstrations_from_mat(mat_file_path)
        
        if len(demonstrations) == 0:
            print("✗ No valid demonstrations found! Check your data file.")
            return
        
        print(f"✓ Loaded {len(demonstrations)} valid demonstrations")
        
        # Quick statistics
        total_points = sum(len(demo) for demo in demonstrations)
        avg_length = total_points / len(demonstrations)
        print(f"  Total data points: {total_points}")
        print(f"  Average trajectory length: {avg_length:.1f}")
        
        # 2. Train TP-GMM model
        print("\n=== Step 2: Training TP-GMM Model ===")
        model_data = trainer.train_tpgmm_model(demonstrations)
        
        # 3. Save model
        print("\n=== Step 3: Saving Model ===")
        trainer.save_tpgmm_model(model_data, model_save_path)
        
        # 4. Visualizations
        print("\n=== Step 4: Generating Visualizations ===")
        
        print("Creating training data visualization...")
        trainer.visualize_training_data(model_data)
        
        print("Creating 2D trajectory visualization...")
        trainer.visualize_2d_trajectories(model_data)
        
        print("Creating comprehensive gait comparison plots...")
        trainer.create_gait_comparison_plots(model_data)
        
        print("Visualizing all individual demonstrations...")
        trainer.visualize_all_demonstrations(model_data)
        
        print("Performing PCA analysis...")
        trainer.analyze_latent_space_pca(model_data)
        trainer.visualize_pca_analysis(model_data)
        
        # 5. Model validation
        print("\n=== Step 5: Model Validation ===")
        
        # Test loading the saved model
        print("Testing model loading...")
        loaded_model = trainer.load_tpgmm_model(model_save_path)
        if loaded_model is not None:
            print("✓ Model successfully saved and loaded")
        else:
            print("✗ Model loading failed")
        
        # Summary
        print("\n=== Training Summary ===")
        print(f"✓ Successfully trained TP-GMM model")
        print(f"  Demonstrations processed: {len(demonstrations)}")
        print(f"  GMM components: {model_data['n_components']}")
        print(f"  Data dimension: {model_data['data_structure']['total_dim']}")
        print(f"  Log-likelihood: {model_data['metrics']['log_likelihood']:.2f}")
        print(f"  BIC score: {model_data['metrics']['bic']:.1f}")
        print(f"  Model saved to: {model_save_path}")
        print(f"  Coordinate system: True ankle positions")
        print(f"  Transformation applied: Both FR1 and FR2 inverse transformed")
        
        return model_data
        
    except FileNotFoundError:
        print(f"✗ Error: File {mat_file_path} not found")
        print("Please check the file path and ensure the .mat file exists")
    except Exception as e:
        print(f"✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    model_data = main()