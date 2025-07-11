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