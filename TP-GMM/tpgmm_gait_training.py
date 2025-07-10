import numpy as np
import scipy.io
from sklearn.mixture import GaussianMixture
import joblib
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

class TPGMMGaitTrainer:
    def __init__(self, reference_frame_id=1, target_frame_id=2):
        """
        TP-GMM Trainer for gait data
        
        Args:
            reference_frame_id: ID of the reference frame
            target_frame_id: ID of the target frame
        """
        self.reference_frame_id = reference_frame_id
        self.target_frame_id = target_frame_id
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

        Args:
            mat_file_path (str): The path to the .mat file.

        Returns:
            tuple: (trajectories_fr1, trajectories_fr2, time_data, frame_origins)
                   - trajectories_fr1: list of trajectories in frame 1 [time, x, y, vx, vy, orientation]
                   - trajectories_fr2: list of trajectories in frame 2 [time, x, y, vx, vy, orientation]  
                   - time_data: list of time vectors for each trajectory
                   - frame_origins: list of frame origins for each trajectory [(x1,y1), (x2,y2)]
        """
        try:
            mat_data = scipy.io.loadmat(mat_file_path)
            processed_gait_data = mat_data['processed_gait_data']
            
            trajectories_fr1 = []
            trajectories_fr2 = []
            time_data = []
            frame_origins = []
            
            # The data is in a cell array, so we iterate through it
            for i in range(processed_gait_data.shape[1]):
                # The struct is often nested inside a 1x1 array within the cell
                trial_data = processed_gait_data[0, i][0, 0]
                
                # Extract all data fields
                time = trial_data['time'].flatten().astype(np.float64)
                
                # Frame 1 data
                ankle_pos = trial_data['ankle_pos'].astype(np.float64)  # [200x2]
                ankle_vel = trial_data['ankle_pos_velocity'].astype(np.float64)  # [200x2]
                ankle_orient = trial_data['ankle_orientation'].flatten().astype(np.float64)  # [200x1]
                
                # Frame 2 data  
                ankle_pos_fr2 = trial_data['ankle_pos_FR2'].astype(np.float64)  # [200x2]
                ankle_vel_fr2 = trial_data['ankle_pos_FR2_velocity'].astype(np.float64)  # [200x2]
                ankle_orient_fr2 = trial_data['ankle_orientation_FR2'].flatten().astype(np.float64)  # [200x1]
                
                # Estimate frame origins (could be first point, mean, or predefined)
                # For now, we'll use the last point as the origin for each frame
                origin_fr1 = ankle_pos[-1, :]  # Last point of frame 1
                origin_fr2 = ankle_pos_fr2[-1, :]  # Last point of frame 2
                
                # Store frame origins for this trajectory
                frame_origins.append([origin_fr1, origin_fr2])
                
                # Create complete trajectories for both frames
                # Frame 1: [time, x, y, vx, vy, orientation] = 6 dimensions
                trajectory_fr1 = np.column_stack([
                    time,
                    ankle_pos[:, 0],     # x position
                    ankle_pos[:, 1],     # y position  
                    ankle_vel[:, 0],     # x velocity
                    ankle_vel[:, 1],     # y velocity
                    ankle_orient         # orientation
                ])
                
                # Frame 2: [time, x, y, vx, vy, orientation] = 6 dimensions
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
            print(f"Frame origins computed for each trajectory")
                
            return trajectories_fr1, trajectories_fr2, time_data, frame_origins
        except FileNotFoundError:
            print(f"Error: The file {mat_file_path} was not found.")
            return None, None, None, None
        except KeyError as e:
            print(f"Error: Could not find required field in the .mat file: {e}")
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
        trajectories_fr1, trajectories_fr2, time_data, frame_origins = self.load_gait_data(mat_file_path)
        
        if trajectories_fr1 is None:
            print("Failed to load gait data")
            return []
        
        demonstrations = []
        
        # Process each trajectory
        for i, (traj_fr1, traj_fr2) in enumerate(zip(trajectories_fr1, trajectories_fr2)):
            print(f"Processing trajectory {i+1}/{len(trajectories_fr1)}")
            
            # Process trajectory for TP-GMM
            demo_array = self.process_gait_trajectory(traj_fr1, traj_fr2)
            
            if demo_array is not None and len(demo_array) > 0:
                demonstrations.append(demo_array)
                print(f"✓ Trajectory processed: {len(demo_array)} points")
            else:
                print(f"✗ Invalid or empty trajectory")
                
        return demonstrations
    
    def process_gait_trajectory(self, traj_fr1: np.ndarray, traj_fr2: np.ndarray) -> np.ndarray:
        """
        Process gait trajectory for TP-GMM format
        
        Args:
            traj_fr1: Trajectory in frame 1 [time, x, y, vx, vy, orientation]
            traj_fr2: Trajectory in frame 2 [time, x, y, vx, vy, orientation]
            
        Returns:
            Array with TP-GMM data [N x 10] = [Frame1_data | Frame2_data]
        """
        valid_points = []
        
        # Check if both trajectories have the same length
        if len(traj_fr1) != len(traj_fr2):
            print(f"Warning: Trajectory lengths don't match: {len(traj_fr1)} vs {len(traj_fr2)}")
            min_len = min(len(traj_fr1), len(traj_fr2))
            traj_fr1 = traj_fr1[:min_len]
            traj_fr2 = traj_fr2[:min_len]
        
        for i in range(len(traj_fr1)):
            try:
                # FRAME 1: Extract data (skip time column)
                # [x, y, vx, vy, orientation] = 5D
                frame1_data = traj_fr1[i, 1:6]  # Skip time column
                
                # FRAME 2: Extract data (skip time column)
                # [x, y, vx, vy, orientation] = 5D
                frame2_data = traj_fr2[i, 1:6]  # Skip time column
                
                # Check for valid data (no NaN or infinite values)
                if np.any(np.isnan(frame1_data)) or np.any(np.isnan(frame2_data)):
                    continue
                if np.any(np.isinf(frame1_data)) or np.any(np.isinf(frame2_data)):
                    continue
                
                # TP-GMM DATA: [Frame1 | Frame2]
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
    
    def optimize_n_components(self, data: np.ndarray, max_components: int =20) -> int:
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
                'frame1_dims': list(range(1, self.point_dim + 1)),
                'frame2_dims': list(range(self.point_dim + 1, self.total_dim + 1)),
                'position_dims': {
                    'frame1': [1, 2],      # x, y
                    'frame2': [6, 7]       # x, y
                },
                'velocity_dims': {
                    'frame1': [3, 4],      # vx, vy
                    'frame2': [8, 9]       # vx, vy
                },
                'orientation_dims': {
                    'frame1': [5],         # orientation
                    'frame2': [10]         # orientation
                }
            },
            'frame_info': {
                'reference_frame_id': self.reference_frame_id,
                'target_frame_id': self.target_frame_id,
                'num_frames': self.num_frames
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
                f.write(f"Reference Frame: {model_data['frame_info']['reference_frame_id']}\n")
                f.write(f"Target Frame: {model_data['frame_info']['target_frame_id']}\n\n")
                f.write("Data Structure (2D):\n")
                f.write(f"  Position dims: {model_data['data_structure']['position_dims']}\n")
                f.write(f"  Velocity dims: {model_data['data_structure']['velocity_dims']}\n")
                f.write(f"  Orientation dims: {model_data['data_structure']['orientation_dims']}\n\n")
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
            fig.suptitle('TP-GMM Gait Training Data', fontsize=16)
            
            data = model_data['training_data']
            pos_dims = model_data['data_structure']['position_dims']
            vel_dims = model_data['data_structure']['velocity_dims']
            orient_dims = model_data['data_structure']['orientation_dims']
            
            # Frame 1 data
            axes[0, 0].plot(data[:, 0], data[:, pos_dims['frame1'][0]], 'b-', alpha=0.7, label='X')
            axes[0, 0].plot(data[:, 0], data[:, pos_dims['frame1'][1]], 'r-', alpha=0.7, label='Y')
            axes[0, 0].set_title('Frame 1 - Position')
            axes[0, 0].set_xlabel('Normalized Time')
            axes[0, 0].set_ylabel('Position (m)')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            axes[0, 1].plot(data[:, 0], data[:, vel_dims['frame1'][0]], 'b-', alpha=0.7, label='Vx')
            axes[0, 1].plot(data[:, 0], data[:, vel_dims['frame1'][1]], 'r-', alpha=0.7, label='Vy')
            axes[0, 1].set_title('Frame 1 - Velocity')
            axes[0, 1].set_xlabel('Normalized Time')
            axes[0, 1].set_ylabel('Velocity (m/s)')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            axes[0, 2].plot(data[:, 0], data[:, orient_dims['frame1'][0]], 'g-', alpha=0.7)
            axes[0, 2].set_title('Frame 1 - Orientation')
            axes[0, 2].set_xlabel('Normalized Time')
            axes[0, 2].set_ylabel('Orientation (rad)')
            axes[0, 2].grid(True)
            
            # Frame 2 data
            axes[1, 0].plot(data[:, 0], data[:, pos_dims['frame2'][0]], 'b-', alpha=0.7, label='X')
            axes[1, 0].plot(data[:, 0], data[:, pos_dims['frame2'][1]], 'r-', alpha=0.7, label='Y')
            axes[1, 0].set_title('Frame 2 - Position')
            axes[1, 0].set_xlabel('Normalized Time')
            axes[1, 0].set_ylabel('Position (m)')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            axes[1, 1].plot(data[:, 0], data[:, vel_dims['frame2'][0]], 'b-', alpha=0.7, label='Vx')
            axes[1, 1].plot(data[:, 0], data[:, vel_dims['frame2'][1]], 'r-', alpha=0.7, label='Vy')
            axes[1, 1].set_title('Frame 2 - Velocity')
            axes[1, 1].set_xlabel('Normalized Time')
            axes[1, 1].set_ylabel('Velocity (m/s)')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            axes[1, 2].plot(data[:, 0], data[:, orient_dims['frame2'][0]], 'g-', alpha=0.7)
            axes[1, 2].set_title('Frame 2 - Orientation')
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
            fig.suptitle('2D Gait Trajectories', fontsize=16)
            
            data = model_data['training_data']
            pos_dims = model_data['data_structure']['position_dims']
            
            # Frame 1 trajectory
            axes[0].plot(data[:, pos_dims['frame1'][0]], data[:, pos_dims['frame1'][1]], 'b-', alpha=0.7)
            axes[0].scatter(data[0, pos_dims['frame1'][0]], data[0, pos_dims['frame1'][1]], 
                          c='green', s=50, label='Start')
            axes[0].scatter(data[-1, pos_dims['frame1'][0]], data[-1, pos_dims['frame1'][1]], 
                          c='red', s=50, label='End')
            axes[0].set_title('Frame 1 - 2D Trajectory')
            axes[0].set_xlabel('X Position (m)')
            axes[0].set_ylabel('Y Position (m)')
            axes[0].legend()
            axes[0].grid(True)
            axes[0].axis('equal')
            
            # Frame 2 trajectory
            axes[1].plot(data[:, pos_dims['frame2'][0]], data[:, pos_dims['frame2'][1]], 'r-', alpha=0.7)
            axes[1].scatter(data[0, pos_dims['frame2'][0]], data[0, pos_dims['frame2'][1]], 
                          c='green', s=50, label='Start')
            axes[1].scatter(data[-1, pos_dims['frame2'][0]], data[-1, pos_dims['frame2'][1]], 
                          c='red', s=50, label='End')
            axes[1].set_title('Frame 2 - 2D Trajectory')
            axes[1].set_xlabel('X Position (m)')
            axes[1].set_ylabel('Y Position (m)')
            axes[1].legend()
            axes[1].grid(True)
            axes[1].axis('equal')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error in 2D visualization: {e}")

def main():
    """
    Example usage of the TP-GMM Gait Trainer
    """
    # Initialize trainer
    trainer = TPGMMGaitTrainer(reference_frame_id=1, target_frame_id=2)
    
    # Path to your .mat file
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
        trainer.save_tpgmm_model(model_data, 'tpgmm_gait_model.pkl')
        
        # 4. Visualize data (optional)
        print("\n=== Visualizing data ===")
        trainer.visualize_training_data(model_data)
        trainer.visualize_2d_trajectories(model_data)
        
        print("\n✓ TP-GMM gait model trained successfully!")
        print(f"  Demonstrations: {len(demonstrations)}")
        print(f"  Components: {model_data['n_components']}")
        print(f"  Dimension: {model_data['data_structure']['total_dim']}")
        
    except Exception as e:
        print(f"✗ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()