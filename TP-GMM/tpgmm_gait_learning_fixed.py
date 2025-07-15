"""
Task-Parameterized Gaussian Mixture Model (TP-GMM) Implementation
================================================================

This implementation provides a complete TP-GMM system for gait trajectory learning
and adaptation using scikit-learn's GMM library.

Features:
- Handles 5D trajectory data per frame (2D position + 2D velocity + 1D orientation)
- Supports two coordinate frames (FR1: robot frame, FR2: task frame)
- Implements product of Gaussians for trajectory adaptation
- Endpoint adaptation for new target positions
- Supports multiple demonstrations from JSON array

Author: Generated for gait trajectory learning
Date: 2025-07-15
"""

import json
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.linalg import block_diag, inv
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


class TPGMM:
    """
    Task-Parameterized Gaussian Mixture Model for trajectory learning and adaptation
    """

    def __init__(self, n_components=5, n_frames=2):
        """
        Initialize TP-GMM model

        Args:
            n_components: Number of Gaussian components
            n_frames: Number of coordinate frames (default: 2 for FR1 and FR2)
        """
        self.n_components = n_components
        self.n_frames = n_frames
        self.frame_dims = 5  # 2D pos + 2D vel + 1D orientation
        self.frame_gmms = {}
        self.means = {}
        self.covariances = {}
        self.weights = None
        self.training_data_shape = None

    def _prepare_frame_data(self, trajectory_data, transformations, frame_name):
        """
        Prepare and transform trajectory data for a specific frame

        Args:
            trajectory_data: Dictionary containing trajectory arrays
            transformations: Dictionary containing transformation matrices and vectors
            frame_name: 'FR1' or 'FR2'

        Returns:
            Transformed frame data [T x 5]
        """
        if frame_name == 'FR1':
            # FR1 uses robot frame (identity transformations)
            pos_data = trajectory_data['ankle_pos_FR1']
            vel_data = trajectory_data['ankle_pos_FR1_velocity']
            ori_data = trajectory_data['ankle_orientation_FR1']
            A_matrices = transformations['ankle_A_FR1']
            b_vectors = transformations['ankle_b_FR1']
        else:  # FR2
            # FR2 uses task frame (actual transformations)
            pos_data = trajectory_data['ankle_pos_FR2']
            vel_data = trajectory_data['ankle_pos_FR2_velocity']
            ori_data = trajectory_data['ankle_orientation_FR2']
            A_matrices = transformations['ankle_A_FR2']
            b_vectors = transformations['ankle_b_FR2']

        T = len(pos_data)
        frame_data = np.zeros((T, self.frame_dims))

        for t in range(T):
            # Transform position: A * pos + b
            transformed_pos = A_matrices[t] @ pos_data[t] + b_vectors[t]
            frame_data[t, 0:2] = transformed_pos

            # Transform velocity: A * vel (no translation for velocities)
            transformed_vel = A_matrices[t] @ vel_data[t]
            frame_data[t, 2:4] = transformed_vel

            # Transform orientation: add rotation angle from A matrix
            rotation_angle = np.arctan2(A_matrices[t][1,0], A_matrices[t][0,0])
            frame_data[t, 4] = ori_data[t, 0] + rotation_angle

        return frame_data

    def fit(self, demonstrations):
        """
        Fit TP-GMM model using product of Gaussians approach

        Args:
            demonstrations: List of demonstration dictionaries containing:
                - trajectory_data: Dictionary with trajectory arrays
                - transformations: Dictionary with A matrices and b vectors

        Returns:
            Self (fitted model)
        """
        print(f"Fitting TP-GMM with {self.n_components} components...")
        print(f"Number of demonstrations: {len(demonstrations)}")

        # Prepare training data for each frame
        all_frame_data = {f'FR{i+1}': [] for i in range(self.n_frames)}

        for demo_idx, demo in enumerate(demonstrations):
            trajectory_data = demo['trajectory_data']
            transformations = demo['transformations']

            print(f"Processing demonstration {demo_idx + 1}/{len(demonstrations)}")

            # Prepare data for each frame
            for i in range(self.n_frames):
                frame_name = f'FR{i+1}'
                frame_data = self._prepare_frame_data(trajectory_data, transformations, frame_name)
                all_frame_data[frame_name].append(frame_data)

        # Concatenate all demonstrations for each frame
        self.frame_datasets = {}
        for frame_name in all_frame_data:
            self.frame_datasets[frame_name] = np.vstack(all_frame_data[frame_name])
            print(f"{frame_name} data shape: {self.frame_datasets[frame_name].shape}")

        # Store training data shape for reference
        self.training_data_shape = self.frame_datasets['FR1'].shape

        # Fit separate GMMs for each frame
        for frame_name in self.frame_datasets:
            print(f"Fitting GMM for {frame_name}...")

            gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type='full',
                random_state=42,
                max_iter=200,
                tol=1e-3
            )

            gmm.fit(self.frame_datasets[frame_name])
            self.frame_gmms[frame_name] = gmm
            self.means[frame_name] = gmm.means_
            self.covariances[frame_name] = gmm.covariances_

            print(f"{frame_name} - Log-likelihood: {gmm.score(self.frame_datasets[frame_name]):.2f}")
            print(f"{frame_name} - Converged: {gmm.converged_}")

        # Use weights from FR1 (they should be similar across frames)
        self.weights = self.frame_gmms['FR1'].weights_

        print(f"\nTP-GMM fitting completed successfully!")
        print(f"Component weights: {self.weights.round(3)}")

        return self

    def change_endpoint(self, new_endpoint, trajectory_length=200):
        """
        Generate adapted trajectory for new endpoint

        Args:
            new_endpoint: [x, y] coordinates of new target position
            trajectory_length: Length of trajectory to generate

        Returns:
            Dictionary containing adapted trajectory data
        """
        print(f"Adapting trajectory for new endpoint: {new_endpoint}")

        # Create transformation matrices for new endpoint
        new_transformations = {
            'ankle_A_FR2': np.tile(np.eye(2), (trajectory_length, 1, 1)),
            'ankle_b_FR2': np.zeros((trajectory_length, 2))
        }

        # Linear interpolation towards new endpoint
        for t in range(trajectory_length):
            alpha = t / (trajectory_length - 1)  # 0 to 1
            new_transformations['ankle_b_FR2'][t] = alpha * np.array(new_endpoint)

        return self._adapt_trajectory(new_transformations, trajectory_length)

    def adapt_trajectory(self, new_transformations_fr2, trajectory_length=200):
        """
        Generate adapted trajectory for custom transformation matrices

        Args:
            new_transformations_fr2: Dictionary containing:
                - 'ankle_A_FR2': [T x 2 x 2] transformation matrices
                - 'ankle_b_FR2': [T x 2] translation vectors
            trajectory_length: Length of trajectory to generate

        Returns:
            Dictionary containing adapted trajectory data
        """
        return self._adapt_trajectory(new_transformations_fr2, trajectory_length)

    def _adapt_trajectory(self, new_transformations_fr2, trajectory_length):
        """
        Internal method to perform trajectory adaptation using product of Gaussians

        Args:
            new_transformations_fr2: New transformation parameters for FR2
            trajectory_length: Length of trajectory to generate

        Returns:
            Dictionary with adapted trajectory data
        """
        if self.frame_gmms is None:
            raise ValueError("Model must be fitted before adaptation")

        # Initialize output trajectory
        adapted_trajectory = {
            'ankle_pos_FR1': np.zeros((trajectory_length, 2)),
            'ankle_pos_FR1_velocity': np.zeros((trajectory_length, 2)),
            'ankle_orientation_FR1': np.zeros((trajectory_length, 1)),
            'ankle_pos_FR2': np.zeros((trajectory_length, 2)),
            'ankle_pos_FR2_velocity': np.zeros((trajectory_length, 2)),
            'ankle_orientation_FR2': np.zeros((trajectory_length, 1)),
        }

        # Generate adapted trajectory point by point
        for t_idx in range(trajectory_length):
            A_fr2 = new_transformations_fr2['ankle_A_FR2'][t_idx]
            b_fr2 = new_transformations_fr2['ankle_b_FR2'][t_idx]

            # Compute product of Gaussians for this time step
            adapted_point = self._compute_product_of_gaussians(t_idx, A_fr2, b_fr2)

            # Fill adapted trajectory
            adapted_trajectory['ankle_pos_FR1'][t_idx] = adapted_point['FR1'][0:2]
            adapted_trajectory['ankle_pos_FR1_velocity'][t_idx] = adapted_point['FR1'][2:4]
            adapted_trajectory['ankle_orientation_FR1'][t_idx] = adapted_point['FR1'][4:5]

            adapted_trajectory['ankle_pos_FR2'][t_idx] = adapted_point['FR2'][0:2]
            adapted_trajectory['ankle_pos_FR2_velocity'][t_idx] = adapted_point['FR2'][2:4]
            adapted_trajectory['ankle_orientation_FR2'][t_idx] = adapted_point['FR2'][4:5]

        print("Trajectory adaptation completed!")
        return adapted_trajectory

    def _compute_product_of_gaussians(self, time_idx, A_fr2, b_fr2):
        """
        Compute product of Gaussians for adaptation at specific time step
        This implements the core TP-GMM adaptation mechanism

        Args:
            time_idx: Current time index
            A_fr2: Transformation matrix for FR2 at this time step
            b_fr2: Translation vector for FR2 at this time step

        Returns:
            Dictionary with adapted points for each frame
        """
        # Initialize result
        adapted_point = {'FR1': np.zeros(self.frame_dims), 'FR2': np.zeros(self.frame_dims)}

        # Compute weighted average of transformed Gaussian means
        weighted_sum_fr1 = np.zeros(self.frame_dims)
        weighted_sum_fr2 = np.zeros(self.frame_dims)

        for k in range(self.n_components):
            # Get original means
            mu_fr1 = self.means['FR1'][k]
            mu_fr2 = self.means['FR2'][k]

            # Transform FR2 mean according to new task parameters
            mu_fr2_adapted = mu_fr2.copy()

            # Transform position: A * pos + b
            mu_fr2_adapted[0:2] = A_fr2 @ mu_fr2[0:2] + b_fr2

            # Transform velocity: A * vel (no translation)
            mu_fr2_adapted[2:4] = A_fr2 @ mu_fr2[2:4]

            # Transform orientation: add rotation angle
            rotation_angle = np.arctan2(A_fr2[1,0], A_fr2[0,0])
            mu_fr2_adapted[4] = mu_fr2[4] + rotation_angle

            # Weight by component probability
            weight = self.weights[k]
            weighted_sum_fr1 += weight * mu_fr1
            weighted_sum_fr2 += weight * mu_fr2_adapted

        # Set adapted points
        adapted_point['FR1'] = weighted_sum_fr1
        adapted_point['FR2'] = weighted_sum_fr2

        return adapted_point

    def plot_trajectory(self, trajectory_data, title="Trajectory", frame='FR2'):
        """
        Plot trajectory data

        Args:
            trajectory_data: Dictionary containing trajectory data
            title: Plot title
            frame: Which frame to plot ('FR1' or 'FR2')
        """
        plt.figure(figsize=(10, 6))

        # Plot position trajectory
        pos_key = f'ankle_pos_{frame}'
        if pos_key in trajectory_data:
            pos_data = trajectory_data[pos_key]
            plt.subplot(1, 2, 1)
            plt.plot(pos_data[:, 0], pos_data[:, 1], 'b-', linewidth=2)
            plt.scatter(pos_data[0, 0], pos_data[0, 1], c='green', s=100, marker='o', label='Start')
            plt.scatter(pos_data[-1, 0], pos_data[-1, 1], c='red', s=100, marker='x', label='End')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.title(f'{title} - {frame} Position')
            plt.legend()
            plt.grid(True)

        # Plot orientation
        ori_key = f'ankle_orientation_{frame}'
        if ori_key in trajectory_data:
            ori_data = trajectory_data[ori_key]
            plt.subplot(1, 2, 2)
            plt.plot(ori_data, 'r-', linewidth=2)
            plt.xlabel('Time Step')
            plt.ylabel('Orientation (rad)')
            plt.title(f'{title} - {frame} Orientation')
            plt.grid(True)

        plt.tight_layout()
        plt.show()

    def get_model_info(self):
        """
        Get information about the fitted model
        
        Returns:
            Dictionary with model information
        """
        if self.frame_gmms is None:
            return {"status": "Model not fitted"}
        
        info = {
            "n_components": self.n_components,
            "n_frames": self.n_frames,
            "frame_dims": self.frame_dims,
            "training_data_shape": self.training_data_shape,
            "component_weights": self.weights.tolist(),
            "converged": {frame: self.frame_gmms[frame].converged_ for frame in self.frame_gmms.keys()},
            "log_likelihood": {frame: self.frame_gmms[frame].score(self.frame_datasets[frame]) 
                            for frame in self.frame_gmms.keys()}
        }

        return info


def fit_tpgmm_from_json(json_file_path, n_components=5, max_demonstrations=None):
    """
    Main function to load JSON data and fit TP-GMM model

    Args:
        json_file_path: Path to your JSON data file
        n_components: Number of Gaussian components
        max_demonstrations: Maximum number of demonstrations to use (None = use all)

    Returns:
        Fitted TP-GMM model
    """
    print(f"Loading data from: {json_file_path}")

    # Load JSON data
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        print("JSON data loaded successfully!")
        print(f"Found {len(data)} demonstrations in the dataset")
    except FileNotFoundError:
        print(f"Error: File {json_file_path} not found!")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_file_path}")
        return None

    # Limit demonstrations if specified
    if max_demonstrations is not None and max_demonstrations < len(data):
        data = data[:max_demonstrations]
        print(f"Using first {max_demonstrations} demonstrations")

    # Process each demonstration
    demonstrations = []

    try:
        for demo_idx, demo_data in enumerate(data):
            print(f"Processing demonstration {demo_idx + 1}/{len(data)}...")

            # Structure trajectory data
            trajectory_data = {
                'ankle_pos_FR1': np.array(demo_data['ankle_pos_FR1']),
                'ankle_pos_FR1_velocity': np.array(demo_data['ankle_pos_FR1_velocity']),
                'ankle_orientation_FR1': np.array(demo_data['ankle_orientation_FR1']),
                'ankle_pos_FR2': np.array(demo_data['ankle_pos_FR2']),
                'ankle_pos_FR2_velocity': np.array(demo_data['ankle_pos_FR2_velocity']),
                'ankle_orientation_FR2': np.array(demo_data['ankle_orientation_FR2']),
            }

            # Structure transformation data
            transformations = {
                'ankle_A_FR1': np.array(demo_data['ankle_A_FR1']),
                'ankle_b_FR1': np.array(demo_data['ankle_b_FR1']),
                'ankle_A_FR2': np.array(demo_data['ankle_A_FR2']),
                'ankle_b_FR2': np.array(demo_data['ankle_b_FR2']),
            }

            # Validate data shapes
            expected_length = len(trajectory_data['ankle_pos_FR1'])
            print(f"  Trajectory length: {expected_length}")

            # Add to demonstrations list
            demonstrations.append({
                'trajectory_data': trajectory_data,
                'transformations': transformations,
                'demonstration_index': demo_data.get('demonstration_index', demo_idx)
            })

        print(f"Successfully processed {len(demonstrations)} demonstrations!")

    except KeyError as e:
        print(f"Error: Missing key in JSON data: {e}")
        return None
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

    # Create and fit TP-GMM
    print(f"\nCreating TP-GMM with {n_components} components...")
    tpgmm = TPGMM(n_components=n_components, n_frames=2)
    tpgmm.fit(demonstrations)

    return tpgmm


def main():
    """
    Main program demonstrating TP-GMM usage
    """
    print("=" * 60)
    print("Task-Parameterized Gaussian Mixture Model (TP-GMM)")
    print("Gait Trajectory Learning and Adaptation")
    print("=" * 60)

    # Configuration
    json_file_path = "data/new_processed_gait_data#39_16.json"  # Update with your path
    n_components = 5
    max_demonstrations = 10  # Use first 5 demonstrations for faster training

    # Load and fit TP-GMM model
    print("\n1. Loading and fitting TP-GMM model...")
    tpgmm_model = fit_tpgmm_from_json(
        json_file_path, 
        n_components=n_components,
        max_demonstrations=max_demonstrations
    )

    if tpgmm_model is None:
        print("Failed to load and fit model. Please check your data file.")
        return

    # Display model information
    print("\n2. Model Information:")
    model_info = tpgmm_model.get_model_info()
    for key, value in model_info.items():
        print(f"   {key}: {value}")

    # Example 1: Change endpoint and generate adapted trajectory
    print("\n3. Adapting trajectory for new endpoint...")
    new_endpoint = [0.5, 0.3]  # New target position [x, y]
    adapted_trajectory = tpgmm_model.change_endpoint(new_endpoint)

    print(f"   New endpoint: {new_endpoint}")
    print(f"   Adapted trajectory shape: {adapted_trajectory['ankle_pos_FR2'].shape}")

    # Example 2: Custom transformation matrices
    print("\n4. Example with custom transformations...")
    trajectory_length = 200
    custom_transformations = {
        'ankle_A_FR2': np.tile(np.eye(2), (trajectory_length, 1, 1)),
        'ankle_b_FR2': np.zeros((trajectory_length, 2))
    }

    # Create a curved path to new endpoint
    for t in range(trajectory_length):
        alpha = t / (trajectory_length - 1)
        # Curved trajectory towards [0.8, 0.2]
        custom_transformations['ankle_b_FR2'][t] = [
            0.8 * alpha,
            0.2 * alpha + 0.1 * np.sin(np.pi * alpha)  # Add curve
        ]

    curved_trajectory = tpgmm_model.adapt_trajectory(custom_transformations, trajectory_length)
    print(f"   Curved trajectory generated with shape: {curved_trajectory['ankle_pos_FR2'].shape}")

    # Optional: Plot trajectories (uncomment if you want to visualize)
    # print("\n5. Plotting trajectories...")
    # tpgmm_model.plot_trajectory(adapted_trajectory, "Adapted Trajectory", frame='FR2')
    # tpgmm_model.plot_trajectory(curved_trajectory, "Curved Trajectory", frame='FR2')

    print("\n" + "=" * 60)
    print("TP-GMM demonstration completed successfully!")
    print("You can now use the adapted trajectories for robot control.")
    print("=" * 60)


if __name__ == "__main__":
    main()
