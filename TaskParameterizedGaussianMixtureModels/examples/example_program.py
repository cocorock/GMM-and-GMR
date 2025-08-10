#!/usr/bin/env python3
"""
Task Parameterized Gaussian Mixture Models (TPGMM) Example Program

This program demonstrates the use of TPGMM for trajectory analysis and generation.
Based on the TrajAir dataset: https://theairlab.org/trajair/
"""

import sys
import numpy as np
from glob import glob
from numpy import ndarray
from enum import Enum
from typing import Any, List, Literal, Union
from scipy import interpolate
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append("/".join(sys.path[0].split("/")[:-1]))

from tpgmm.utils.file_system import load_txt
from tpgmm.utils.casting import ssv_to_ndarray
from tpgmm.utils.plot.plot import plot_trajectories, plot_ellipsoids, scatter
from tpgmm.tpgmm.tpgmm import TPGMM
from tpgmm.gmr.gmr import GaussianMixtureRegression

import os
print("Current directory")
print(os.getcwd())


class DiffFilter:
    """Filter trajectories based on maximum difference threshold."""
    
    def __init__(self, threshold: float = np.exp(-2)) -> None:
        # if max diff is bigger than np.exp(-2) ~ 0.1353352832366127 -> return false
        self._threshold: float = threshold

    def __call__(self, data: ndarray) -> bool:
        if np.max(np.abs(np.diff(data, axis=0))) >= self._threshold:
            return False
        return True


class GroundFilter:
    """Filter trajectories based on minimum ground height."""
    
    def __init__(self, threshold: float = 0.1) -> None:
        self._threshold: float = threshold

    def __call__(self, data: ndarray) -> bool:
        if min(data[:, 2]) <= self._threshold:
            return True
        return False


class SamplingMode(Enum):
    """Enumeration for different sampling modes."""
    MIN = 0
    MEAN = 1
    MAX = 2


class PreProcessor:
    """Preprocess trajectories for TPGMM analysis."""
    
    def __init__(
        self,
        capture_frequency: float,
        num_samples: Union[
            Literal[SamplingMode.MAX],
            Literal[SamplingMode.MEAN],
            Literal[SamplingMode.MIN],
            int,
        ] = SamplingMode.MIN,
        frame_idx: List[int] = [0, -1],
    ) -> None:
        self.capture_frequency = capture_frequency
        self.num_samples = num_samples
        self.frame_idx = frame_idx

    def __call__(self, data: Union[List[ndarray], ndarray]) -> ndarray:
        """
        Preprocess trajectory data.
        
        Args:
            data: Shape (num_trajectories, num_points, num_features)
            
        Returns:
            ndarray: Shape (num_trajectories, num_frames, num_points, num_features)
        """
        trajectories = self.unify_length(data, self.num_samples)
        
        # Translate trajectories into respected start and end frames
        local_trajectories = self.translate_trajectories(trajectories)

        # Add capture time information
        local_trajectories = self.add_time(local_trajectories)

        return local_trajectories
    
    def unify_length(self, 
        trajectories: List[ndarray],
        sampling_mode: Union[
            Literal[SamplingMode.MIN],
            Literal[SamplingMode.MAX],
            Literal[SamplingMode.MEAN],
            int,
        ] = SamplingMode.MIN,
    ) -> ndarray:
        """
        Normalize all trajectories to the same length using interpolation.
        
        Args:
            trajectories: List of trajectory arrays
            sampling_mode: How to determine the target length
            
        Returns:
            ndarray: All trajectories interpolated to the same length
        """
        num_samples = None
        trajectory_lengths = list(map(lambda x: len(x), trajectories))
        
        if sampling_mode == SamplingMode.MIN:
            num_samples = min(trajectory_lengths)
        elif sampling_mode == SamplingMode.MEAN:
            num_samples = int(np.mean(trajectory_lengths))
        elif sampling_mode == SamplingMode.MAX:
            num_samples = max(trajectory_lengths)
        elif isinstance(sampling_mode, int):
            num_samples = sampling_mode

        unified_trajectories = []
        for trajectory in trajectories:
            # Create splines and sample points from the spline
            tck, u = interpolate.splprep(trajectory[:, :3].T, k=3, s=0)
            u = np.linspace(0, 1, num=num_samples, endpoint=True)
            unified_trajectories.append(interpolate.splev(u, tck))

        unified_trajectories = np.stack(unified_trajectories).swapaxes(-1, -2)

        return unified_trajectories

    def translate_trajectories(self, trajectories: ndarray) -> ndarray:
        """Transform trajectories to start and end reference frames."""
        # Get pick and place translation from trajectories
        start_end_translation = trajectories[:, [0, -1], :]
        means = start_end_translation.mean(0)
        print("mean start: ", means[0])
        print("mean end: ", means[1])
        
        result = np.empty((2, *trajectories.shape))
        for frame_idx in range(2):
            points = trajectories.swapaxes(0, 1) - start_end_translation[:, frame_idx] 
            result[frame_idx] = points.swapaxes(0, 1)
        
        return result

    def add_time(self, trajectories: Union[List[ndarray], ndarray]) -> Union[List[ndarray], ndarray]:
        """
        Add timestamp to every point on the trajectory.
        
        Args:
            trajectories: Shape (..., num_points, 3)
            
        Returns:
            Trajectories with time information. Shape (..., num_points, 4)
        """
        if isinstance(trajectories, ndarray):
            # Create time vector
            traj_shape = trajectories.shape
            time = np.arange(
                0,
                traj_shape[-2] * (1 / self.capture_frequency),
                1 / self.capture_frequency,
            )
            print(f"time feature between {time[0]} and {time[-1]}. {len(time)} time instances")

            # Bump up dimensionality to match trajectories
            time = time.reshape((1,) * 2 + (-1, 1))
            time = np.broadcast_to(time, (*traj_shape[:-1], 1))

            return np.concatenate([trajectories, time], axis=-1)
        else:
            raise ValueError("wrong input type. Supported is: ndarray")


def load_trajectories(data_path_pattern: str = "C:/Users/quepe/Documents/GitHub/GMM-and-GMR/TaskParameterizedGaussianMixtureModels/examples/data/7days1/processed_data/*/*.txt") -> List[ndarray]:
    """Load trajectories from text files."""
    trajectories = []
    for data_path in glob(data_path_pattern):
        data = map(lambda x: ssv_to_ndarray(x), load_txt(data_path))
        data = np.stack(list(data))[:, 2:5]
        trajectories.append(data)
    
    print(f"Loaded {len(trajectories)} trajectories")
    return trajectories


def generate_synthetic_trajectories(base_trajectory: ndarray, num_trajectories: int = 120) -> List[ndarray]:
    """Generate synthetic trajectories based on a reference trajectory."""
    trajectory_vector = np.diff(base_trajectory, axis=0)
    
    synthetic_trajectories = []
    for _ in range(num_trajectories):
        synthetic_trajectory = np.zeros_like(base_trajectory) 
        noise = np.random.normal(0, 0.01, size=synthetic_trajectory.shape)
        synthetic_trajectory[0] = base_trajectory[0] + noise[0]
        
        for traj_idx in range(1, len(base_trajectory)):
            traj_point = (synthetic_trajectory[traj_idx - 1] + 
                         trajectory_vector[traj_idx - 1] + noise[traj_idx])
            synthetic_trajectory[traj_idx] = traj_point
            
        synthetic_trajectories.append(np.stack(synthetic_trajectory))
    
    return synthetic_trajectories


def main():
    """Main function to run the TPGMM example."""
    
    # Load and filter trajectories
    trajectories = load_trajectories()
    
    # Apply filters
    diff_filter = DiffFilter()
    ground_filter = GroundFilter(0.3)
    trajectories = list(filter(ground_filter, filter(diff_filter, trajectories)))
    
    # Select a base trajectory and plot it
    base_trajectory_idx = 54
    if len(trajectories) > base_trajectory_idx:
        base_trajectory = trajectories[base_trajectory_idx]
    else:
        print(f"Warning: Only {len(trajectories)} trajectories available. Using first one.")
        base_trajectory = trajectories[0]
    
    plot_trajectories(
        title="Base Flight Trajectory",
        trajectories=base_trajectory[None],
        legend=True,
    )
    
    # Generate synthetic trajectories
    print("\nGenerating synthetic trajectories...")
    synthetic_trajectories = generate_synthetic_trajectories(base_trajectory, num_trajectories=120)
    
    plot_trajectories(
        title="Synthetic Flight Trajectories",
        trajectories=synthetic_trajectories,
        legend=False,
    )
    
    # Use synthetic trajectories for the rest of the analysis
    trajectories = synthetic_trajectories
    
    # Preprocess data
    print("\nPreprocessing trajectories...")
    num_samples = 200
    capture_freq = 100
    pre_processor = PreProcessor(capture_freq, num_samples)
    pre_processed_data = pre_processor(trajectories)

    print(f"trajectories.shape : {np.array(trajectories).shape}")
    print(f"pre_processed_data.shape : {pre_processed_data.shape}")

    plot_trajectories(trajectories=pre_processed_data[0, :, :, 0:3])
    plot_trajectories(trajectories=pre_processed_data[1, :, :, 0:3])
    
    print(f"Pre-processed data shape: {pre_processed_data.shape}")
    
    # Fit TPGMM
    print("\nFitting TPGMM...")
    tpgmm = TPGMM(6, verbose=True)
    num_trajectories = pre_processed_data.shape[1]
    concatenated_trajectories = np.reshape(
        pre_processed_data, (2, num_trajectories * num_samples, 4)
    )
    
    print(f"Concatenated trajectories shape: {concatenated_trajectories.shape}")
    tpgmm.fit(concatenated_trajectories)
    
    # Plot GMM ellipsoids from different perspectives
    print("\nPlotting GMM ellipsoids...")
        
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')

    frame_idx = 0
    returned_ax = plot_ellipsoids(
        title="GMM from start perspective", 
        means=tpgmm.means_[frame_idx, :, :3], 
        covs=tpgmm.covariances_[frame_idx, :, :3, :3], 
        legend=True
    )
  
    
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')

    frame_idx = 1
    returned_ax = plot_ellipsoids(
        title="GMM from end perspective", 
        means=tpgmm.means_[frame_idx, :, :3], 
        covs=tpgmm.covariances_[frame_idx, :, :3, :3], 
        legend=True
    )

    # plt.show()
    
    # Perform Gaussian Mixture Regression
    print("\nPerforming Gaussian Mixture Regression...")
    gmr = GaussianMixtureRegression.from_tpgmm(tpgmm, [3])
    
    # Define translation and rotation for new trajectory
    translation = np.array([
        [0.64892812, -0.00127319, 0.27268653], 
        [3.25658728, 4.49575041, 0.95660171]
    ])
    rotation_matrix = np.eye(3)[None].repeat(2, axis=0)
    
    gmr.fit(translation=translation, rotation_matrix=rotation_matrix)
    
    # Predict trajectory
    print("\nPredicting trajectory...")
    input_data = np.linspace(0, 2, 200)[:, None]
    mu, cov = gmr.predict(input_data)
    
    # Visualize reconstructed trajectory
    fig, ax = scatter(title="Reconstructed Trajectory", data=translation[:, None])
    fig, ax = plot_trajectories(trajectories=mu[None], fig=fig, ax=ax, legend=True)
    plot_ellipsoids(
        means=mu[::20], 
        covs=cov[::20], 
        fig=fig, 
        ax=ax, 
        alpha=0.3
    )
    
    plt.show()
    
    # Print distance metrics
    start_distance = np.linalg.norm(translation[0] - mu[0])
    end_distance = np.linalg.norm(translation[1] - mu[-1])
    print(f"\nDistance to desired positions:")
    print(f"  Start: {start_distance:.6f}")
    print(f"  End: {end_distance:.6f}")
    
    print("\nTPGMM analysis complete!")


if __name__ == "__main__":
    main()