from typing import List, Union, Tuple

import numpy as np
from matplotlib.axes import Axes
from numpy import ndarray
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

from tpgmm.utils.plot.decorator import plot3D


@plot3D
def plot_trajectories(
    trajectories: ndarray, ax=None, color: str = None, alpha: float = 1
) -> Axes:
    """Plot demo trajectories in 3D space.

    Args:
        trajectories (ndarray): Demo data with shape (num_demos, demo_length, 3).
        ax (Axes, optional): Axis object in case you want to add the trajectory plots to an existing axes. Defaults to None.
        color (str, optional): Color for the trajectories. If 'auto', different colors are used for each trajectory. Default is set to None.

    Returns:
        Axes: The configured axis object.
    """
    # plot demo trajectories
    for demo_idx, demo in enumerate(trajectories):
        ax.plot(*(demo.T), label=f"Demo {demo_idx}", color=color, alpha=alpha)

    return ax


@plot3D
def scatter(
    data: Union[List[ndarray], ndarray],
    marker: str = ".",
    color: str = None,
    alpha: float = 1,
    ax=None,
) -> Axes:
    """Scatter 3D cluster data.

    Args:
        data (Union[List[ndarray], ndarray]): Data points to scatter with shape (num_clusters, num_points_per_cluster, 3).
        marker (str, optional): Marker style for scatter plot. Defaults to ".".
        ax (Axes, optional): Axis object in case you want to add the trajectory plots to an existing axes. Defaults to None.
        color (str, optional): Color for the trajectories. If 'auto', different colors are used for each trajectory. Default is set to 'auto'.

    Returns:
        Axes: The configured axis object.
    """
    if color is None or (len(color) == 1 and isinstance(color, str)):
        color = [color for _ in data]

    for frame_idx, (cluster, c) in enumerate(zip(data, color)):
        ax.scatter3D(
            *(cluster.T),
            s=50,
            label=f"frame: {frame_idx}",
            alpha=alpha,
            marker=marker,
            c=c,
        )

    return ax


@plot3D
def plot_ellipsoids(
    means: ndarray, covs: ndarray, ax=None, color: str = None, alpha: float = 1
) -> Axes:
    """Plot Gaussian ellipsoids in 3D space.

    Args:
        means (ndarray): Means of each cluster with shape (num_cluster, 3).
        cov (ndarray): Covariance matrix with shape (num_cluster, 3, 3).
        ax (Axes, optional): Axis object in case you want to add the trajectory plots to an existing axes. Defaults to None.
        color (str, optional): Color for the trajectories. If 'auto', different colors are used for each trajectory. Default is set to 'auto'.

    Returns:
        Axes: The configured axis object.
    """
    if color is None:
        color = [list(mcolors.TABLEAU_COLORS.keys())[idx % len(mcolors.TABLEAU_COLORS)] for idx in range(len(means))]
    elif len(color) == 1:
        color = [color for _ in means]
    
    for idx, (mean, cov, c) in enumerate(zip(means, covs, color)):
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # generate sphere
        u = np.linspace(0, 2 * np.pi, 10)
        v = np.linspace(0, np.pi, 10)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))

        # apply transformation to align with eigenvectors
        ellipsoid = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        ellipsoid = np.dot(ellipsoid, np.sqrt(np.diag(eigenvalues))) @ eigenvectors.T

        # shift ellipsoid with mean
        ellipsoid += mean

        # make it compliant with writeframes
        ellipsoid = ellipsoid.reshape((*x.shape, 3))
        ax.plot_wireframe(
            ellipsoid[:, :, 0],
            ellipsoid[:, :, 1],
            ellipsoid[:, :, 2],
            color=c,
            label=f"distr: {idx}",
            alpha=alpha,
        )

    return ax

def plot_ellipsoids_2d(
    means: np.ndarray, 
    covs: np.ndarray, 
    ax: plt.Axes, 
    color: str = None, 
    alpha: float = 0.5
) -> plt.Axes:
    """Plot Gaussian ellipsoids in 2D space.

    Args:
        means (np.ndarray): Means of each cluster with shape (num_cluster, 2).
        covs (np.ndarray): Covariance matrix with shape (num_cluster, 2, 2).
        ax (plt.Axes): Axis object to plot on.
        color (str, optional): Color for the ellipsoids. If None, uses a default color cycle. Defaults to None.
        alpha (float, optional): Alpha for the ellipsoids. Defaults to 0.5.

    Returns:
        plt.Axes: The configured axis object.
    """
    if color is None:
        colors = [list(mcolors.TABLEAU_COLORS.keys())[idx % len(mcolors.TABLEAU_COLORS)] for idx in range(len(means))]
    else:
        colors = [color for _ in means]

    for i, (mean, cov, c) in enumerate(zip(means, covs, colors)):
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        order = eigenvals.argsort()[::-1]
        eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]

        angle = np.degrees(np.arctan2(*eigenvecs[:, 0][::-1]))

        width, height = 2 * np.sqrt(eigenvals)
        
        ell = Ellipse(xy=mean, width=width, height=height, angle=angle, color=c, alpha=alpha)
        ax.add_artist(ell)

    return ax


## 2D versions 

def plot_ellipsoids_2d(
    means: ndarray, 
    covs: ndarray, 
    ax=None, 
    color: str = None, 
    alpha: float = 0.3,
    n_std: float = 2.0,
    fill: bool = True,
    linewidth: float = 2.0
) -> Axes:
    """Plot Gaussian ellipsoids in 2D space.

    Args:
        means (ndarray): Means of each cluster with shape (num_cluster, 2).
        covs (ndarray): Covariance matrices with shape (num_cluster, 2, 2).
        ax (Axes, optional): Axis object to plot on. If None, creates new axes.
        color (str or list, optional): Color(s) for the ellipsoids. If None, uses different colors.
        alpha (float, optional): Transparency level. Defaults to 0.3.
        n_std (float, optional): Number of standard deviations for ellipse size. Defaults to 2.0.
        fill (bool, optional): Whether to fill the ellipsoids. Defaults to True.
        linewidth (float, optional): Width of ellipse boundary. Defaults to 2.0.

    Returns:
        Axes: The configured axis object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Handle color assignment
    if color is None:
        colors = [list(mcolors.TABLEAU_COLORS.values())[idx % len(mcolors.TABLEAU_COLORS)] 
                 for idx in range(len(means))]
    elif isinstance(color, str):
        colors = [color for _ in means]
    else:
        colors = color
    
    for idx, (mean, cov, c) in enumerate(zip(means, covs, colors)):
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Calculate ellipse parameters
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width = 2 * n_std * np.sqrt(eigenvalues[0])
        height = 2 * n_std * np.sqrt(eigenvalues[1])
        
        # Create ellipse patch
        ellipse = Ellipse(
            xy=mean,
            width=width,
            height=height,
            angle=angle,
            facecolor=c if fill else 'none',
            edgecolor=c,
            alpha=alpha,
            linewidth=linewidth,
            label=f"Gaussian {idx}"
        )
        
        ax.add_patch(ellipse)
        
        # Plot center point
        ax.plot(mean[0], mean[1], 'o', color=c, markersize=6, markeredgecolor='black', markeredgewidth=1)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    return ax


def plot_ellipsoids_2d_contour(
    means: ndarray, 
    covs: ndarray, 
    ax=None, 
    color: str = None, 
    alpha: float = 0.7,
    n_std_levels: List[float] = [1.0, 2.0, 3.0],
    grid_size: int = 100
) -> Axes:
    """Plot Gaussian ellipsoids as contour lines in 2D space.

    Args:
        means (ndarray): Means of each cluster with shape (num_cluster, 2).
        covs (ndarray): Covariance matrices with shape (num_cluster, 2, 2).
        ax (Axes, optional): Axis object to plot on. If None, creates new axes.
        color (str or list, optional): Color(s) for the contours. If None, uses different colors.
        alpha (float, optional): Transparency level. Defaults to 0.7.
        n_std_levels (List[float], optional): Standard deviation levels for contours. Defaults to [1.0, 2.0, 3.0].
        grid_size (int, optional): Resolution of the contour grid. Defaults to 100.

    Returns:
        Axes: The configured axis object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Handle color assignment
    if color is None:
        colors = [list(mcolors.TABLEAU_COLORS.values())[idx % len(mcolors.TABLEAU_COLORS)] 
                 for idx in range(len(means))]
    elif isinstance(color, str):
        colors = [color for _ in means]
    else:
        colors = color
    
    # Create grid for plotting
    x_min = np.min(means[:, 0]) - 3 * np.max([np.sqrt(np.max(cov)) for cov in covs])
    x_max = np.max(means[:, 0]) + 3 * np.max([np.sqrt(np.max(cov)) for cov in covs])
    y_min = np.min(means[:, 1]) - 3 * np.max([np.sqrt(np.max(cov)) for cov in covs])
    y_max = np.max(means[:, 1]) + 3 * np.max([np.sqrt(np.max(cov)) for cov in covs])
    
    x = np.linspace(x_min, x_max, grid_size)
    y = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    for idx, (mean, cov, c) in enumerate(zip(means, covs, colors)):
        # Calculate multivariate normal PDF
        cov_inv = np.linalg.inv(cov)
        cov_det = np.linalg.det(cov)
        
        diff = pos - mean
        exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=2)
        pdf = np.exp(exponent) / (2 * np.pi * np.sqrt(cov_det))
        
        # Calculate contour levels based on standard deviations
        max_pdf = 1 / (2 * np.pi * np.sqrt(cov_det))
        levels = [max_pdf * np.exp(-0.5 * std**2) for std in n_std_levels]
        
        # Plot contours
        contour = ax.contour(X, Y, pdf, levels=levels, colors=[c], alpha=alpha, linewidths=2)
        ax.clabel(contour, inline=True, fontsize=8, fmt=f'σ={n_std_levels[0]:.0f}')
        
        # Plot center point
        ax.plot(mean[0], mean[1], 'o', color=c, markersize=8, 
               markeredgecolor='black', markeredgewidth=1, label=f"Gaussian {idx}")
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    return ax


def scatter_2d(
    data: Union[List[ndarray], ndarray],
    marker: str = ".",
    color: str = None,
    alpha: float = 1,
    s: float = 50,
    ax=None,
) -> Axes:
    """Scatter 2D cluster data.

    Args:
        data (Union[List[ndarray], ndarray]): Data points with shape (num_clusters, num_points_per_cluster, 2).
        marker (str, optional): Marker style for scatter plot. Defaults to ".".
        color (str or list, optional): Color(s) for the trajectories. If None, uses different colors.
        alpha (float, optional): Transparency level. Defaults to 1.
        s (float, optional): Size of markers. Defaults to 50.
        ax (Axes, optional): Axis object to plot on. If None, creates new axes.

    Returns:
        Axes: The configured axis object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    if color is None:
        colors = [list(mcolors.TABLEAU_COLORS.values())[idx % len(mcolors.TABLEAU_COLORS)] 
                 for idx in range(len(data))]
    elif isinstance(color, str):
        colors = [color for _ in data]
    else:
        colors = color

    for cluster_idx, (cluster, c) in enumerate(zip(data, colors)):
        ax.scatter(
            cluster[:, 0], cluster[:, 1],
            s=s,
            label=f"Cluster {cluster_idx}",
            alpha=alpha,
            marker=marker,
            c=c,
        )

    ax.grid(True, alpha=0.3)
    return ax


def plot_trajectories_2d(
    trajectories: ndarray, 
    ax=None, 
    color: str = None, 
    alpha: float = 1,
    linewidth: float = 2
) -> Axes:
    """Plot demo trajectories in 2D space.

    Args:
        trajectories (ndarray): Demo data with shape (num_demos, demo_length, 2).
        ax (Axes, optional): Axis object to plot on. If None, creates new axes.
        color (str or list, optional): Color for the trajectories. If None, uses different colors.
        alpha (float, optional): Transparency level. Defaults to 1.
        linewidth (float, optional): Width of trajectory lines. Defaults to 2.

    Returns:
        Axes: The configured axis object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    if color is None:
        colors = [list(mcolors.TABLEAU_COLORS.values())[idx % len(mcolors.TABLEAU_COLORS)] 
                 for idx in range(len(trajectories))]
    elif isinstance(color, str):
        colors = [color for _ in trajectories]
    else:
        colors = color

    for demo_idx, (demo, c) in enumerate(zip(trajectories, colors)):
        ax.plot(demo[:, 0], demo[:, 1], 
               label=f"Demo {demo_idx}", 
               color=c, 
               alpha=alpha,
               linewidth=linewidth)

    ax.grid(True, alpha=0.3)
    return ax
