import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
import warnings
import os

class TPGMMReproducer:
    """
    Improved TP-GMM reproducer with robust GMR implementation
    Handles loading a trained TP-GMM model and reproducing trajectories
    for new target frames with enhanced error handling and validation.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the reproducer by loading a pre-trained model.
        
        Args:
            model_path (str): Path to the saved .pkl model file.
        """
        print(f"Loading TP-GMM model from: {model_path}")
        try:
            self.model_data = joblib.load(model_path)
            self.gmm = self.model_data['gmm_model']
            self.data_structure = self.model_data['data_structure']
            
            # Validate model structure
            self._validate_model_structure()
            
            print("✓ Model loaded successfully.")
            print(f"  - Number of components: {self.gmm.n_components}")
            print(f"  - Original demonstrations: {len(self.model_data['individual_demos'])}")
            print(f"  - Data dimensions: {self.data_structure['total_dim']}")
            print(f"  - Training data shape: {self.model_data['training_data'].shape}")
            
        except FileNotFoundError:
            print(f"✗ Error: Model file not found at '{model_path}'")
            print("Please check the file path and ensure the model has been trained.")
            raise
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise

    def _validate_model_structure(self):
        """
        Validate that the loaded model has the expected structure
        and all required components are present.
        """
        print("Validating model structure...")
        
        # Check required top-level keys
        required_keys = ['gmm_model', 'data_structure', 'individual_demos', 'training_data']
        for key in required_keys:
            if key not in self.model_data:
                raise ValueError(f"Missing required key in model: {key}")
        
        # Check data structure keys
        ds = self.data_structure
        required_ds_keys = ['time_dim', 'fr1_dims', 'fr2_dims', 'position_dims']
        for key in required_ds_keys:
            if key not in ds:
                raise ValueError(f"Missing required data structure key: {key}")
        
        # Validate dimensions
        expected_total_dim = 11  # 1 time + 5 FR1 + 5 FR2
        if ds['total_dim'] != expected_total_dim:
            print(f"Warning: Expected total dimension {expected_total_dim}, got {ds['total_dim']}")
        
        # Check GMM model
        if not hasattr(self.gmm, 'n_components'):
            raise ValueError("Invalid GMM model: missing n_components")
        
        if not hasattr(self.gmm, 'means_') or not hasattr(self.gmm, 'covariances_'):
            raise ValueError("GMM model is not fitted (missing means_ or covariances_)")
        
        print("✓ Model structure validation passed")

    def get_adapted_trajectory(self, target_pos_fr2: np.ndarray, n_steps: int = 100, 
                              include_covariance: bool = True, verbose: bool = True) -> tuple:
        """
        Generate adapted trajectory using improved GMR
        
        Args:
            target_pos_fr2 (np.ndarray): Target position [x, y] for FR2
            n_steps (int): Number of trajectory points to generate
            include_covariance (bool): Whether to compute output covariance matrices
            verbose (bool): Whether to print detailed progress information
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - reproduced_mean: Mean trajectory for FR1 [n_steps x 5]
                - reproduced_cov: Covariance matrices [n_steps x 5 x 5] or None
        """
        if verbose:
            print(f"\nGenerating adapted trajectory for target FR2: {target_pos_fr2}")
        
        # Validate and process inputs
        target_pos_fr2 = np.array(target_pos_fr2).flatten()
        if len(target_pos_fr2) != 2:
            raise ValueError("Target position must be 2D [x, y]")
        
        if n_steps < 2:
            raise ValueError("n_steps must be at least 2")
        
        # Define dimension indices based on data structure
        time_dim = [self.data_structure['time_dim']]  # [0]
        fr2_pos_dims = self.data_structure['position_dims']['fr2']  # [6, 7]
        in_idx = time_dim + fr2_pos_dims  # [0, 6, 7] - time + FR2 position
        out_idx = self.data_structure['fr1_dims']  # [1, 2, 3, 4, 5] - all FR1 dims
        
        if verbose:
            print(f"Input dimensions: {in_idx} (time + FR2 position)")
            print(f"Output dimensions: {out_idx} (FR1 all dimensions)")
        
        # Create time vector and query points
        t = np.linspace(0, 1, n_steps).reshape(-1, 1)
        query = np.hstack([t, np.tile(target_pos_fr2, (n_steps, 1))])
        
        if verbose:
            print(f"Query shape: {query.shape} (expected: {n_steps} x 3)")
        
        # Perform Gaussian Mixture Regression
        repro_mean, repro_cov = self._gaussian_mixture_regression(
            query, in_idx, out_idx, include_covariance, verbose
        )
        
        if verbose:
            print("✓ Trajectory generation complete.")
            print(f"Output mean shape: {repro_mean.shape}")
            if repro_cov is not None:
                print(f"Output covariance shape: {repro_cov.shape}")
        
        return repro_mean, repro_cov

    def _gaussian_mixture_regression(self, query: np.ndarray, in_idx: list, 
                                   out_idx: list, include_covariance: bool = True,
                                   verbose: bool = True):
        """
        Robust implementation of Gaussian Mixture Regression
        
        Args:
            query (np.ndarray): Input query points [n_points x input_dim]
            in_idx (list): Input dimension indices
            out_idx (list): Output dimension indices
            include_covariance (bool): Whether to compute output covariance
            verbose (bool): Whether to print progress information
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (mean, covariance)
        """
        n_points, input_dim = query.shape
        output_dim = len(out_idx)
        n_components = self.gmm.n_components
        
        if verbose:
            print(f"GMR: {input_dim}D -> {output_dim}D, {n_components} components")
        
        # Extract means for input and output dimensions
        mu_in = self.gmm.means_[:, in_idx]  # [n_components x input_dim]
        mu_out = self.gmm.means_[:, out_idx]  # [n_components x output_dim]
        
        # Prepare regression matrices and conditional covariances
        regression_matrices = []
        conditional_covariances = []
        
        for k in range(n_components):
            # Extract covariance submatrices
            sigma_in = self.gmm.covariances_[k][np.ix_(in_idx, in_idx)]
            sigma_out = self.gmm.covariances_[k][np.ix_(out_idx, out_idx)]
            sigma_in_out = self.gmm.covariances_[k][np.ix_(in_idx, out_idx)]
            sigma_out_in = self.gmm.covariances_[k][np.ix_(out_idx, in_idx)]
            
            # Add regularization for numerical stability
            sigma_in_reg = sigma_in + np.eye(sigma_in.shape[0]) * 1e-6
            
            try:
                # Compute regression matrix: A = Σ_out,in * Σ_in^(-1)
                # Shape: (output_dim, input_dim)
                reg_matrix = np.linalg.solve(sigma_in_reg, sigma_in_out).T
                regression_matrices.append(reg_matrix)
                
                if include_covariance:
                    # Conditional covariance: Σ_out|in = Σ_out - Σ_out,in * Σ_in^(-1) * Σ_in,out
                    cond_cov = sigma_out - reg_matrix @ sigma_in_out
                    # Ensure positive definiteness
                    cond_cov += np.eye(cond_cov.shape[0]) * 1e-6
                    conditional_covariances.append(cond_cov)
                
            except np.linalg.LinAlgError as e:
                if verbose:
                    print(f"Warning: Singular matrix in component {k}, using stronger regularization")
                
                # Fallback with stronger regularization
                sigma_in_reg = sigma_in + np.eye(sigma_in.shape[0]) * 1e-4
                try:
                    reg_matrix = np.linalg.solve(sigma_in_reg, sigma_in_out).T
                    regression_matrices.append(reg_matrix)
                    
                    if include_covariance:
                        cond_cov = sigma_out + np.eye(sigma_out.shape[0]) * 1e-4
                        conditional_covariances.append(cond_cov)
                        
                except np.linalg.LinAlgError:
                    if verbose:
                        print(f"Error: Component {k} is severely ill-conditioned, skipping")
                    # Use identity matrix as fallback
                    reg_matrix = np.eye(output_dim, input_dim) * 0.1
                    regression_matrices.append(reg_matrix)
                    
                    if include_covariance:
                        cond_cov = np.eye(output_dim) * 1.0
                        conditional_covariances.append(cond_cov)
        
        # Compute component weights for each query point
        weights = self._compute_component_weights(query, mu_in, in_idx, verbose)
        
        # Compute weighted regression
        repro_mean = np.zeros((n_points, output_dim))
        repro_cov = None
        
        if include_covariance:
            repro_cov = np.zeros((n_points, output_dim, output_dim))
        
        # Weighted mixture of experts regression
        for k in range(n_components):
            # Conditional mean: μ_out|in = μ_out + A * (x - μ_in)
            diff = query - mu_in[k]  # [n_points x input_dim]
            cond_mean = mu_out[k] + diff @ regression_matrices[k].T  # [n_points x output_dim]
            
            # Weight and accumulate
            weight_k = weights[:, k:k+1]  # [n_points x 1]
            repro_mean += weight_k * cond_mean
            
            if include_covariance and len(conditional_covariances) > k:
                for i in range(n_points):
                    repro_cov[i] += weights[i, k] * conditional_covariances[k]
        
        return repro_mean, repro_cov

    def _compute_component_weights(self, query: np.ndarray, mu_in: np.ndarray, 
                                 in_idx: list, verbose: bool = True):
        """
        Compute component weights for GMR using robust numerical methods
        
        Args:
            query (np.ndarray): Query points [n_points x input_dim]
            mu_in (np.ndarray): Input means for all components [n_components x input_dim]
            in_idx (list): Input dimension indices
            verbose (bool): Whether to print warnings
            
        Returns:
            np.ndarray: Component weights [n_points x n_components]
        """
        n_points, input_dim = query.shape
        n_components = self.gmm.n_components
        
        # Compute log probabilities to avoid numerical underflow
        log_weights = np.zeros((n_points, n_components))
        
        for k in range(n_components):
            # Extract input covariance for component k
            sigma_in = self.gmm.covariances_[k][np.ix_(in_idx, in_idx)]
            sigma_in_reg = sigma_in + np.eye(sigma_in.shape[0]) * 1e-6
            
            try:
                # Compute log probability using multivariate normal
                mvn = multivariate_normal(mu_in[k], sigma_in_reg, allow_singular=True)
                log_prob = mvn.logpdf(query)
                
                # Add log prior
                log_prior = np.log(self.gmm.weights_[k] + 1e-12)
                log_weights[:, k] = log_prior + log_prob
                
            except Exception as e:
                if verbose:
                    print(f"Warning: Error computing weights for component {k}: {e}")
                log_weights[:, k] = -np.inf
        
        # Convert to linear weights with numerical stability
        max_log_weight = np.max(log_weights, axis=1, keepdims=True)
        
        # Handle case where all weights are -inf
        valid_mask = np.isfinite(max_log_weight).flatten()
        if not np.any(valid_mask):
            if verbose:
                print("Warning: All component weights are invalid, using uniform distribution")
            return np.ones((n_points, n_components)) / n_components
        
        weights = np.exp(log_weights - max_log_weight)
        
        # Normalize weights
        weight_sum = np.sum(weights, axis=1, keepdims=True)
        weights = weights / (weight_sum + 1e-12)
        
        # Final check for numerical issues
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            if verbose:
                print("Warning: NaN or Inf in weights, using uniform weighting")
            weights = np.ones((n_points, n_components)) / n_components
        
        return weights

    def plot_results(self, reproduced_trajectory: np.ndarray, target_pos_fr2: np.ndarray,
                    reproduced_cov: np.ndarray = None, save_plots: bool = True,
                    plot_title: str = None):
        """
        Enhanced visualization with uncertainty if available
        
        Args:
            reproduced_trajectory (np.ndarray): Reproduced mean trajectory [n_steps x 5]
            target_pos_fr2 (np.ndarray): Target position in FR2 [2]
            reproduced_cov (np.ndarray): Covariance matrices [n_steps x 5 x 5] or None
            save_plots (bool): Whether to save plots to file
            plot_title (str): Custom title for the plot
        """
        print("Plotting results...")
        
        # Create output directory if it doesn't exist
        if save_plots:
            os.makedirs('plots', exist_ok=True)
        
        # Determine subplot layout
        if reproduced_cov is not None:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            if not isinstance(axes, np.ndarray):
                axes = [axes]
        
        # Set main title
        if plot_title is None:
            plot_title = 'TP-GMM Trajectory Adaptation'
            if reproduced_cov is not None:
                plot_title += ' with Uncertainty'
        fig.suptitle(plot_title, fontsize=16)
        
        # Plot 1: FR1 Position Trajectory
        ax1 = axes[0]
        ax1.set_title('Adapted Trajectory in FR1 (Robot Leg)', fontweight='bold')
        
        # Plot original demonstrations
        for i, demo in enumerate(self.model_data['individual_demos']):
            pos_fr1 = demo[:, self.data_structure['position_dims']['fr1']]
            ax1.plot(pos_fr1[:, 0], pos_fr1[:, 1], color='gray', alpha=0.4, 
                    label='Original Demos' if i == 0 else "", linewidth=1)
        
        # Plot reproduced trajectory
        repro_pos_fr1 = reproduced_trajectory[:, 0:2]
        ax1.plot(repro_pos_fr1[:, 0], repro_pos_fr1[:, 1], color='red', linewidth=3, 
                label='Adapted Trajectory')
        ax1.scatter(repro_pos_fr1[0, 0], repro_pos_fr1[0, 1], c='red', marker='o', 
                   s=100, label='Start', zorder=5, edgecolor='black')
        ax1.scatter(repro_pos_fr1[-1, 0], repro_pos_fr1[-1, 1], c='red', marker='x', 
                   s=150, label='End', zorder=5, linewidth=3)
        
        # Add uncertainty ellipses if available
        if reproduced_cov is not None:
            self._plot_uncertainty_ellipses(ax1, repro_pos_fr1, reproduced_cov[:, 0:2, 0:2])
        
        ax1.set_xlabel('Position X')
        ax1.set_ylabel('Position Y')
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.legend()
        ax1.axis('equal')
        
        # Plot 2: FR2 Target Context
        ax2 = axes[1]
        ax2.set_title('Target in FR2 (Task Frame)', fontweight='bold')
        
        # Plot original FR2 trajectories
        for i, demo in enumerate(self.model_data['individual_demos']):
            pos_fr2 = demo[:, self.data_structure['position_dims']['fr2']]
            ax2.plot(pos_fr2[:, 0], pos_fr2[:, 1], color='gray', alpha=0.4, 
                    label='Original Demos' if i == 0 else "", linewidth=1)
        
        # Highlight target position
        ax2.scatter(target_pos_fr2[0], target_pos_fr2[1], c='blue', marker='*', s=300, 
                   edgecolor='black', linewidth=2, label='New Target Position', zorder=10)
        
        ax2.set_xlabel('Position X')
        ax2.set_ylabel('Position Y')
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.legend()
        ax2.axis('equal')
        
        # Additional plots if covariance is available
        if reproduced_cov is not None and len(axes) > 2:
            # Plot 3: Velocity trajectory
            ax3 = axes[2]
            ax3.set_title('FR1 Velocity Trajectory', fontweight='bold')
            
            repro_vel_fr1 = reproduced_trajectory[:, 2:4]
            t = np.linspace(0, 1, len(repro_vel_fr1))
            
            ax3.plot(t, repro_vel_fr1[:, 0], color='green', linewidth=2, label='Velocity X')
            ax3.plot(t, repro_vel_fr1[:, 1], color='orange', linewidth=2, label='Velocity Y')
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Velocity')
            ax3.grid(True, linestyle='--', alpha=0.3)
            ax3.legend()
            
            # Plot 4: Uncertainty evolution
            ax4 = axes[3]
            ax4.set_title('Position Uncertainty over Time', fontweight='bold')
            
            uncertainty_x = np.sqrt(reproduced_cov[:, 0, 0])
            uncertainty_y = np.sqrt(reproduced_cov[:, 1, 1])
            
            ax4.plot(t, uncertainty_x, label='X uncertainty', color='red', linewidth=2)
            ax4.plot(t, uncertainty_y, label='Y uncertainty', color='blue', linewidth=2)
            ax4.fill_between(t, uncertainty_x, alpha=0.3, color='red')
            ax4.fill_between(t, uncertainty_y, alpha=0.3, color='blue')
            
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Standard Deviation')
            ax4.grid(True, linestyle='--', alpha=0.3)
            ax4.legend()
        
        plt.tight_layout()
        
        if save_plots:
            filename = f'plots/tpgmm_adaptation_target_{target_pos_fr2[0]:.1f}_{target_pos_fr2[1]:.1f}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved: {filename}")
        
        plt.show()

    def _plot_uncertainty_ellipses(self, ax, positions, covariances, confidence=0.95, n_ellipses=10):
        """
        Plot uncertainty ellipses at selected time points
        
        Args:
            ax: Matplotlib axis
            positions: Position trajectory [n_points x 2]
            covariances: Covariance matrices [n_points x 2 x 2]
            confidence: Confidence level for ellipses
            n_ellipses: Number of ellipses to plot
        """
        from scipy.stats import chi2
        
        chi2_val = chi2.ppf(confidence, df=2)
        indices = np.linspace(0, len(positions)-1, n_ellipses, dtype=int)
        
        for i in indices:
            pos = positions[i]
            cov = covariances[i]
            
            # Compute eigenvalues and eigenvectors
            try:
                eigenvals, eigenvecs = np.linalg.eigh(cov)
                
                # Skip if covariance is degenerate
                if np.any(eigenvals <= 0):
                    continue
                    
                angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                width = 2 * np.sqrt(chi2_val * eigenvals[0])
                height = 2 * np.sqrt(chi2_val * eigenvals[1])
                
                ellipse = Ellipse(pos, width, height, angle=angle, 
                                facecolor='red', alpha=0.1, edgecolor='red', 
                                linewidth=0.5)
                ax.add_patch(ellipse)
                
            except np.linalg.LinAlgError:
                continue  # Skip problematic covariances

    def evaluate_reproduction_quality(self, target_pos_fr2: np.ndarray, n_steps: int = 100):
        """
        Evaluate the quality of trajectory reproduction with comprehensive metrics
        
        Args:
            target_pos_fr2 (np.ndarray): Target position for FR2
            n_steps (int): Number of trajectory steps
            
        Returns:
            Tuple[dict, np.ndarray, np.ndarray]: (metrics, trajectory, covariance)
        """
        print("\n=== Evaluating Reproduction Quality ===")
        
        # Generate trajectory with covariance
        repro_mean, repro_cov = self.get_adapted_trajectory(
            target_pos_fr2, n_steps, include_covariance=True, verbose=False
        )
        
        # Compute quality metrics
        metrics = {}
        
        # 1. Smoothness (acceleration-based metric)
        pos_traj = repro_mean[:, 0:2]
        if len(pos_traj) > 2:
            vel_traj = np.gradient(pos_traj, axis=0)
            accel_traj = np.gradient(vel_traj, axis=0)
            smoothness = np.mean(np.linalg.norm(accel_traj, axis=1))
            metrics['smoothness'] = smoothness
        else:
            metrics['smoothness'] = 0.0
        
        # 2. Trajectory length
        if len(pos_traj) > 1:
            distances = np.linalg.norm(np.diff(pos_traj, axis=0), axis=1)
            path_length = np.sum(distances)
            metrics['path_length'] = path_length
        else:
            metrics['path_length'] = 0.0
        
        # 3. Average uncertainty (if available)
        if repro_cov is not None:
            avg_uncertainty = np.mean([np.trace(cov[0:2, 0:2]) for cov in repro_cov])
            metrics['avg_uncertainty'] = avg_uncertainty
        
        # 4. Consistency with demonstrations
        demo_positions = []
        for demo in self.model_data['individual_demos']:
            demo_pos = demo[:, self.data_structure['position_dims']['fr1']]
            if len(demo_pos) > 0:
                demo_positions.append(demo_pos)
        
        # Find closest demonstration and compute distance
        min_distance = float('inf')
        if demo_positions:
            for demo_pos in demo_positions:
                distance = self._compute_trajectory_distance(pos_traj, demo_pos)
                min_distance = min(min_distance, distance)
        
        metrics['min_demo_distance'] = min_distance if min_distance != float('inf') else 0.0
        
        # 5. Endpoint deviation
        if len(pos_traj) > 1:
            endpoint_distance = np.linalg.norm(pos_traj[-1] - pos_traj[0])
            metrics['endpoint_distance'] = endpoint_distance
        else:
            metrics['endpoint_distance'] = 0.0
        
        # Print results
        print("Quality Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics, repro_mean, repro_cov

    def _compute_trajectory_distance(self, traj1, traj2):
        """
        Compute distance between two trajectories using dynamic time alignment
        
        Args:
            traj1, traj2: Trajectory arrays [n_points x 2]
            
        Returns:
            float: Average distance between trajectories
        """
        # Simple resampling approach (could be improved with DTW)
        n = min(len(traj1), len(traj2), 50)  # Limit for efficiency
        if n < 2:
            return float('inf')
        
        # Resample both trajectories to same length
        indices1 = np.linspace(0, len(traj1)-1, n, dtype=int)
        indices2 = np.linspace(0, len(traj2)-1, n, dtype=int)
        
        t1_resampled = traj1[indices1]
        t2_resampled = traj2[indices2]
        
        return np.mean(np.linalg.norm(t1_resampled - t2_resampled, axis=1))


def run_comprehensive_tests(model_path: str):
    """
    Run comprehensive tests with multiple targets and analysis
    
    Args:
        model_path (str): Path to the trained model file
    """
    print("=" * 60)
    print("COMPREHENSIVE TP-GMM TESTING")
    print("=" * 60)
    
    # Test targets covering different regions
    test_targets = [
        np.array([1.0, 1.0]),    # Positive quadrant
        np.array([0.5, -0.5]),   # Mixed signs
        np.array([-1.0, 0.0]),   # Negative x-axis
        np.array([0.0, 1.5]),    # Positive y-axis
        np.array([-0.5, -1.0]),  # Negative quadrant
    ]
    
    trajectory_steps = 100
    all_metrics = []
    
    try:
        # Initialize reproducer
        reproducer = TPGMMReproducer(model_path=model_path)
        
        # Test each target
        for i, target in enumerate(test_targets):
            print(f"\n{'='*50}")
            print(f"Testing Target {i+1}/{len(test_targets)}: {target}")
            print(f"{'='*50}")
            
            # Evaluate reproduction quality
            metrics, repro_mean, repro_cov = reproducer.evaluate_reproduction_quality(
                target_pos_fr2=target, n_steps=trajectory_steps
            )
            
            # Store metrics for comparison
            metrics['target'] = target.copy()
            all_metrics.append(metrics)
            
            # Plot results
            reproducer.plot_results(
                reproduced_trajectory=repro_mean,
                target_pos_fr2=target,
                reproduced_cov=repro_cov,
                plot_title=f'Target {i+1}: [{target[0]:.1f}, {target[1]:.1f}]'
            )
            
            print(f"✓ Target {i+1} completed successfully")
        
        # Print summary comparison
        print(f"\n{'='*60}")
        print("SUMMARY COMPARISON")
        print(f"{'='*60}")
        
        print(f"{'Target':<15} {'Smoothness':<12} {'Path Len':<10} {'Demo Dist':<10} {'Uncertainty':<12}")
        print("-" * 60)
        
        for i, metrics in enumerate(all_metrics):
            target_str = f"[{metrics['target'][0]:.1f},{metrics['target'][1]:.1f}]"
            smooth = metrics.get('smoothness', 0)
            path_len = metrics.get('path_length', 0)
            demo_dist = metrics.get('min_demo_distance', 0)
            uncertainty = metrics.get('avg_uncertainty', 0)
            
            print(f"{target_str:<15} {smooth:<12.4f} {path_len:<10.4f} {demo_dist:<10.4f} {uncertainty:<12.4f}")
        
        print("\n✓ All comprehensive tests completed successfully!")
        
        return all_metrics
        
    except Exception as e:
        print(f"\n✗ Error in testing pipeline: {e}")
        import traceback
        traceback.print_exc()
        return []


def interactive_target_selection(reproducer):
    """
    Interactive tool for selecting targets by clicking on a plot
    
    Args:
        reproducer: TPGMMReproducer instance
    """
    print("\n=== Interactive Target Selection ===")
    print("Click on the plot to test different target positions")
    print("Close the plot window when done")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title('Interactive Target Selection\n(Click to test targets)', fontsize=14, fontweight='bold')
    
    # Plot original demonstrations in FR2
    colors = plt.cm.tab10(np.linspace(0, 1, len(reproducer.model_data['individual_demos'])))
    for i, demo in enumerate(reproducer.model_data['individual_demos']):
        pos_fr2 = demo[:, reproducer.data_structure['position_dims']['fr2']]
        ax.plot(pos_fr2[:, 0], pos_fr2[:, 1], color=colors[i], alpha=0.6, 
               linewidth=2, label=f'Demo {i+1}' if i < 5 else "")
    
    ax.set_xlabel('FR2 Position X')
    ax.set_ylabel('FR2 Position Y')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.axis('equal')
    
    # Store clicked targets
    clicked_targets = []
    
    def onclick(event):
        if event.inaxes == ax and event.button == 1:  # Left click
            target = np.array([event.xdata, event.ydata])
            print(f"\nTesting target: [{target[0]:.2f}, {target[1]:.2f}]")
            
            try:
                # Generate trajectory
                traj, cov = reproducer.get_adapted_trajectory(target, verbose=False)
                
                # Plot target point
                ax.scatter(target[0], target[1], c='red', s=150, marker='*', 
                          edgecolor='black', linewidth=1, zorder=10,
                          label=f'Target {len(clicked_targets)+1}')
                
                # Add text annotation
                ax.annotate(f'T{len(clicked_targets)+1}', 
                           (target[0], target[1]), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=10, fontweight='bold')
                
                clicked_targets.append(target)
                plt.draw()
                
                print(f"✓ Target {len(clicked_targets)} processed")
                
            except Exception as e:
                print(f"✗ Error processing target: {e}")
    
    # Connect click event
    fig.canvas.mpl_connect('button_press_event', onclick)
    
    plt.tight_layout()
    plt.show()
    
    return clicked_targets


def analyze_workspace_coverage(reproducer, grid_resolution=10):
    """
    Analyze reproduction quality across the workspace
    
    Args:
        reproducer: TPGMMReproducer instance
        grid_resolution: Number of points per dimension for grid
    """
    print(f"\n=== Workspace Coverage Analysis ===")
    print(f"Testing {grid_resolution}x{grid_resolution} grid of targets")
    
    # Determine workspace bounds from demonstrations
    all_fr2_positions = []
    for demo in reproducer.model_data['individual_demos']:
        pos_fr2 = demo[:, reproducer.data_structure['position_dims']['fr2']]
        all_fr2_positions.append(pos_fr2)
    
    all_fr2_positions = np.vstack(all_fr2_positions)
    
    # Expand bounds slightly
    margin = 0.2
    x_min, x_max = all_fr2_positions[:, 0].min() - margin, all_fr2_positions[:, 0].max() + margin
    y_min, y_max = all_fr2_positions[:, 1].min() - margin, all_fr2_positions[:, 1].max() + margin
    
    print(f"Workspace bounds: X[{x_min:.2f}, {x_max:.2f}], Y[{y_min:.2f}, {y_max:.2f}]")
    
    # Create grid
    x_grid = np.linspace(x_min, x_max, grid_resolution)
    y_grid = np.linspace(y_min, y_max, grid_resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Test each grid point
    smoothness_map = np.zeros_like(X)
    uncertainty_map = np.zeros_like(X)
    success_map = np.zeros_like(X)
    
    total_points = grid_resolution * grid_resolution
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            target = np.array([X[i, j], Y[i, j]])
            
            try:
                metrics, _, _ = reproducer.evaluate_reproduction_quality(target, n_steps=50)
                smoothness_map[i, j] = metrics.get('smoothness', 0)
                uncertainty_map[i, j] = metrics.get('avg_uncertainty', 0)
                success_map[i, j] = 1
                
            except Exception:
                smoothness_map[i, j] = np.nan
                uncertainty_map[i, j] = np.nan
                success_map[i, j] = 0
            
            # Progress indicator
            progress = ((i * grid_resolution + j + 1) / total_points) * 100
            if (i * grid_resolution + j + 1) % max(1, total_points // 10) == 0:
                print(f"Progress: {progress:.0f}%")
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Success rate map
    im1 = axes[0].imshow(success_map, extent=[x_min, x_max, y_min, y_max], 
                        origin='lower', cmap='RdYlGn', vmin=0, vmax=1)
    axes[0].set_title('Success Rate')
    axes[0].set_xlabel('X Position')
    axes[0].set_ylabel('Y Position')
    plt.colorbar(im1, ax=axes[0])
    
    # Smoothness map
    smoothness_masked = np.ma.masked_where(success_map == 0, smoothness_map)
    im2 = axes[1].imshow(smoothness_masked, extent=[x_min, x_max, y_min, y_max], 
                        origin='lower', cmap='viridis_r')
    axes[1].set_title('Smoothness (lower is better)')
    axes[1].set_xlabel('X Position')
    axes[1].set_ylabel('Y Position')
    plt.colorbar(im2, ax=axes[1])
    
    # Uncertainty map
    uncertainty_masked = np.ma.masked_where(success_map == 0, uncertainty_map)
    im3 = axes[2].imshow(uncertainty_masked, extent=[x_min, x_max, y_min, y_max], 
                        origin='lower', cmap='plasma')
    axes[2].set_title('Uncertainty')
    axes[2].set_xlabel('X Position')
    axes[2].set_ylabel('Y Position')
    plt.colorbar(im3, ax=axes[2])
    
    # Overlay demonstration points
    for ax in axes:
        ax.scatter(all_fr2_positions[:, 0], all_fr2_positions[:, 1], 
                  c='white', s=2, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/workspace_coverage_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    success_rate = np.mean(success_map)
    print(f"✓ Workspace analysis complete")
    print(f"Overall success rate: {success_rate:.1%}")
    
    return X, Y, smoothness_map, uncertainty_map, success_map


def main():
    """
    Main execution function with multiple testing modes
    """
    # Configuration
    especific_path = '#39_16'
    model_file = f'data/tpgmm_gait_model{especific_path}.pkl'
    
    print("TP-GMM Trajectory Reproduction Testing")
    print("=" * 50)
    
    # Check if model file exists
    if not os.path.exists(model_file):
        print(f"✗ Model file not found: {model_file}")
        print("Please ensure the model has been trained and saved.")
        return
    
    try:
        # Initialize reproducer
        reproducer = TPGMMReproducer(model_path=model_file)
        
        # Mode selection
        print("\nSelect testing mode:")
        print("1. Quick test with predefined targets")
        print("2. Comprehensive test suite")
        print("3. Interactive target selection")
        print("4. Workspace coverage analysis")
        print("5. All tests")
        
        # For automated testing, run comprehensive suite
        mode = "1"  # Change this to test different modes
        
        if mode == "1":
            # Quick test
            print("\n--- Quick Test ---")
            target = np.array([0.1, 0.1])
            metrics, traj, cov = reproducer.evaluate_reproduction_quality(target)
            reproducer.plot_results(traj, target, cov)
            
        elif mode == "2":
            # Comprehensive test suite
            print("\n--- Comprehensive Test Suite ---")
            all_metrics = run_comprehensive_tests(model_file)
            
        elif mode == "3":
            # Interactive selection
            print("\n--- Interactive Target Selection ---")
            clicked_targets = interactive_target_selection(reproducer)
            print(f"Tested {len(clicked_targets)} interactive targets")
            
        elif mode == "4":
            # Workspace analysis
            print("\n--- Workspace Coverage Analysis ---")
            analyze_workspace_coverage(reproducer, grid_resolution=8)
            
        elif mode == "5":
            # All tests
            print("\n--- Running All Tests ---")
            
            # Quick test
            print("\n1/4: Quick Test")
            target = np.array([0.5, 0.5])
            metrics, traj, cov = reproducer.evaluate_reproduction_quality(target)
            reproducer.plot_results(traj, target, cov)
            
            # Comprehensive tests
            print("\n2/4: Comprehensive Test Suite")
            all_metrics = run_comprehensive_tests(model_file)
            
            # Workspace analysis
            print("\n3/4: Workspace Coverage Analysis")
            analyze_workspace_coverage(reproducer, grid_resolution=6)
            
            print("\n4/4: Interactive mode available")
            print("Run with mode='3' for interactive target selection")
        
        else:
            print("Invalid mode selected, running comprehensive tests")
            all_metrics = run_comprehensive_tests(model_file)
        
        print(f"\n{'='*60}")
        print("TESTING COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print("Check the 'plots/' directory for saved visualizations")
        
    except Exception as e:
        print(f"\n✗ Error in main execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()