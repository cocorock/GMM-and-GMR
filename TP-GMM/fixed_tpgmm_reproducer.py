import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
import warnings
import os

class ProperTPGMMReproducer:
    """
    PROPER TP-GMM reproducer that understands the true nature of Task-Parameterized GMM
    
    Key insight: TP-GMM should generate a COORDINATED trajectory between frames,
    not just map a fixed target to FR1.
    """
    
    def __init__(self, model_path: str):
        """Initialize with model validation"""
        print(f"Loading TP-GMM model from: {model_path}")
        try:
            self.model_data = joblib.load(model_path)
            self.gmm = self.model_data['gmm_model']
            self.data_structure = self.model_data['data_structure']
            
            self._validate_and_analyze_model()
            
            print("✓ Model loaded and analyzed successfully.")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise

    def _validate_and_analyze_model(self):
        """Validate model and analyze the actual data distribution"""
        print("\n=== Model Analysis ===")
        
        # Basic validation
        required_keys = ['gmm_model', 'data_structure', 'individual_demos', 'training_data']
        for key in required_keys:
            if key not in self.model_data:
                raise ValueError(f"Missing required key: {key}")
        
        # Analyze training data
        training_data = self.model_data['training_data']
        print(f"Training data shape: {training_data.shape}")
        print(f"Expected: [n_points, 11] = [n_points, time + FR1(5D) + FR2(5D)]")
        
        # Analyze data ranges and distributions
        print("\n--- Data Range Analysis ---")
        for i, label in enumerate(['Time', 'FR1_pos_x', 'FR1_pos_y', 'FR1_vel_x', 'FR1_vel_y', 'FR1_orient',
                                  'FR2_pos_x', 'FR2_pos_y', 'FR2_vel_x', 'FR2_vel_y', 'FR2_orient']):
            if i < training_data.shape[1]:
                data_col = training_data[:, i]
                print(f"{label:12s}: [{np.min(data_col):8.3f}, {np.max(data_col):8.3f}] "
                      f"mean={np.mean(data_col):7.3f} std={np.std(data_col):7.3f}")
        
        # Analyze correlations between FR1 and FR2
        print("\n--- Frame Correlation Analysis ---")
        fr1_pos = training_data[:, 1:3]  # FR1 position
        fr2_pos = training_data[:, 6:8]  # FR2 position
        
        # Compute correlation between FR1 and FR2 positions
        corr_xx = np.corrcoef(fr1_pos[:, 0], fr2_pos[:, 0])[0, 1]
        corr_yy = np.corrcoef(fr1_pos[:, 1], fr2_pos[:, 1])[0, 1]
        
        print(f"FR1_x vs FR2_x correlation: {corr_xx:.3f}")
        print(f"FR1_y vs FR2_y correlation: {corr_yy:.3f}")
        
        # Check if demonstrations show coordinated movement
        print("\n--- Demonstration Coordination Check ---")
        for i, demo in enumerate(self.model_data['individual_demos'][:3]):  # Check first 3
            fr1_range = np.ptp(demo[:, 1:3], axis=0)  # FR1 position range
            fr2_range = np.ptp(demo[:, 6:8], axis=0)  # FR2 position range
            print(f"Demo {i}: FR1 range={fr1_range}, FR2 range={fr2_range}")
        
        print("✓ Model analysis complete")

    def approach_1_generate_coordinated_trajectory(self, target_fr2_end: np.ndarray, n_steps: int = 100):
        """
        APPROACH 1: Generate coordinated trajectory where FR2 moves toward target
        """
        print(f"\n=== Approach 1: Coordinated Movement ===")
        print(f"Target FR2 endpoint: {target_fr2_end}")
        
        # Step 1: Generate a realistic FR2 trajectory toward the target
        fr2_trajectory = self._generate_fr2_trajectory_to_target(target_fr2_end, n_steps)
        
        # Step 2: Use GMR to predict FR1 given the FR2 trajectory
        fr1_trajectory = self._predict_fr1_from_fr2_trajectory(fr2_trajectory)
        
        return fr1_trajectory, fr2_trajectory

    def _generate_fr2_trajectory_to_target(self, target_end: np.ndarray, n_steps: int):
        """
        Generate a realistic FR2 trajectory that moves toward the target
        based on the patterns observed in training data
        """
        print("Generating FR2 trajectory to target...")
        
        # Analyze typical FR2 starting positions and movements from demos
        demo_starts = []
        demo_ends = []
        demo_paths = []
        
        for demo in self.model_data['individual_demos']:
            fr2_pos = demo[:, self.data_structure['position_dims']['fr2']]
            if len(fr2_pos) > 0:
                demo_starts.append(fr2_pos[0])
                demo_ends.append(fr2_pos[-1])
                demo_paths.append(fr2_pos)
        
        if not demo_starts:
            raise ValueError("No demonstration data found")
        
        demo_starts = np.array(demo_starts)
        demo_ends = np.array(demo_ends)
        
        # Choose a typical starting position (mean of demonstrations)
        start_pos = np.mean(demo_starts, axis=0)
        print(f"FR2 start position: {start_pos}")
        print(f"FR2 target position: {target_end}")
        
        # Create smooth trajectory from start to target
        t = np.linspace(0, 1, n_steps)
        
        # Linear interpolation for position (could be made more sophisticated)
        trajectory_pos = np.outer(1-t, start_pos) + np.outer(t, target_end)
        
        # Compute velocities (simple finite differences)
        trajectory_vel = np.zeros_like(trajectory_pos)
        if n_steps > 1:
            trajectory_vel[1:] = np.diff(trajectory_pos, axis=0) * n_steps  # Scale by time
            trajectory_vel[0] = trajectory_vel[1]  # Handle first point
        
        # Simple orientation (could be improved)
        # For now, assume orientation follows movement direction
        trajectory_orient = np.zeros(n_steps)
        for i in range(1, n_steps):
            dx, dy = trajectory_vel[i]
            trajectory_orient[i] = np.arctan2(dy, dx)
        trajectory_orient[0] = trajectory_orient[1]
        
        # Combine into full FR2 trajectory
        fr2_trajectory = np.column_stack([
            t.reshape(-1, 1),           # time
            trajectory_pos,             # position x, y
            trajectory_vel,             # velocity x, y  
            trajectory_orient.reshape(-1, 1)  # orientation
        ])
        
        print(f"Generated FR2 trajectory shape: {fr2_trajectory.shape}")
        return fr2_trajectory

    def _predict_fr1_from_fr2_trajectory(self, fr2_trajectory):
        """
        Use GMR to predict FR1 trajectory given complete FR2 trajectory
        """
        print("Predicting FR1 from FR2 trajectory...")
        
        n_steps = len(fr2_trajectory)
        
        # Define dimensions for this approach
        time_dim = [0]  # time
        fr2_dims = list(range(6, 11))  # FR2 full state [6,7,8,9,10]
        fr1_dims = list(range(1, 6))   # FR1 full state [1,2,3,4,5]
        
        in_idx = time_dim + fr2_dims  # [0,6,7,8,9,10] = 6D input
        out_idx = fr1_dims            # [1,2,3,4,5] = 5D output
        
        print(f"Input dimensions: {in_idx} (time + full FR2)")
        print(f"Output dimensions: {out_idx} (full FR1)")
        
        # Create query matrix: [time, FR2_full_state]
        query = np.column_stack([
            fr2_trajectory[:, 0],  # time
            fr2_trajectory[:, 1:6]  # FR2 full state [pos_x, pos_y, vel_x, vel_y, orient]
        ])
        
        print(f"Query shape: {query.shape}")
        
        # Apply GMR
        fr1_trajectory, _ = self._apply_gmr(query, in_idx, out_idx)
        
        return fr1_trajectory

    def approach_2_time_based_adaptation(self, target_fr2_pos: np.ndarray, n_steps: int = 100):
        """
        APPROACH 2: Traditional time-based GMR but with better query construction
        """
        print(f"\n=== Approach 2: Time-Based Adaptation ===")
        print(f"Target FR2 position: {target_fr2_pos}")
        
        # Find the most appropriate demonstration pattern
        best_demo_idx = self._find_closest_demonstration(target_fr2_pos)
        reference_demo = self.model_data['individual_demos'][best_demo_idx]
        
        print(f"Using demonstration {best_demo_idx} as reference")
        
        # Extract reference FR2 trajectory pattern
        ref_fr2 = reference_demo[:, self.data_structure['position_dims']['fr2']]
        ref_fr2_vel = reference_demo[:, self.data_structure['velocity_dims']['fr2']]
        ref_fr2_orient = reference_demo[:, self.data_structure['orientation_dims']['fr2']]
        
        # Scale/translate the reference to reach our target
        adapted_fr2 = self._adapt_reference_trajectory(
            ref_fr2, ref_fr2_vel, ref_fr2_orient, target_fr2_pos, n_steps
        )
        
        # Use GMR with the adapted FR2 trajectory
        time_vec = np.linspace(0, 1, n_steps).reshape(-1, 1)
        query = np.column_stack([time_vec, adapted_fr2])
        
        # GMR: time + FR2_full -> FR1_full
        in_idx = [0] + list(range(6, 11))  # time + FR2 full
        out_idx = list(range(1, 6))        # FR1 full
        
        fr1_trajectory, covariance = self._apply_gmr(query, in_idx, out_idx, include_cov=True)
        
        return fr1_trajectory, adapted_fr2, covariance

    def _find_closest_demonstration(self, target_pos):
        """Find demonstration with FR2 endpoint closest to target"""
        min_dist = float('inf')
        best_idx = 0
        
        for i, demo in enumerate(self.model_data['individual_demos']):
            fr2_pos = demo[:, self.data_structure['position_dims']['fr2']]
            if len(fr2_pos) > 0:
                endpoint = fr2_pos[-1]
                dist = np.linalg.norm(endpoint - target_pos)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
        
        print(f"Closest demo endpoint distance: {min_dist:.3f}")
        return best_idx

    def _adapt_reference_trajectory(self, ref_pos, ref_vel, ref_orient, target_end, n_steps):
        """Adapt reference trajectory to reach target"""
        
        # Resample reference to desired length
        if len(ref_pos) != n_steps:
            t_ref = np.linspace(0, 1, len(ref_pos))
            t_new = np.linspace(0, 1, n_steps)
            
            ref_pos_new = np.zeros((n_steps, 2))
            ref_vel_new = np.zeros((n_steps, 2))
            ref_orient_new = np.zeros(n_steps)
            
            for i in range(2):
                ref_pos_new[:, i] = np.interp(t_new, t_ref, ref_pos[:, i])
                ref_vel_new[:, i] = np.interp(t_new, t_ref, ref_vel[:, i])
            ref_orient_new = np.interp(t_new, t_ref, ref_orient.flatten())
            
            ref_pos = ref_pos_new
            ref_vel = ref_vel_new
            ref_orient = ref_orient_new
        
        # Scale and translate to reach target
        ref_start = ref_pos[0]
        ref_end = ref_pos[-1]
        
        # Assume we start from a typical position
        demo_starts = [demo[0, self.data_structure['position_dims']['fr2']] 
                      for demo in self.model_data['individual_demos']]
        typical_start = np.mean(demo_starts, axis=0)
        
        # Linear transformation to map ref_start->typical_start and ref_end->target_end
        if np.linalg.norm(ref_end - ref_start) > 1e-6:
            scale = np.linalg.norm(target_end - typical_start) / np.linalg.norm(ref_end - ref_start)
            
            # Apply transformation
            adapted_pos = typical_start + (ref_pos - ref_start) * scale
            adapted_vel = ref_vel * scale  # Scale velocities too
            adapted_orient = ref_orient  # Keep orientation pattern
        else:
            # Reference doesn't move, create simple trajectory to target
            t = np.linspace(0, 1, n_steps)
            adapted_pos = np.outer(1-t, typical_start) + np.outer(t, target_end)
            adapted_vel = np.zeros_like(adapted_pos)
            adapted_orient = np.zeros(n_steps)
        
        # Combine adapted FR2 trajectory
        adapted_fr2 = np.column_stack([adapted_pos, adapted_vel, adapted_orient])
        
        return adapted_fr2

    def _apply_gmr(self, query, in_idx, out_idx, include_cov=False):
        """Apply Gaussian Mixture Regression with proper implementation"""
        
        n_points, input_dim = query.shape
        output_dim = len(out_idx)
        n_components = self.gmm.n_components
        
        print(f"GMR: {input_dim}D -> {output_dim}D using {n_components} components")
        
        # Extract component parameters
        mu_in = self.gmm.means_[:, in_idx]
        mu_out = self.gmm.means_[:, out_idx]
        
        # Compute regression matrices
        regression_matrices = []
        cond_covariances = []
        
        for k in range(n_components):
            sigma_in = self.gmm.covariances_[k][np.ix_(in_idx, in_idx)]
            sigma_out = self.gmm.covariances_[k][np.ix_(out_idx, out_idx)]
            sigma_out_in = self.gmm.covariances_[k][np.ix_(out_idx, in_idx)]
            sigma_in_out = self.gmm.covariances_[k][np.ix_(in_idx, out_idx)]
            
            # Regularize
            sigma_in_reg = sigma_in + np.eye(len(in_idx)) * 1e-6
            
            try:
                # Regression matrix: A = Σ_out,in * Σ_in^(-1)
                A = sigma_out_in @ np.linalg.inv(sigma_in_reg)
                regression_matrices.append(A)
                
                if include_cov:
                    cond_cov = sigma_out - A @ sigma_in_out
                    cond_cov += np.eye(len(out_idx)) * 1e-6
                    cond_covariances.append(cond_cov)
                    
            except np.linalg.LinAlgError:
                print(f"Warning: Regularizing component {k}")
                A = np.zeros((len(out_idx), len(in_idx)))
                regression_matrices.append(A)
                
                if include_cov:
                    cond_covariances.append(np.eye(len(out_idx)))
        
        # Compute weights
        weights = self._compute_gmr_weights(query, mu_in, in_idx)
        
        # Weighted regression
        output_mean = np.zeros((n_points, output_dim))
        output_cov = np.zeros((n_points, output_dim, output_dim)) if include_cov else None
        
        for k in range(n_components):
            # Conditional mean
            diff = query - mu_in[k]
            cond_mean = mu_out[k] + (regression_matrices[k] @ diff.T).T
            
            # Accumulate weighted result
            weights_k = weights[:, k:k+1]
            output_mean += weights_k * cond_mean
            
            if include_cov:
                for i in range(n_points):
                    output_cov[i] += weights[i, k] * cond_covariances[k]
        
        return output_mean, output_cov

    def _compute_gmr_weights(self, query, mu_in, in_idx):
        """Compute GMR weights with numerical stability"""
        n_points, _ = query.shape
        n_components = self.gmm.n_components
        
        log_weights = np.zeros((n_points, n_components))
        
        for k in range(n_components):
            sigma_in = self.gmm.covariances_[k][np.ix_(in_idx, in_idx)]
            sigma_in_reg = sigma_in + np.eye(len(in_idx)) * 1e-6
            
            try:
                mvn = multivariate_normal(mu_in[k], sigma_in_reg, allow_singular=True)
                log_prob = mvn.logpdf(query)
                log_prior = np.log(self.gmm.weights_[k] + 1e-12)
                log_weights[:, k] = log_prior + log_prob
            except:
                log_weights[:, k] = -np.inf
        
        # Normalize
        max_log = np.max(log_weights, axis=1, keepdims=True)
        weights = np.exp(log_weights - max_log)
        weight_sums = np.sum(weights, axis=1, keepdims=True)
        weights = weights / (weight_sums + 1e-12)
        
        # Handle edge cases
        if np.any(np.isnan(weights)):
            weights = np.ones((n_points, n_components)) / n_components
        
        return weights

    def compare_approaches(self, target_pos_fr2, n_steps=100):
        """Compare different approaches side by side"""
        print(f"\n{'='*60}")
        print(f"COMPARING APPROACHES FOR TARGET: {target_pos_fr2}")
        print(f"{'='*60}")
        
        results = {}
        
        try:
            # Approach 1: Coordinated movement
            print("\n--- Running Approach 1 ---")
            fr1_traj_1, fr2_traj_1 = self.approach_1_generate_coordinated_trajectory(target_pos_fr2, n_steps)
            results['approach_1'] = {
                'fr1_trajectory': fr1_traj_1,
                'fr2_trajectory': fr2_traj_1,
                'method': 'Coordinated Movement'
            }
            
        except Exception as e:
            print(f"Approach 1 failed: {e}")
            results['approach_1'] = None
        
        try:
            # Approach 2: Time-based adaptation
            print("\n--- Running Approach 2 ---")
            fr1_traj_2, fr2_traj_2, cov_2 = self.approach_2_time_based_adaptation(target_pos_fr2, n_steps)
            results['approach_2'] = {
                'fr1_trajectory': fr1_traj_2,
                'fr2_trajectory': fr2_traj_2,
                'covariance': cov_2,
                'method': 'Reference Adaptation'
            }
            
        except Exception as e:
            print(f"Approach 2 failed: {e}")
            results['approach_2'] = None
        
        # Visualize comparison
        self._plot_approach_comparison(results, target_pos_fr2)
        
        return results

    def _plot_approach_comparison(self, results, target_pos):
        """Plot comparison of different approaches"""
        
        n_approaches = sum(1 for r in results.values() if r is not None)
        if n_approaches == 0:
            print("No successful approaches to plot")
            return
        
        fig, axes = plt.subplots(2, n_approaches, figsize=(6*n_approaches, 10))
        if n_approaches == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle(f'TP-GMM Approach Comparison\nTarget: [{target_pos[0]:.2f}, {target_pos[1]:.2f}]', 
                     fontsize=16)
        
        approach_idx = 0
        colors = ['red', 'blue', 'green', 'orange']
        
        for approach_name, result in results.items():
            if result is None:
                continue
                
            # Plot FR1 trajectories
            ax1 = axes[0, approach_idx]
            ax1.set_title(f'{result["method"]}\nFR1 Trajectory', fontweight='bold')
            
            # Original demonstrations
            for i, demo in enumerate(self.model_data['individual_demos'][:5]):
                pos_fr1 = demo[:, self.data_structure['position_dims']['fr1']]
                ax1.plot(pos_fr1[:, 0], pos_fr1[:, 1], 'gray', alpha=0.3, linewidth=1)
            
            # Reproduced trajectory
            fr1_pos = result['fr1_trajectory'][:, 0:2]
            ax1.plot(fr1_pos[:, 0], fr1_pos[:, 1], colors[approach_idx], linewidth=3, 
                    label=result['method'])
            ax1.scatter(fr1_pos[0, 0], fr1_pos[0, 1], c='green', s=100, marker='o', zorder=10)
            ax1.scatter(fr1_pos[-1, 0], fr1_pos[-1, 1], c='red', s=100, marker='X', zorder=10)
            
            ax1.set_xlabel('X Position')
            ax1.set_ylabel('Y Position')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.axis('equal')
            
            # Plot FR2 trajectories
            ax2 = axes[1, approach_idx]
            ax2.set_title(f'{result["method"]}\nFR2 Trajectory', fontweight='bold')
            
            # Original demonstrations
            for i, demo in enumerate(self.model_data['individual_demos'][:5]):
                pos_fr2 = demo[:, self.data_structure['position_dims']['fr2']]
                ax2.plot(pos_fr2[:, 0], pos_fr2[:, 1], 'gray', alpha=0.3, linewidth=1)
            
            # Generated/adapted FR2 trajectory
            if 'fr2_trajectory' in result:
                if result['fr2_trajectory'].shape[1] >= 2:
                    fr2_pos = result['fr2_trajectory'][:, 1:3]  # Skip time column
                else:
                    fr2_pos = result['fr2_trajectory']
                    
                ax2.plot(fr2_pos[:, 0], fr2_pos[:, 1], colors[approach_idx], linewidth=3,
                        label=result['method'])
                ax2.scatter(fr2_pos[0, 0], fr2_pos[0, 1], c='green', s=100, marker='o', zorder=10)
                ax2.scatter(fr2_pos[-1, 0], fr2_pos[-1, 1], c='red', s=100, marker='X', zorder=10)
            
            # Target position
            ax2.scatter(target_pos[0], target_pos[1], c='blue', s=200, marker='*', 
                       edgecolor='black', linewidth=2, label='Target', zorder=15)
            
            ax2.set_xlabel('X Position')
            ax2.set_ylabel('Y Position')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.axis('equal')
            
            approach_idx += 1
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs('plots', exist_ok=True)
        filename = f'plots/tpgmm_approach_comparison_{target_pos[0]:.2f}_{target_pos[1]:.2f}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved: {filename}")
        
        plt.show()


def test_proper_tpgmm(model_path):
    """Test the proper TP-GMM implementation"""
    print("=" * 60)
    print("TESTING PROPER TP-GMM APPROACHES")
    print("=" * 60)
    
    try:
        reproducer = ProperTPGMMReproducer(model_path)
        
        # Test targets
        test_targets = [
            np.array([0.1, 0.1]),
            np.array([0.01, 0.15]),
            np.array([0.5, 0.3]),
            np.array([0.5, 0])
        ]
        
        for i, target in enumerate(test_targets):
            print(f"\n{'='*50}")
            print(f"TEST {i+1}: Target {target}")
            print(f"{'='*50}")
            
            results = reproducer.compare_approaches(target, n_steps=200)
            
            print(f"✓ Test {i+1} completed")
        
        print(f"\n{'='*60}")
        print("ALL TESTS COMPLETED")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main execution"""
    especific_path = '#39_16'
    model_file = f'data/tpgmm_gait_model{especific_path}.pkl'
    
    if not os.path.exists(model_file):
        print(f"✗ Model file not found: {model_file}")
        return
    
    test_proper_tpgmm(model_file)


if __name__ == "__main__":
    main()