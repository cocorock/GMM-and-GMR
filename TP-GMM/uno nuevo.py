import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
import os

class WorkspaceAwareTPGMM:
    """
    TP-GMM reproducer with workspace awareness and validation
    """
    
    def __init__(self, model_path: str):
        """Initialize with workspace analysis"""
        print(f"Loading TP-GMM model from: {model_path}")
        
        self.model_data = joblib.load(model_path)
        self.gmm = self.model_data['gmm_model']
        self.data_structure = self.model_data['data_structure']
        
        # Analyze workspace
        self._analyze_workspace()
        
        print("✓ Model loaded with workspace analysis complete")

    def _analyze_workspace(self):
        """Comprehensive workspace analysis"""
        print("\n=== Workspace Analysis ===")
        
        training_data = self.model_data['training_data']
        
        # Extract frame positions
        fr1_pos = training_data[:, 1:3]  # FR1 position
        fr2_pos = training_data[:, 6:8]  # FR2 position
        
        # Compute workspace bounds
        self.fr1_bounds = {
            'x_min': np.min(fr1_pos[:, 0]), 'x_max': np.max(fr1_pos[:, 0]),
            'y_min': np.min(fr1_pos[:, 1]), 'y_max': np.max(fr1_pos[:, 1])
        }
        
        self.fr2_bounds = {
            'x_min': np.min(fr2_pos[:, 0]), 'x_max': np.max(fr2_pos[:, 0]),
            'y_min': np.min(fr2_pos[:, 1]), 'y_max': np.max(fr2_pos[:, 1])
        }
        
        print(f"FR1 workspace: X[{self.fr1_bounds['x_min']:.3f}, {self.fr1_bounds['x_max']:.3f}] "
              f"Y[{self.fr1_bounds['y_min']:.3f}, {self.fr1_bounds['y_max']:.3f}]")
        print(f"FR2 workspace: X[{self.fr2_bounds['x_min']:.3f}, {self.fr2_bounds['x_max']:.3f}] "
              f"Y[{self.fr2_bounds['y_min']:.3f}, {self.fr2_bounds['y_max']:.3f}]")
        
        # Analyze demonstration endpoints for realistic targets
        self.demo_endpoints = []
        self.demo_startpoints = []
        
        for demo in self.model_data['individual_demos']:
            fr2_traj = demo[:, self.data_structure['position_dims']['fr2']]
            if len(fr2_traj) > 0:
                self.demo_startpoints.append(fr2_traj[0])
                self.demo_endpoints.append(fr2_traj[-1])
        
        self.demo_endpoints = np.array(self.demo_endpoints)
        self.demo_startpoints = np.array(self.demo_startpoints)
        
        print(f"Demonstration endpoints range:")
        print(f"  X: [{np.min(self.demo_endpoints[:, 0]):.3f}, {np.max(self.demo_endpoints[:, 0]):.3f}]")
        print(f"  Y: [{np.min(self.demo_endpoints[:, 1]):.3f}, {np.max(self.demo_endpoints[:, 1]):.3f}]")

    def validate_target(self, target_pos):
        """Check if target is within reasonable workspace"""
        x, y = target_pos
        
        # Check against FR2 bounds with some margin
        margin = 0.1
        x_valid = (self.fr2_bounds['x_min'] - margin <= x <= self.fr2_bounds['x_max'] + margin)
        y_valid = (self.fr2_bounds['y_min'] - margin <= y <= self.fr2_bounds['y_max'] + margin)
        
        is_valid = x_valid and y_valid
        
        if not is_valid:
            print(f"⚠️  Target {target_pos} is outside workspace bounds!")
            print(f"   Valid range: X[{self.fr2_bounds['x_min']:.2f}, {self.fr2_bounds['x_max']:.2f}] "
                  f"Y[{self.fr2_bounds['y_min']:.2f}, {self.fr2_bounds['y_max']:.2f}]")
        
        return is_valid

    def suggest_realistic_targets(self, n_targets=5):
        """Suggest realistic targets based on demonstration patterns"""
        print(f"\n=== Suggesting {n_targets} Realistic Targets ===")
        
        # Method 1: Interpolate between existing endpoints
        targets = []
        
        # Sample some existing endpoints
        if len(self.demo_endpoints) >= 2:
            indices = np.random.choice(len(self.demo_endpoints), min(n_targets//2, len(self.demo_endpoints)), replace=False)
            for idx in indices:
                targets.append(self.demo_endpoints[idx])
        
        # Method 2: Create slight variations of endpoints
        for i in range(n_targets - len(targets)):
            base_endpoint = self.demo_endpoints[i % len(self.demo_endpoints)]
            noise = np.random.normal(0, 0.05, 2)  # Small perturbation
            perturbed = base_endpoint + noise
            
            # Ensure it's still within bounds
            perturbed[0] = np.clip(perturbed[0], self.fr2_bounds['x_min'], self.fr2_bounds['x_max'])
            perturbed[1] = np.clip(perturbed[1], self.fr2_bounds['y_min'], self.fr2_bounds['y_max'])
            
            targets.append(perturbed)
        
        # Print suggestions
        for i, target in enumerate(targets):
            print(f"Target {i+1}: [{target[0]:6.3f}, {target[1]:6.3f}]")
        
        return targets

    def reproduce_trajectory(self, target_pos, method='coordinated', n_steps=100):
        """
        Reproduce trajectory with workspace validation
        """
        print(f"\nReproducing trajectory for target: {target_pos}")
        
        # Validate target
        if not self.validate_target(target_pos):
            print("Proceeding anyway, but results may be unrealistic...")
        
        if method == 'coordinated':
            return self._coordinated_approach(target_pos, n_steps)
        elif method == 'reference':
            return self._reference_approach(target_pos, n_steps)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _coordinated_approach(self, target_pos, n_steps):
        """Generate coordinated trajectory"""
        
        # Start from typical starting position
        start_pos = np.mean(self.demo_startpoints, axis=0)
        
        # Create smooth FR2 trajectory
        t = np.linspace(0, 1, n_steps)
        
        # Use smooth interpolation (could be improved with splines)
        fr2_pos = np.outer(1-t, start_pos) + np.outer(t, target_pos)
        
        # Compute velocities
        fr2_vel = np.zeros_like(fr2_pos)
        if n_steps > 1:
            fr2_vel[1:] = np.diff(fr2_pos, axis=0) * n_steps
            fr2_vel[0] = fr2_vel[1]
        
        # Simple orientation following movement direction
        fr2_orient = np.zeros(n_steps)
        for i in range(1, n_steps):
            if np.linalg.norm(fr2_vel[i]) > 1e-6:
                fr2_orient[i] = np.arctan2(fr2_vel[i, 1], fr2_vel[i, 0])
        fr2_orient[0] = fr2_orient[1] if n_steps > 1 else 0
        
        # Create query for GMR
        query = np.column_stack([
            t,                    # time
            fr2_pos,             # FR2 position
            fr2_vel,             # FR2 velocity  
            fr2_orient           # FR2 orientation
        ])
        
        # Apply GMR
        in_idx = [0, 6, 7, 8, 9, 10]  # time + full FR2
        out_idx = [1, 2, 3, 4, 5]     # full FR1
        
        fr1_trajectory = self._apply_gmr(query, in_idx, out_idx)
        
        return fr1_trajectory, fr2_pos, fr2_vel, fr2_orient

    def _reference_approach(self, target_pos, n_steps):
        """Reference-based adaptation"""
        
        # Find closest demonstration
        distances = [np.linalg.norm(ep - target_pos) for ep in self.demo_endpoints]
        best_demo_idx = np.argmin(distances)
        
        print(f"Using demonstration {best_demo_idx} (distance: {distances[best_demo_idx]:.3f})")
        
        reference_demo = self.model_data['individual_demos'][best_demo_idx]
        
        # Extract reference FR2 trajectory
        ref_time = reference_demo[:, 0]
        ref_fr2_pos = reference_demo[:, self.data_structure['position_dims']['fr2']]
        ref_fr2_vel = reference_demo[:, self.data_structure['velocity_dims']['fr2']]
        ref_fr2_orient = reference_demo[:, self.data_structure['orientation_dims']['fr2']]
        
        # Adapt reference to reach target
        adapted_fr2_pos, adapted_fr2_vel, adapted_fr2_orient = self._adapt_reference_trajectory(
            ref_fr2_pos, ref_fr2_vel, ref_fr2_orient, target_pos, n_steps
        )
        
        # Create query
        t = np.linspace(0, 1, n_steps)
        query = np.column_stack([
            t,
            adapted_fr2_pos,
            adapted_fr2_vel,
            adapted_fr2_orient
        ])
        
        # Apply GMR
        in_idx = [0, 6, 7, 8, 9, 10]
        out_idx = [1, 2, 3, 4, 5]
        
        fr1_trajectory = self._apply_gmr(query, in_idx, out_idx)
        
        return fr1_trajectory, adapted_fr2_pos, adapted_fr2_vel, adapted_fr2_orient

    def _adapt_reference_trajectory(self, ref_pos, ref_vel, ref_orient, target_pos, n_steps):
        """Adapt reference trajectory to reach target"""
        
        # Resample to desired length
        if len(ref_pos) != n_steps:
            t_ref = np.linspace(0, 1, len(ref_pos))
            t_new = np.linspace(0, 1, n_steps)
            
            adapted_pos = np.zeros((n_steps, 2))
            adapted_vel = np.zeros((n_steps, 2))
            adapted_orient = np.zeros(n_steps)
            
            for i in range(2):
                adapted_pos[:, i] = np.interp(t_new, t_ref, ref_pos[:, i])
                adapted_vel[:, i] = np.interp(t_new, t_ref, ref_vel[:, i])
            adapted_orient = np.interp(t_new, t_ref, ref_orient.flatten())
        else:
            adapted_pos = ref_pos.copy()
            adapted_vel = ref_vel.copy()
            adapted_orient = ref_orient.copy()
        
        # Scale and translate to reach target
        start_pos = np.mean(self.demo_startpoints, axis=0)
        ref_start = adapted_pos[0]
        ref_end = adapted_pos[-1]
        
        if np.linalg.norm(ref_end - ref_start) > 1e-6:
            # Linear transformation
            scale = np.linalg.norm(target_pos - start_pos) / np.linalg.norm(ref_end - ref_start)
            direction_ref = (ref_end - ref_start) / np.linalg.norm(ref_end - ref_start)
            direction_target = (target_pos - start_pos) / np.linalg.norm(target_pos - start_pos)
            
            # Apply transformation
            adapted_pos = start_pos + (adapted_pos - ref_start) * scale
            # Rotate to match target direction (simplified)
            if np.dot(direction_ref, direction_target) < 0.9:  # Need rotation
                # For simplicity, just scale without rotation
                pass
        
        return adapted_pos, adapted_vel, adapted_orient

    def _apply_gmr(self, query, in_idx, out_idx):
        """Apply GMR with error handling"""
        from scipy.stats import multivariate_normal
        
        n_points, input_dim = query.shape
        output_dim = len(out_idx)
        n_components = self.gmm.n_components
        
        # Extract means
        mu_in = self.gmm.means_[:, in_idx]
        mu_out = self.gmm.means_[:, out_idx]
        
        # Compute regression matrices
        regression_matrices = []
        for k in range(n_components):
            sigma_in = self.gmm.covariances_[k][np.ix_(in_idx, in_idx)]
            sigma_out_in = self.gmm.covariances_[k][np.ix_(out_idx, in_idx)]
            
            sigma_in_reg = sigma_in + np.eye(len(in_idx)) * 1e-6
            
            try:
                A = sigma_out_in @ np.linalg.inv(sigma_in_reg)
                regression_matrices.append(A)
            except:
                A = np.zeros((len(out_idx), len(in_idx)))
                regression_matrices.append(A)
        
        # Compute weights
        weights = np.zeros((n_points, n_components))
        for k in range(n_components):
            sigma_in = self.gmm.covariances_[k][np.ix_(in_idx, in_idx)]
            sigma_in_reg = sigma_in + np.eye(len(in_idx)) * 1e-6
            
            try:
                mvn = multivariate_normal(mu_in[k], sigma_in_reg, allow_singular=True)
                log_prob = mvn.logpdf(query)
                log_prior = np.log(self.gmm.weights_[k] + 1e-12)
                weights[:, k] = np.exp(log_prior + log_prob)
            except:
                weights[:, k] = 1.0 / n_components
        
        # Normalize weights
        weight_sums = np.sum(weights, axis=1, keepdims=True)
        weights = weights / (weight_sums + 1e-12)
        
        # Weighted regression
        output = np.zeros((n_points, output_dim))
        for k in range(n_components):
            diff = query - mu_in[k]
            cond_mean = mu_out[k] + (regression_matrices[k] @ diff.T).T
            weights_k = weights[:, k:k+1]
            output += weights_k * cond_mean
        
        return output

    def plot_workspace_and_results(self, target_pos, fr1_traj, fr2_pos, method_name):
        """Plot results with workspace context"""
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'TP-GMM Results: {method_name}\nTarget: [{target_pos[0]:.3f}, {target_pos[1]:.3f}]', 
                     fontsize=16)
        
        # Plot 1: FR1 results
        ax1 = axes[0]
        ax1.set_title('FR1 Trajectory', fontweight='bold')
        
        # Plot original demonstrations
        for i, demo in enumerate(self.model_data['individual_demos']):
            fr1_demo_pos = demo[:, self.data_structure['position_dims']['fr1']]
            ax1.plot(fr1_demo_pos[:, 0], fr1_demo_pos[:, 1], 'gray', alpha=0.3, linewidth=1)
        
        # Plot workspace bounds
        rect1 = Rectangle((self.fr1_bounds['x_min'], self.fr1_bounds['y_min']),
                         self.fr1_bounds['x_max'] - self.fr1_bounds['x_min'],
                         self.fr1_bounds['y_max'] - self.fr1_bounds['y_min'],
                         fill=False, edgecolor='blue', linestyle='--', linewidth=2,
                         label='Workspace')
        ax1.add_patch(rect1)
        
        # Plot reproduced trajectory
        ax1.plot(fr1_traj[:, 0], fr1_traj[:, 1], 'red', linewidth=3, label='Reproduced')
        ax1.scatter(fr1_traj[0, 0], fr1_traj[0, 1], c='green', s=100, marker='o', zorder=10, label='Start')
        ax1.scatter(fr1_traj[-1, 0], fr1_traj[-1, 1], c='red', s=100, marker='X', zorder=10, label='End')
        
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.axis('equal')
        
        # Plot 2: FR2 context
        ax2 = axes[1]
        ax2.set_title('FR2 Trajectory and Target', fontweight='bold')
        
        # Plot original demonstrations
        for i, demo in enumerate(self.model_data['individual_demos']):
            fr2_demo_pos = demo[:, self.data_structure['position_dims']['fr2']]
            ax2.plot(fr2_demo_pos[:, 0], fr2_demo_pos[:, 1], 'gray', alpha=0.3, linewidth=1)
        
        # Plot workspace bounds
        rect2 = Rectangle((self.fr2_bounds['x_min'], self.fr2_bounds['y_min']),
                         self.fr2_bounds['x_max'] - self.fr2_bounds['x_min'],
                         self.fr2_bounds['y_max'] - self.fr2_bounds['y_min'],
                         fill=False, edgecolor='blue', linestyle='--', linewidth=2,
                         label='Workspace')
        ax2.add_patch(rect2)
        
        # Plot demonstration endpoints
        ax2.scatter(self.demo_endpoints[:, 0], self.demo_endpoints[:, 1], 
                   c='lightblue', s=50, alpha=0.7, label='Demo Endpoints')
        
        # Plot generated FR2 trajectory
        ax2.plot(fr2_pos[:, 0], fr2_pos[:, 1], 'blue', linewidth=3, label='Generated FR2')
        ax2.scatter(fr2_pos[0, 0], fr2_pos[0, 1], c='green', s=100, marker='o', zorder=10)
        ax2.scatter(fr2_pos[-1, 0], fr2_pos[-1, 1], c='blue', s=100, marker='X', zorder=10)
        
        # Target position
        ax2.scatter(target_pos[0], target_pos[1], c='red', s=200, marker='*', 
                   edgecolor='black', linewidth=2, label='Target', zorder=15)
        
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.axis('equal')
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs('plots', exist_ok=True)
        safe_target = f"{target_pos[0]:.3f}_{target_pos[1]:.3f}".replace('-', 'neg')
        filename = f'plots/workspace_aware_{method_name}_{safe_target}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved: {filename}")
        
        plt.show()

    def comprehensive_test(self):
        """Run comprehensive test with realistic targets"""
        print("\n" + "="*60)
        print("COMPREHENSIVE WORKSPACE-AWARE TESTING")
        print("="*60)
        
        # Get realistic targets
        realistic_targets = self.suggest_realistic_targets(4)
        
        for i, target in enumerate(realistic_targets):
            print(f"\n{'='*50}")
            print(f"Test {i+1}: Target [{target[0]:.3f}, {target[1]:.3f}]")
            print(f"{'='*50}")
            
            # Test coordinated approach
            print("\n--- Coordinated Approach ---")
            try:
                fr1_traj, fr2_pos, fr2_vel, fr2_orient = self.reproduce_trajectory(
                    target, method='coordinated', n_steps=100
                )
                self.plot_workspace_and_results(target, fr1_traj, fr2_pos, 'Coordinated')
                print("✓ Coordinated approach completed")
            except Exception as e:
                print(f"✗ Coordinated approach failed: {e}")
            
            # Test reference approach
            print("\n--- Reference Approach ---")
            try:
                fr1_traj, fr2_pos, fr2_vel, fr2_orient = self.reproduce_trajectory(
                    target, method='reference', n_steps=100
                )
                self.plot_workspace_and_results(target, fr1_traj, fr2_pos, 'Reference')
                print("✓ Reference approach completed")
            except Exception as e:
                print(f"✗ Reference approach failed: {e}")
        
        print(f"\n{'='*60}")
        print("COMPREHENSIVE TESTING COMPLETED")
        print(f"{'='*60}")


def main():
    """Main execution with workspace awareness"""
    especific_path = '#39_16'
    model_file = f'data/tpgmm_gait_model{especific_path}.pkl'
    
    if not os.path.exists(model_file):
        print(f"✗ Model file not found: {model_file}")
        return
    
    try:
        reproducer = WorkspaceAwareTPGMM(model_file)
        reproducer.comprehensive_test()
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()