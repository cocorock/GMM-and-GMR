import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
import joblib
import os

class TaskParameterizedGMM:
    """
    Proper Task-Parameterized GMM implementation with coordinate frame transformations
    
    This implements the true TP-GMM algorithm where:
    1. Skills are learned relative to multiple coordinate frames
    2. Adaptation happens through frame transformations
    3. Gaussian components are transformed according to new task parameters
    """
    
    def __init__(self, model_path: str):
        """Initialize TP-GMM with frame analysis"""
        print(f"Loading TP-GMM model: {model_path}")
        
        self.model_data = joblib.load(model_path)
        self.gmm = self.model_data['gmm_model']
        self.data_structure = self.model_data['data_structure']
        
        # Analyze coordinate frames from training data
        self._analyze_coordinate_frames()
        
        print("✓ TP-GMM loaded with frame analysis complete")

    def _analyze_coordinate_frames(self):
        """
        Analyze the coordinate frames present in the training data
        
        For gait data, we identify:
        - Frame 1 (FR1): Robot leg frame - typically relative to robot base
        - Frame 2 (FR2): Task frame - typically relative to environment/target
        """
        print("\n=== Coordinate Frame Analysis ===")
        
        training_data = self.model_data['training_data']
        demonstrations = self.model_data['individual_demos']
        
        # Extract frame data
        self.fr1_data = training_data[:, 1:6]   # FR1: [pos_x, pos_y, vel_x, vel_y, orient]
        self.fr2_data = training_data[:, 6:11]  # FR2: [pos_x, pos_y, vel_x, vel_y, orient]
        
        # Analyze frame origins and orientations from demonstrations
        self.frame_origins = {'fr1': [], 'fr2': []}
        self.frame_orientations = {'fr1': [], 'fr2': []}
        
        for demo in demonstrations:
            # For each demonstration, extract the "origin" of each frame
            # This could be the starting position, ending position, or reference point
            
            fr1_pos = demo[:, 1:3]  # FR1 positions throughout demo
            fr2_pos = demo[:, 6:8]  # FR2 positions throughout demo
            
            # Use starting position as frame origin for this demo
            fr1_origin = fr1_pos[0]  # Could also use ending position or center
            fr2_origin = fr2_pos[0]
            
            # Estimate frame orientation (simplified - could be more sophisticated)
            if len(fr1_pos) > 1:
                fr1_direction = fr1_pos[-1] - fr1_pos[0]  # Overall movement direction
                fr1_angle = np.arctan2(fr1_direction[1], fr1_direction[0])
            else:
                fr1_angle = 0.0
                
            if len(fr2_pos) > 1:
                fr2_direction = fr2_pos[-1] - fr2_pos[0]
                fr2_angle = np.arctan2(fr2_direction[1], fr2_direction[0])
            else:
                fr2_angle = 0.0
            
            self.frame_origins['fr1'].append(fr1_origin)
            self.frame_origins['fr2'].append(fr2_origin)
            self.frame_orientations['fr1'].append(fr1_angle)
            self.frame_orientations['fr2'].append(fr2_angle)
        
        # Convert to arrays
        self.frame_origins['fr1'] = np.array(self.frame_origins['fr1'])
        self.frame_origins['fr2'] = np.array(self.frame_origins['fr2'])
        self.frame_orientations['fr1'] = np.array(self.frame_orientations['fr1'])
        self.frame_orientations['fr2'] = np.array(self.frame_orientations['fr2'])
        
        print(f"Analyzed {len(demonstrations)} demonstrations")
        print(f"FR1 origins range: X[{np.min(self.frame_origins['fr1'][:, 0]):.3f}, {np.max(self.frame_origins['fr1'][:, 0]):.3f}] "
              f"Y[{np.min(self.frame_origins['fr1'][:, 1]):.3f}, {np.max(self.frame_origins['fr1'][:, 1]):.3f}]")
        print(f"FR2 origins range: X[{np.min(self.frame_origins['fr2'][:, 0]):.3f}, {np.max(self.frame_origins['fr2'][:, 0]):.3f}] "
              f"Y[{np.min(self.frame_origins['fr2'][:, 1]):.3f}, {np.max(self.frame_origins['fr2'][:, 1]):.3f}]")

    def compute_frame_transformation(self, old_origin, old_angle, new_origin, new_angle):
        """
        Compute 2D transformation matrix from old frame to new frame
        
        Args:
            old_origin: [x, y] position of old frame origin
            old_angle: orientation of old frame (radians)
            new_origin: [x, y] position of new frame origin  
            new_angle: orientation of new frame (radians)
            
        Returns:
            A: 2x2 rotation matrix
            b: 2x1 translation vector
        """
        # Rotation from old to new frame
        angle_diff = new_angle - old_angle
        cos_a, sin_a = np.cos(angle_diff), np.sin(angle_diff)
        
        A = np.array([[cos_a, -sin_a],
                      [sin_a,  cos_a]])
        
        # Translation: new_origin - A * old_origin
        b = new_origin - A @ old_origin
        
        return A, b

    def transform_gaussian_component(self, mu, sigma, A_fr1, b_fr1, A_fr2, b_fr2):
        """
        Transform a Gaussian component according to frame transformations
        
        Args:
            mu: mean vector [time, fr1_pos, fr1_vel, fr1_orient, fr2_pos, fr2_vel, fr2_orient]
            sigma: covariance matrix
            A_fr1, b_fr1: transformation for FR1 frame
            A_fr2, b_fr2: transformation for FR2 frame
            
        Returns:
            mu_new: transformed mean
            sigma_new: transformed covariance
        """
        # Create transformation matrices for full state vector
        # State vector: [time, fr1_pos_x, fr1_pos_y, fr1_vel_x, fr1_vel_y, fr1_orient, 
        #                      fr2_pos_x, fr2_pos_y, fr2_vel_x, fr2_vel_y, fr2_orient]
        
        dim = len(mu)
        A_full = np.eye(dim)  # Start with identity
        b_full = np.zeros(dim)
        
        # Time dimension (index 0) - no transformation
        # A_full[0, 0] = 1.0  # Already set by identity
        
        # FR1 position transformation (indices 1, 2)
        A_full[1:3, 1:3] = A_fr1
        b_full[1:3] = b_fr1
        
        # FR1 velocity transformation (indices 3, 4) - only rotation, no translation
        A_full[3:5, 3:5] = A_fr1
        
        # FR1 orientation transformation (index 5) - add angle difference
        # For simplicity, assume orientation transforms like: new_orient = old_orient + angle_diff
        angle_diff_fr1 = np.arctan2(A_fr1[1, 0], A_fr1[0, 0])  # Extract angle from rotation matrix
        b_full[5] = angle_diff_fr1
        
        # FR2 position transformation (indices 6, 7)
        A_full[6:8, 6:8] = A_fr2
        b_full[6:8] = b_fr2
        
        # FR2 velocity transformation (indices 8, 9) - only rotation
        A_full[8:10, 8:10] = A_fr2
        
        # FR2 orientation transformation (index 10)
        angle_diff_fr2 = np.arctan2(A_fr2[1, 0], A_fr2[0, 0])
        b_full[10] = angle_diff_fr2
        
        # Apply transformation to Gaussian
        mu_new = A_full @ mu + b_full
        sigma_new = A_full @ sigma @ A_full.T
        
        return mu_new, sigma_new

    def adapt_trajectory(self, new_fr2_origin, new_fr2_angle=None, reference_demo_idx=None, n_steps=100):
        """
        Adapt trajectory using proper TP-GMM frame transformation
        
        Args:
            new_fr2_origin: [x, y] new origin for FR2 frame
            new_fr2_angle: new orientation for FR2 frame (if None, estimated from direction)
            reference_demo_idx: which demo to use as reference (if None, auto-select)
            n_steps: number of points in output trajectory
            
        Returns:
            adapted_trajectory: adapted trajectory
            transformation_info: details about the transformation applied
        """
        print(f"\n=== TP-GMM Frame Adaptation ===")
        print(f"New FR2 origin: {new_fr2_origin}")
        
        # Step 1: Select reference demonstration
        if reference_demo_idx is None:
            # Find demo with FR2 endpoint closest to new origin
            distances = [np.linalg.norm(origin - new_fr2_origin) 
                        for origin in self.frame_origins['fr2']]
            reference_demo_idx = np.argmin(distances)
        
        print(f"Using reference demonstration: {reference_demo_idx}")
        
        # Step 2: Get reference frame parameters
        old_fr2_origin = self.frame_origins['fr2'][reference_demo_idx]
        old_fr2_angle = self.frame_orientations['fr2'][reference_demo_idx]
        
        # For FR1, we might keep it fixed or adapt it too
        # For now, let's keep FR1 frame unchanged
        old_fr1_origin = self.frame_origins['fr1'][reference_demo_idx]
        old_fr1_angle = self.frame_orientations['fr1'][reference_demo_idx]
        new_fr1_origin = old_fr1_origin  # Keep FR1 frame unchanged
        new_fr1_angle = old_fr1_angle
        
        # If new_fr2_angle not provided, estimate it
        if new_fr2_angle is None:
            # Could be more sophisticated - for now use the old angle
            new_fr2_angle = old_fr2_angle
        
        print(f"Frame transformation:")
        print(f"  FR2: {old_fr2_origin} @ {old_fr2_angle:.3f} → {new_fr2_origin} @ {new_fr2_angle:.3f}")
        print(f"  FR1: {old_fr1_origin} @ {old_fr1_angle:.3f} → {new_fr1_origin} @ {new_fr1_angle:.3f}")
        
        # Step 3: Compute transformation matrices
        A_fr1, b_fr1 = self.compute_frame_transformation(
            old_fr1_origin, old_fr1_angle, new_fr1_origin, new_fr1_angle
        )
        A_fr2, b_fr2 = self.compute_frame_transformation(
            old_fr2_origin, old_fr2_angle, new_fr2_origin, new_fr2_angle
        )
        
        # Step 4: Transform all Gaussian components
        print("Transforming Gaussian components...")
        
        adapted_means = []
        adapted_covariances = []
        adapted_weights = self.gmm.weights_.copy()  # Weights typically don't change
        
        for k in range(self.gmm.n_components):
            mu_k = self.gmm.means_[k]
            sigma_k = self.gmm.covariances_[k]
            
            # Transform this component
            mu_new, sigma_new = self.transform_gaussian_component(
                mu_k, sigma_k, A_fr1, b_fr1, A_fr2, b_fr2
            )
            
            adapted_means.append(mu_new)
            adapted_covariances.append(sigma_new)
        
        adapted_means = np.array(adapted_means)
        adapted_covariances = np.array(adapted_covariances)
        
        print(f"✓ Transformed {self.gmm.n_components} Gaussian components")
        
        # Step 5: Generate adapted trajectory using GMR on transformed model
        print("Generating adapted trajectory...")
        
        # Create adapted GMM
        from sklearn.mixture import GaussianMixture
        adapted_gmm = GaussianMixture(n_components=self.gmm.n_components, covariance_type='full')
        adapted_gmm.weights_ = adapted_weights
        adapted_gmm.means_ = adapted_means
        adapted_gmm.covariances_ = adapted_covariances
        adapted_gmm.converged_ = True
        adapted_gmm.n_iter_ = 1
        
        # Use GMR to generate trajectory
        time_vec = np.linspace(0, 1, n_steps).reshape(-1, 1)
        
        # For GMR, we typically condition on time and predict the spatial dimensions
        in_idx = [0]  # time
        out_idx = list(range(1, 11))  # all spatial dimensions
        
        adapted_trajectory = self._apply_gmr_to_adapted_model(
            adapted_gmm, time_vec, in_idx, out_idx
        )
        
        # Step 6: Prepare transformation info
        transformation_info = {
            'reference_demo_idx': reference_demo_idx,
            'old_fr1_frame': {'origin': old_fr1_origin, 'angle': old_fr1_angle},
            'new_fr1_frame': {'origin': new_fr1_origin, 'angle': new_fr1_angle},
            'old_fr2_frame': {'origin': old_fr2_origin, 'angle': old_fr2_angle},
            'new_fr2_frame': {'origin': new_fr2_origin, 'angle': new_fr2_angle},
            'A_fr1': A_fr1, 'b_fr1': b_fr1,
            'A_fr2': A_fr2, 'b_fr2': b_fr2,
            'adapted_gmm': adapted_gmm
        }
        
        print("✓ TP-GMM adaptation complete")
        
        return adapted_trajectory, transformation_info

    def _apply_gmr_to_adapted_model(self, adapted_gmm, query, in_idx, out_idx):
        """Apply GMR to the adapted GMM model"""
        from scipy.stats import multivariate_normal
        
        n_points, input_dim = query.shape
        output_dim = len(out_idx)
        n_components = adapted_gmm.n_components
        
        # Extract means
        mu_in = adapted_gmm.means_[:, in_idx]
        mu_out = adapted_gmm.means_[:, out_idx]
        
        # Compute regression matrices
        regression_matrices = []
        for k in range(n_components):
            sigma_in = adapted_gmm.covariances_[k][np.ix_(in_idx, in_idx)]
            sigma_out_in = adapted_gmm.covariances_[k][np.ix_(out_idx, in_idx)]
            
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
            sigma_in = adapted_gmm.covariances_[k][np.ix_(in_idx, in_idx)]
            sigma_in_reg = sigma_in + np.eye(len(in_idx)) * 1e-6
            
            try:
                mvn = multivariate_normal(mu_in[k], sigma_in_reg, allow_singular=True)
                log_prob = mvn.logpdf(query.flatten())
                log_prior = np.log(adapted_gmm.weights_[k] + 1e-12)
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

    def plot_adaptation_comparison(self, adapted_trajectory, transformation_info, target_origin):
        """Plot comparison between original and adapted trajectories"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'TP-GMM Frame Adaptation\nTarget FR2 Origin: [{target_origin[0]:.3f}, {target_origin[1]:.3f}]', 
                     fontsize=16)
        
        # Get reference demonstration
        ref_idx = transformation_info['reference_demo_idx']
        reference_demo = self.model_data['individual_demos'][ref_idx]
        
        # Plot 1: FR1 trajectories
        ax1 = axes[0, 0]
        ax1.set_title('FR1 Trajectories', fontweight='bold')
        
        # Original demonstrations (gray)
        for i, demo in enumerate(self.model_data['individual_demos'][:5]):
            fr1_pos = demo[:, 1:3]
            ax1.plot(fr1_pos[:, 0], fr1_pos[:, 1], 'gray', alpha=0.3, linewidth=1)
        
        # Reference demonstration (blue)
        ref_fr1_pos = reference_demo[:, 1:3]
        ax1.plot(ref_fr1_pos[:, 0], ref_fr1_pos[:, 1], 'blue', linewidth=2, label=f'Reference Demo {ref_idx}')
        
        # Adapted trajectory (red)
        adapted_fr1_pos = adapted_trajectory[:, 0:2]  # First 2 dims are FR1 position
        ax1.plot(adapted_fr1_pos[:, 0], adapted_fr1_pos[:, 1], 'red', linewidth=3, label='Adapted Trajectory')
        
        # Mark start/end points
        ax1.scatter(adapted_fr1_pos[0, 0], adapted_fr1_pos[0, 1], c='green', s=100, marker='o', zorder=10)
        ax1.scatter(adapted_fr1_pos[-1, 0], adapted_fr1_pos[-1, 1], c='red', s=100, marker='X', zorder=10)
        
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.axis('equal')
        
        # Plot 2: FR2 trajectories
        ax2 = axes[0, 1]
        ax2.set_title('FR2 Trajectories', fontweight='bold')
        
        # Original demonstrations (gray)
        for i, demo in enumerate(self.model_data['individual_demos'][:5]):
            fr2_pos = demo[:, 6:8]
            ax2.plot(fr2_pos[:, 0], fr2_pos[:, 1], 'gray', alpha=0.3, linewidth=1)
        
        # Reference demonstration (blue)
        ref_fr2_pos = reference_demo[:, 6:8]
        ax2.plot(ref_fr2_pos[:, 0], ref_fr2_pos[:, 1], 'blue', linewidth=2, label=f'Reference Demo {ref_idx}')
        
        # Adapted trajectory (red)
        adapted_fr2_pos = adapted_trajectory[:, 5:7]  # Dims 5,6 are FR2 position
        ax2.plot(adapted_fr2_pos[:, 0], adapted_fr2_pos[:, 1], 'red', linewidth=3, label='Adapted Trajectory')
        
        # Show frame origins
        old_origin = transformation_info['old_fr2_frame']['origin']
        new_origin = transformation_info['new_fr2_frame']['origin']
        
        ax2.scatter(old_origin[0], old_origin[1], c='blue', s=150, marker='s', 
                   label='Old FR2 Origin', edgecolor='black', zorder=15)
        ax2.scatter(new_origin[0], new_origin[1], c='red', s=150, marker='*', 
                   label='New FR2 Origin', edgecolor='black', zorder=15)
        
        # Target position
        ax2.scatter(target_origin[0], target_origin[1], c='orange', s=200, marker='D', 
                   label='Target', edgecolor='black', zorder=20)
        
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.axis('equal')
        
        # Plot 3: Velocity profiles
        ax3 = axes[1, 0]
        ax3.set_title('Velocity Profiles', fontweight='bold')
        
        t = np.linspace(0, 1, len(adapted_trajectory))
        
        # FR1 velocities
        adapted_fr1_vel = adapted_trajectory[:, 2:4]
        ax3.plot(t, np.linalg.norm(adapted_fr1_vel, axis=1), 'red', linewidth=2, label='FR1 Speed')
        
        # FR2 velocities  
        adapted_fr2_vel = adapted_trajectory[:, 7:9]
        ax3.plot(t, np.linalg.norm(adapted_fr2_vel, axis=1), 'blue', linewidth=2, label='FR2 Speed')
        
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Speed')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Transformation visualization
        ax4 = axes[1, 1]
        ax4.set_title('Frame Transformation', fontweight='bold')
        
        # Show coordinate frames
        def draw_frame(ax, origin, angle, color, label, scale=0.1):
            """Draw coordinate frame axes"""
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            # X axis (red)
            x_end = origin + scale * np.array([cos_a, sin_a])
            ax.arrow(origin[0], origin[1], x_end[0]-origin[0], x_end[1]-origin[1],
                    head_width=0.02, head_length=0.02, fc=color, ec=color)
            
            # Y axis (green)  
            y_end = origin + scale * np.array([-sin_a, cos_a])
            ax.arrow(origin[0], origin[1], y_end[0]-origin[0], y_end[1]-origin[1],
                    head_width=0.02, head_length=0.02, fc=color, ec=color, alpha=0.7)
            
            ax.text(origin[0], origin[1]-0.05, label, ha='center', color=color, fontweight='bold')
        
        # Draw old and new frames
        old_fr2_origin = transformation_info['old_fr2_frame']['origin']
        old_fr2_angle = transformation_info['old_fr2_frame']['angle']
        new_fr2_origin = transformation_info['new_fr2_frame']['origin']
        new_fr2_angle = transformation_info['new_fr2_frame']['angle']
        
        draw_frame(ax4, old_fr2_origin, old_fr2_angle, 'blue', 'Old FR2')
        draw_frame(ax4, new_fr2_origin, new_fr2_angle, 'red', 'New FR2')
        
        # Draw transformation arrow
        ax4.annotate('', xy=new_fr2_origin, xytext=old_fr2_origin,
                    arrowprops=dict(arrowstyle='->', lw=2, color='green'))
        
        ax4.set_xlabel('X Position')
        ax4.set_ylabel('Y Position')
        ax4.grid(True, alpha=0.3)
        ax4.axis('equal')
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs('plots', exist_ok=True)
        filename = f'plots/tpgmm_frame_adaptation_{target_origin[0]:.3f}_{target_origin[1]:.3f}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Adaptation plot saved: {filename}")
        
        plt.show()

    def test_multiple_adaptations(self, target_origins):
        """Test adaptation with multiple target origins"""
        
        print(f"\n{'='*60}")
        print("TESTING TP-GMM FRAME ADAPTATION")
        print(f"{'='*60}")
        
        results = []
        
        for i, target_origin in enumerate(target_origins):
            print(f"\n{'='*50}")
            print(f"Test {i+1}: Target Origin {target_origin}")
            print(f"{'='*50}")
            
            try:
                # Perform adaptation
                adapted_trajectory, transformation_info = self.adapt_trajectory(
                    new_fr2_origin=target_origin,
                    n_steps=200
                )
                
                # Plot results
                self.plot_adaptation_comparison(adapted_trajectory, transformation_info, target_origin)
                
                # Store results
                results.append({
                    'target_origin': target_origin,
                    'adapted_trajectory': adapted_trajectory,
                    'transformation_info': transformation_info,
                    'success': True
                })
                
                print(f"✓ Test {i+1} completed successfully")
                
            except Exception as e:
                print(f"✗ Test {i+1} failed: {e}")
                results.append({
                    'target_origin': target_origin,
                    'success': False,
                    'error': str(e)
                })
        
        print(f"\n{'='*60}")
        print("ADAPTATION TESTING COMPLETED")
        print(f"{'='*60}")
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        print(f"Successful adaptations: {successful}/{len(results)}")
        
        return results


def main():
    """Main execution for TP-GMM frame adaptation testing"""
    
    # Configuration
    especific_path = '#39_16'
    model_file = f'data/tpgmm_gait_model{especific_path}.pkl'
    
    if not os.path.exists(model_file):
        print(f"✗ Model file not found: {model_file}")
        return
    
    try:
        # Initialize TP-GMM
        tpgmm = TaskParameterizedGMM(model_file)
        
        # Define test targets within reasonable workspace
        # Based on previous analysis, FR2 workspace is roughly X[-1.6, 0] Y[-0.04, 0.18]
        target_origins = [
            np.array([-0.5, 0.05]),   # Moderate target
            np.array([-1.0, 0.10]),   # Different position
            np.array([-0.3, 0.15]),   # Near boundary
            np.array([-1.2, 0.02]),   # Another test point
        ]
        
        # Test adaptations
        results = tpgmm.test_multiple_adaptations(target_origins)
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()