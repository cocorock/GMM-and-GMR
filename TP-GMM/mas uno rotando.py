import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import joblib
import os

class RotationalTPGMM:
    """
    Enhanced Task-Parameterized GMM with full rotational support
    
    This implementation properly handles:
    1. Target position AND orientation
    2. Rotational transformations of Gaussian components
    3. Proper handling of angular coordinates
    4. Visualization of rotated coordinate frames
    """
    
    def __init__(self, model_path: str):
        """Initialize TP-GMM with rotational analysis"""
        print(f"Loading TP-GMM model: {model_path}")
        
        self.model_data = joblib.load(model_path)
        self.gmm = self.model_data['gmm_model']
        self.data_structure = self.model_data['data_structure']
        
        # Analyze coordinate frames including orientations
        self._analyze_frames_with_orientations()
        
        print("✓ TP-GMM loaded with rotational analysis complete")

    def _analyze_frames_with_orientations(self):
        """
        Comprehensive frame analysis including orientations
        """
        print("\n=== Frame Analysis with Orientations ===")
        
        training_data = self.model_data['training_data']
        demonstrations = self.model_data['individual_demos']
        
        # Initialize frame data storage
        self.frame_data = {
            'fr1': {'origins': [], 'orientations': [], 'end_orientations': []},
            'fr2': {'origins': [], 'orientations': [], 'end_orientations': []}
        }
        
        for demo in demonstrations:
            # Extract position and orientation data
            fr1_pos = demo[:, 1:3]      # FR1 positions
            fr1_orient = demo[:, 5]     # FR1 orientations
            fr2_pos = demo[:, 6:8]      # FR2 positions  
            fr2_orient = demo[:, 10]    # FR2 orientations
            
            if len(demo) > 0:
                # Frame origins (starting positions)
                self.frame_data['fr1']['origins'].append(fr1_pos[0])
                self.frame_data['fr2']['origins'].append(fr2_pos[0])
                
                # Starting orientations
                self.frame_data['fr1']['orientations'].append(fr1_orient[0])
                self.frame_data['fr2']['orientations'].append(fr2_orient[0])
                
                # Ending orientations (for target orientation estimation)
                self.frame_data['fr1']['end_orientations'].append(fr1_orient[-1])
                self.frame_data['fr2']['end_orientations'].append(fr2_orient[-1])
        
        # Convert to arrays
        for frame in ['fr1', 'fr2']:
            for key in ['origins', 'orientations', 'end_orientations']:
                self.frame_data[frame][key] = np.array(self.frame_data[frame][key])
        
        # Print analysis
        print(f"Analyzed {len(demonstrations)} demonstrations")
        
        for frame_name in ['fr1', 'fr2']:
            origins = self.frame_data[frame_name]['origins']
            orientations = self.frame_data[frame_name]['orientations']
            end_orientations = self.frame_data[frame_name]['end_orientations']
            
            print(f"{frame_name.upper()} analysis:")
            print(f"  Origins range: X[{np.min(origins[:, 0]):.3f}, {np.max(origins[:, 0]):.3f}] "
                  f"Y[{np.min(origins[:, 1]):.3f}, {np.max(origins[:, 1]):.3f}]")
            print(f"  Start orientations: [{np.min(orientations):.3f}, {np.max(orientations):.3f}] rad "
                  f"({np.degrees(np.min(orientations)):.1f}°, {np.degrees(np.max(orientations)):.1f}°)")
            print(f"  End orientations: [{np.min(end_orientations):.3f}, {np.max(end_orientations):.3f}] rad "
                  f"({np.degrees(np.min(end_orientations)):.1f}°, {np.degrees(np.max(end_orientations)):.1f}°)")

    def compute_full_transformation(self, old_pos, old_angle, new_pos, new_angle):
        """
        Compute full 2D transformation (rotation + translation)
        
        Args:
            old_pos: [x, y] old frame position
            old_angle: old frame orientation (radians)
            new_pos: [x, y] new frame position
            new_angle: new frame orientation (radians)
            
        Returns:
            A: 2x2 rotation matrix
            b: 2x1 translation vector
            angle_diff: angular difference (radians)
        """
        # Angular difference (handle wrapping)
        angle_diff = self._angle_difference(new_angle, old_angle)
        
        # Rotation matrix for the angular difference
        cos_a, sin_a = np.cos(angle_diff), np.sin(angle_diff)
        A = np.array([[cos_a, -sin_a],
                      [sin_a,  cos_a]])
        
        # Translation: new_pos - A * old_pos
        b = new_pos - A @ old_pos
        
        return A, b, angle_diff

    def _angle_difference(self, angle_new, angle_old):
        """
        Compute angular difference handling 2π wrapping
        """
        diff = angle_new - angle_old
        
        # Wrap to [-π, π]
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
            
        return diff

    def transform_gaussian_with_rotation(self, mu, sigma, A_fr1, b_fr1, angle_diff_fr1, 
                                       A_fr2, b_fr2, angle_diff_fr2):
        """
        Transform Gaussian component with proper rotational handling
        
        Args:
            mu: mean vector [11D]
            sigma: covariance matrix [11x11]
            A_fr1, b_fr1, angle_diff_fr1: FR1 transformation parameters
            A_fr2, b_fr2, angle_diff_fr2: FR2 transformation parameters
        """
        dim = len(mu)
        A_full = np.eye(dim)
        b_full = np.zeros(dim)
        
        # State vector: [time, fr1_pos_x, fr1_pos_y, fr1_vel_x, fr1_vel_y, fr1_orient, 
        #                      fr2_pos_x, fr2_pos_y, fr2_vel_x, fr2_vel_y, fr2_orient]
        
        # Time (index 0) - no transformation
        # A_full[0, 0] = 1.0  (already set)
        
        # FR1 transformations
        # Position (indices 1, 2)
        A_full[1:3, 1:3] = A_fr1
        b_full[1:3] = b_fr1
        
        # Velocity (indices 3, 4) - rotation only, no translation
        A_full[3:5, 3:5] = A_fr1
        
        # Orientation (index 5) - add angular difference
        b_full[5] = angle_diff_fr1
        
        # FR2 transformations  
        # Position (indices 6, 7)
        A_full[6:8, 6:8] = A_fr2
        b_full[6:8] = b_fr2
        
        # Velocity (indices 8, 9) - rotation only
        A_full[8:10, 8:10] = A_fr2
        
        # Orientation (index 10) - add angular difference
        b_full[10] = angle_diff_fr2
        
        # Apply transformation
        mu_new = A_full @ mu + b_full
        sigma_new = A_full @ sigma @ A_full.T
        
        # Handle angular wrapping in the transformed mean
        mu_new[5] = self._wrap_angle(mu_new[5])   # FR1 orientation
        mu_new[10] = self._wrap_angle(mu_new[10]) # FR2 orientation
        
        return mu_new, sigma_new

    def _wrap_angle(self, angle):
        """Wrap angle to [-π, π]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def adapt_to_target_pose(self, target_fr2_pos, target_fr2_angle=None, 
                           target_fr1_pos=None, target_fr1_angle=None,
                           reference_demo_idx=None, n_steps=100):
        """
        Adapt trajectory to reach target pose (position + orientation)
        
        Args:
            target_fr2_pos: [x, y] target position for FR2
            target_fr2_angle: target orientation for FR2 (radians)
            target_fr1_pos: [x, y] target position for FR1 (optional)
            target_fr1_angle: target orientation for FR1 (optional)
            reference_demo_idx: reference demonstration index
            n_steps: trajectory length
        """
        print(f"\n=== TP-GMM Rotational Adaptation ===")
        print(f"Target FR2 pose: pos={target_fr2_pos}, angle={target_fr2_angle}")
        if target_fr1_pos is not None:
            print(f"Target FR1 pose: pos={target_fr1_pos}, angle={target_fr1_angle}")
        
        # Step 1: Select reference demonstration
        if reference_demo_idx is None:
            reference_demo_idx = self._find_best_reference_for_target(
                target_fr2_pos, target_fr2_angle
            )
        
        print(f"Using reference demonstration: {reference_demo_idx}")
        
        # Step 2: Get reference frame parameters
        ref_fr2_pos = self.frame_data['fr2']['origins'][reference_demo_idx]
        ref_fr2_angle = self.frame_data['fr2']['orientations'][reference_demo_idx]
        ref_fr1_pos = self.frame_data['fr1']['origins'][reference_demo_idx]
        ref_fr1_angle = self.frame_data['fr1']['orientations'][reference_demo_idx]
        
        # Step 3: Determine target frames
        # If target angles not specified, use reasonable defaults
        if target_fr2_angle is None:
            # Use the ending orientation of the reference demo
            target_fr2_angle = self.frame_data['fr2']['end_orientations'][reference_demo_idx]
            print(f"Auto-selected FR2 target angle: {target_fr2_angle:.3f} rad ({np.degrees(target_fr2_angle):.1f}°)")
        
        if target_fr1_pos is None:
            # Keep FR1 frame relatively unchanged (could be more sophisticated)
            target_fr1_pos = ref_fr1_pos
            target_fr1_angle = ref_fr1_angle
        elif target_fr1_angle is None:
            target_fr1_angle = ref_fr1_angle
        
        print(f"Frame transformations:")
        print(f"  FR2: {ref_fr2_pos} @ {ref_fr2_angle:.3f} → {target_fr2_pos} @ {target_fr2_angle:.3f}")
        print(f"  FR1: {ref_fr1_pos} @ {ref_fr1_angle:.3f} → {target_fr1_pos} @ {target_fr1_angle:.3f}")
        
        # Step 4: Compute transformation matrices
        A_fr1, b_fr1, angle_diff_fr1 = self.compute_full_transformation(
            ref_fr1_pos, ref_fr1_angle, target_fr1_pos, target_fr1_angle
        )
        A_fr2, b_fr2, angle_diff_fr2 = self.compute_full_transformation(
            ref_fr2_pos, ref_fr2_angle, target_fr2_pos, target_fr2_angle
        )
        
        print(f"Angular differences: FR1={np.degrees(angle_diff_fr1):.1f}°, FR2={np.degrees(angle_diff_fr2):.1f}°")
        
        # Step 5: Transform all Gaussian components
        print("Transforming Gaussian components...")
        
        adapted_means = []
        adapted_covariances = []
        
        for k in range(self.gmm.n_components):
            mu_k = self.gmm.means_[k]
            sigma_k = self.gmm.covariances_[k]
            
            # Transform with rotations
            mu_new, sigma_new = self.transform_gaussian_with_rotation(
                mu_k, sigma_k, A_fr1, b_fr1, angle_diff_fr1, A_fr2, b_fr2, angle_diff_fr2
            )
            
            adapted_means.append(mu_new)
            adapted_covariances.append(sigma_new)
        
        adapted_means = np.array(adapted_means)
        adapted_covariances = np.array(adapted_covariances)
        
        print(f"✓ Transformed {self.gmm.n_components} components with rotations")
        
        # Step 6: Generate adapted trajectory
        adapted_trajectory = self._generate_trajectory_from_adapted_model(
            adapted_means, adapted_covariances, n_steps
        )
        
        # Step 7: Package transformation info
        transformation_info = {
            'reference_demo_idx': reference_demo_idx,
            'transformations': {
                'fr1': {
                    'old_pos': ref_fr1_pos, 'old_angle': ref_fr1_angle,
                    'new_pos': target_fr1_pos, 'new_angle': target_fr1_angle,
                    'A': A_fr1, 'b': b_fr1, 'angle_diff': angle_diff_fr1
                },
                'fr2': {
                    'old_pos': ref_fr2_pos, 'old_angle': ref_fr2_angle,
                    'new_pos': target_fr2_pos, 'new_angle': target_fr2_angle,
                    'A': A_fr2, 'b': b_fr2, 'angle_diff': angle_diff_fr2
                }
            },
            'adapted_means': adapted_means,
            'adapted_covariances': adapted_covariances
        }
        
        print("✓ Rotational TP-GMM adaptation complete")
        
        return adapted_trajectory, transformation_info

    def _find_best_reference_for_target(self, target_pos, target_angle):
        """
        Find best reference demo considering both position and orientation
        """
        if target_angle is None:
            # Only consider position
            distances = [np.linalg.norm(origin - target_pos) 
                        for origin in self.frame_data['fr2']['origins']]
            return np.argmin(distances)
        
        else:
            # Consider both position and orientation
            scores = []
            for i in range(len(self.frame_data['fr2']['origins'])):
                pos_dist = np.linalg.norm(self.frame_data['fr2']['origins'][i] - target_pos)
                angle_dist = abs(self._angle_difference(
                    target_angle, 
                    self.frame_data['fr2']['end_orientations'][i]
                ))
                
                # Weighted score (you can adjust these weights)
                score = pos_dist + 0.5 * angle_dist  # 0.5 rad ≈ 30°
                scores.append(score)
            
            best_idx = np.argmin(scores)
            print(f"Best reference: pos_dist={scores[best_idx]:.3f}, "
                  f"total_score={scores[best_idx]:.3f}")
            
            return best_idx

    def _generate_trajectory_from_adapted_model(self, adapted_means, adapted_covariances, n_steps):
        """Generate trajectory using adapted GMM"""
        from scipy.stats import multivariate_normal
        
        # Create time query
        time_query = np.linspace(0, 1, n_steps).reshape(-1, 1)
        
        # GMR setup
        in_idx = [0]  # time
        out_idx = list(range(1, 11))  # all spatial dimensions
        
        # Extract adapted parameters
        mu_in = adapted_means[:, in_idx]
        mu_out = adapted_means[:, out_idx]
        
        n_components = len(adapted_means)
        output_dim = len(out_idx)
        
        # Compute regression matrices
        regression_matrices = []
        for k in range(n_components):
            sigma_in = adapted_covariances[k][np.ix_(in_idx, in_idx)]
            sigma_out_in = adapted_covariances[k][np.ix_(out_idx, in_idx)]
            
            sigma_in_reg = sigma_in + np.eye(len(in_idx)) * 1e-6
            
            try:
                A = sigma_out_in @ np.linalg.inv(sigma_in_reg)
                regression_matrices.append(A)
            except:
                A = np.zeros((output_dim, len(in_idx)))
                regression_matrices.append(A)
        
        # Compute weights
        weights = np.zeros((n_steps, n_components))
        for k in range(n_components):
            sigma_in = adapted_covariances[k][np.ix_(in_idx, in_idx)]
            sigma_in_reg = sigma_in + np.eye(len(in_idx)) * 1e-6
            
            try:
                mvn = multivariate_normal(mu_in[k], sigma_in_reg, allow_singular=True)
                log_prob = mvn.logpdf(time_query.flatten())
                log_prior = np.log(self.gmm.weights_[k] + 1e-12)
                weights[:, k] = np.exp(log_prior + log_prob)
            except:
                weights[:, k] = 1.0 / n_components
        
        # Normalize weights
        weight_sums = np.sum(weights, axis=1, keepdims=True)
        weights = weights / (weight_sums + 1e-12)
        
        # Generate trajectory
        trajectory = np.zeros((n_steps, output_dim))
        for k in range(n_components):
            diff = time_query - mu_in[k]
            cond_mean = mu_out[k] + (regression_matrices[k] @ diff.T).T
            weights_k = weights[:, k:k+1]
            trajectory += weights_k * cond_mean
        
        return trajectory

    def plot_rotational_adaptation(self, adapted_trajectory, transformation_info, save_name=None):
        """
        Enhanced plotting with rotation visualization
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Get transformation info
        fr1_info = transformation_info['transformations']['fr1']
        fr2_info = transformation_info['transformations']['fr2']
        ref_idx = transformation_info['reference_demo_idx']
        
        fig.suptitle(f'TP-GMM Rotational Adaptation (Ref Demo: {ref_idx})\n'
                     f'FR2: Δpos=[{fr2_info["new_pos"][0]-fr2_info["old_pos"][0]:.3f}, '
                     f'{fr2_info["new_pos"][1]-fr2_info["old_pos"][1]:.3f}], '
                     f'Δangle={np.degrees(fr2_info["angle_diff"]):.1f}°', fontsize=14)
        
        # Helper function to draw oriented frames
        def draw_oriented_frame(ax, pos, angle, color, label, scale=0.08):
            """Draw coordinate frame with orientation"""
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            # X-axis (main direction)
            x_end = pos + scale * np.array([cos_a, sin_a])
            ax.annotate('', xy=x_end, xytext=pos,
                       arrowprops=dict(arrowstyle='->', lw=2, color=color))
            
            # Y-axis (perpendicular)
            y_end = pos + scale * 0.7 * np.array([-sin_a, cos_a])
            ax.annotate('', xy=y_end, xytext=pos,
                       arrowprops=dict(arrowstyle='->', lw=1.5, color=color, alpha=0.7))
            
            # Label
            ax.text(pos[0], pos[1]-0.06, label, ha='center', color=color, 
                   fontweight='bold', fontsize=10)
            
            # Orientation text
            ax.text(pos[0]+0.1, pos[1], f'{np.degrees(angle):.0f}°', 
                   ha='left', color=color, fontsize=8)
        
        # Plot 1: FR1 trajectories with orientations
        ax1 = axes[0, 0]
        ax1.set_title('FR1 Trajectory with Orientations', fontweight='bold')
        
        # Original demonstrations
        for demo in self.model_data['individual_demos'][:5]:
            fr1_pos = demo[:, 1:3]
            ax1.plot(fr1_pos[:, 0], fr1_pos[:, 1], 'gray', alpha=0.3, linewidth=1)
        
        # Reference demo
        ref_demo = self.model_data['individual_demos'][ref_idx]
        ref_fr1_pos = ref_demo[:, 1:3]
        ax1.plot(ref_fr1_pos[:, 0], ref_fr1_pos[:, 1], 'blue', linewidth=2, 
                label=f'Reference Demo {ref_idx}')
        
        # Adapted trajectory
        adapted_fr1_pos = adapted_trajectory[:, 0:2]
        ax1.plot(adapted_fr1_pos[:, 0], adapted_fr1_pos[:, 1], 'red', linewidth=3, 
                label='Adapted Trajectory')
        
        # Draw orientation frames at key points
        n_frames = 5
        indices = np.linspace(0, len(adapted_trajectory)-1, n_frames, dtype=int)
        
        for i in indices:
            pos = adapted_fr1_pos[i]
            orient = adapted_trajectory[i, 4]  # FR1 orientation
            alpha = 0.3 + 0.7 * (i / (len(indices)-1))  # Fade effect
            color = plt.cm.Reds(alpha)
            ax1.add_patch(plt.Circle(pos, 0.02, color=color, alpha=0.7))
            
            # Small orientation arrow
            cos_o, sin_o = np.cos(orient), np.sin(orient)
            end_pos = pos + 0.05 * np.array([cos_o, sin_o])
            ax1.annotate('', xy=end_pos, xytext=pos,
                        arrowprops=dict(arrowstyle='->', lw=1, color=color))
        
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.axis('equal')
        
        # Plot 2: FR2 trajectories with orientations
        ax2 = axes[0, 1]
        ax2.set_title('FR2 Trajectory with Orientations', fontweight='bold')
        
        # Original demonstrations
        for demo in self.model_data['individual_demos'][:5]:
            fr2_pos = demo[:, 6:8]
            ax2.plot(fr2_pos[:, 0], fr2_pos[:, 1], 'gray', alpha=0.3, linewidth=1)
        
        # Reference demo
        ref_fr2_pos = ref_demo[:, 6:8]
        ax2.plot(ref_fr2_pos[:, 0], ref_fr2_pos[:, 1], 'blue', linewidth=2,
                label=f'Reference Demo {ref_idx}')
        
        # Adapted trajectory
        adapted_fr2_pos = adapted_trajectory[:, 5:7]
        ax2.plot(adapted_fr2_pos[:, 0], adapted_fr2_pos[:, 1], 'red', linewidth=3,
                label='Adapted Trajectory')
        
        # Draw orientation frames
        for i in indices:
            pos = adapted_fr2_pos[i]
            orient = adapted_trajectory[i, 9]  # FR2 orientation
            alpha = 0.3 + 0.7 * (i / (len(indices)-1))
            color = plt.cm.Blues(alpha)
            ax2.add_patch(plt.Circle(pos, 0.02, color=color, alpha=0.7))
            
            cos_o, sin_o = np.cos(orient), np.sin(orient)
            end_pos = pos + 0.05 * np.array([cos_o, sin_o])
            ax2.annotate('', xy=end_pos, xytext=pos,
                        arrowprops=dict(arrowstyle='->', lw=1, color=color))
        
        # Show frame transformations
        draw_oriented_frame(ax2, fr2_info['old_pos'], fr2_info['old_angle'], 
                           'blue', 'Old FR2', 0.1)
        draw_oriented_frame(ax2, fr2_info['new_pos'], fr2_info['new_angle'], 
                           'red', 'New FR2', 0.1)
        
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.axis('equal')
        
        # Plot 3: Frame transformation diagram
        ax3 = axes[0, 2]
        ax3.set_title('Coordinate Frame Transformations', fontweight='bold')
        
        # Draw both frame transformations
        draw_oriented_frame(ax3, fr1_info['old_pos'], fr1_info['old_angle'], 
                           'lightblue', 'Old FR1', 0.15)
        draw_oriented_frame(ax3, fr1_info['new_pos'], fr1_info['new_angle'], 
                           'darkred', 'New FR1', 0.15)
        
        draw_oriented_frame(ax3, fr2_info['old_pos'], fr2_info['old_angle'], 
                           'blue', 'Old FR2', 0.15)
        draw_oriented_frame(ax3, fr2_info['new_pos'], fr2_info['new_angle'], 
                           'red', 'New FR2', 0.15)
        
        # Transformation arrows
        if np.linalg.norm(fr1_info['new_pos'] - fr1_info['old_pos']) > 0.01:
            ax3.annotate('', xy=fr1_info['new_pos'], xytext=fr1_info['old_pos'],
                        arrowprops=dict(arrowstyle='->', lw=2, color='green', alpha=0.7))
        
        if np.linalg.norm(fr2_info['new_pos'] - fr2_info['old_pos']) > 0.01:
            ax3.annotate('', xy=fr2_info['new_pos'], xytext=fr2_info['old_pos'],
                        arrowprops=dict(arrowstyle='->', lw=2, color='orange', alpha=0.7))
        
        ax3.set_xlabel('X Position')
        ax3.set_ylabel('Y Position')
        ax3.grid(True, alpha=0.3)
        ax3.axis('equal')
        
        # Plot 4: Orientation evolution
        ax4 = axes[1, 0]
        ax4.set_title('Orientation Evolution', fontweight='bold')
        
        t = np.linspace(0, 1, len(adapted_trajectory))
        
        # FR1 and FR2 orientations over time
        fr1_orientations = adapted_trajectory[:, 4]
        fr2_orientations = adapted_trajectory[:, 9]
        
        ax4.plot(t, np.degrees(fr1_orientations), 'red', linewidth=2, label='FR1 Orientation')
        ax4.plot(t, np.degrees(fr2_orientations), 'blue', linewidth=2, label='FR2 Orientation')
        
        # Show target orientations
        ax4.axhline(np.degrees(fr1_info['new_angle']), color='red', linestyle='--', 
                   alpha=0.7, label='FR1 Target')
        ax4.axhline(np.degrees(fr2_info['new_angle']), color='blue', linestyle='--', 
                   alpha=0.7, label='FR2 Target')
        
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Orientation (degrees)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Plot 5: Velocity profiles
        ax5 = axes[1, 1]
        ax5.set_title('Velocity Profiles', fontweight='bold')
        
        # Compute velocity magnitudes
        fr1_vel = adapted_trajectory[:, 2:4]
        fr2_vel = adapted_trajectory[:, 7:9]
        
        fr1_speed = np.linalg.norm(fr1_vel, axis=1)
        fr2_speed = np.linalg.norm(fr2_vel, axis=1)
        
        ax5.plot(t, fr1_speed, 'red', linewidth=2, label='FR1 Speed')
        ax5.plot(t, fr2_speed, 'blue', linewidth=2, label='FR2 Speed')
        
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Speed')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        # Plot 6: Transformation matrix visualization
        ax6 = axes[1, 2]
        ax6.set_title('Rotation Matrices', fontweight='bold')
        
        # Visualize rotation matrices as unit circle transformations
        theta = np.linspace(0, 2*np.pi, 50)
        unit_circle_x = np.cos(theta)
        unit_circle_y = np.sin(theta)
        unit_circle = np.array([unit_circle_x, unit_circle_y])
        
        # Original unit circle
        ax6.plot(unit_circle_x, unit_circle_y, 'gray', linewidth=1, alpha=0.5, label='Original')
        
        # Transformed circles
        if np.linalg.norm(fr1_info['A'] - np.eye(2)) > 1e-6:  # FR1 has rotation
            transformed_fr1 = fr1_info['A'] @ unit_circle
            ax6.plot(transformed_fr1[0], transformed_fr1[1], 'red', linewidth=2, 
                    label=f'FR1 Rot ({np.degrees(fr1_info["angle_diff"]):.1f}°)')
        
        if np.linalg.norm(fr2_info['A'] - np.eye(2)) > 1e-6:  # FR2 has rotation
            transformed_fr2 = fr2_info['A'] @ unit_circle
            ax6.plot(transformed_fr2[0], transformed_fr2[1], 'blue', linewidth=2, 
                    label=f'FR2 Rot ({np.degrees(fr2_info["angle_diff"]):.1f}°)')
        
        # Principal axes
        ax6.axhline(0, color='k', linewidth=0.5, alpha=0.3)
        ax6.axvline(0, color='k', linewidth=0.5, alpha=0.3)
        
        ax6.set_xlabel('X')
        ax6.set_ylabel('Y')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        ax6.axis('equal')
        ax6.set_xlim(-1.5, 1.5)
        ax6.set_ylim(-1.5, 1.5)
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs('plots', exist_ok=True)
        if save_name is None:
            save_name = f'rotational_adaptation_{fr2_info["new_pos"][0]:.2f}_{fr2_info["new_pos"][1]:.2f}_{np.degrees(fr2_info["angle_diff"]):.0f}deg'
        filename = f'plots/{save_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Rotational adaptation plot saved: {filename}")
        
        plt.show()

    def test_rotational_adaptations(self):
        """
        Test various rotational adaptations
        """
        print(f"\n{'='*60}")
        print("TESTING ROTATIONAL TP-GMM ADAPTATIONS")
        print(f"{'='*60}")
        
        # Define test cases with position and orientation targets
        test_cases = [
            {
                'name': 'Translation Only',
                'fr2_pos': np.array([-0.5, 0.05]),
                'fr2_angle': None,  # Use reference orientation
                'description': 'Move FR2 to new position, keep orientation'
            },
            {
                'name': 'Rotation Only', 
                'fr2_pos': None,  # Use reference position
                'fr2_angle': np.radians(45),  # 45 degree rotation
                'description': 'Rotate FR2 45°, keep position from reference'
            },
            {
                'name': 'Translation + Rotation',
                'fr2_pos': np.array([-0.8, 0.10]),
                'fr2_angle': np.radians(-30),  # -30 degree rotation
                'description': 'Move and rotate FR2'
            },
            {
                'name': 'Large Rotation',
                'fr2_pos': np.array([-1.0, 0.08]),
                'fr2_angle': np.radians(90),  # 90 degree rotation
                'description': 'Large 90° rotation with translation'
            },
            {
                'name': 'Negative Rotation',
                'fr2_pos': np.array([-0.3, 0.12]),
                'fr2_angle': np.radians(-60),  # -60 degree rotation
                'description': 'Negative rotation with movement'
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(test_cases):
            print(f"\n{'='*50}")
            print(f"Test {i+1}: {test_case['name']}")
            print(f"Description: {test_case['description']}")
            print(f"{'='*50}")
            
            try:
                # Handle cases where position or angle might be None
                fr2_pos = test_case['fr2_pos']
                fr2_angle = test_case['fr2_angle']
                
                # If position is None, use a reference demo's position
                if fr2_pos is None:
                    ref_idx = 0  # Use first demo as reference
                    fr2_pos = self.frame_data['fr2']['origins'][ref_idx]
                    print(f"Using reference position: {fr2_pos}")
                
                # Perform adaptation
                adapted_trajectory, transformation_info = self.adapt_to_target_pose(
                    target_fr2_pos=fr2_pos,
                    target_fr2_angle=fr2_angle,
                    n_steps=200
                )
                
                # Plot results
                save_name = f"test_{i+1}_{test_case['name'].lower().replace(' ', '_')}"
                self.plot_rotational_adaptation(
                    adapted_trajectory, 
                    transformation_info, 
                    save_name=save_name
                )
                
                # Compute quality metrics
                metrics = self._compute_adaptation_metrics(
                    adapted_trajectory, transformation_info, fr2_pos, fr2_angle
                )
                
                results.append({
                    'test_case': test_case,
                    'adapted_trajectory': adapted_trajectory,
                    'transformation_info': transformation_info,
                    'metrics': metrics,
                    'success': True
                })
                
                print(f"✓ Test {i+1} completed successfully")
                self._print_metrics(metrics)
                
            except Exception as e:
                print(f"✗ Test {i+1} failed: {e}")
                import traceback
                traceback.print_exc()
                
                results.append({
                    'test_case': test_case,
                    'success': False,
                    'error': str(e)
                })
        
        # Summary
        print(f"\n{'='*60}")
        print("ROTATIONAL ADAPTATION TESTING SUMMARY")
        print(f"{'='*60}")
        
        successful = sum(1 for r in results if r['success'])
        print(f"Successful tests: {successful}/{len(test_cases)}")
        
        if successful > 0:
            print("\nTest Results Summary:")
            print(f"{'Test':<20} {'Target Angle':<15} {'Final Error':<15} {'Smoothness':<15}")
            print("-" * 65)
            
            for i, result in enumerate(results):
                if result['success']:
                    test_name = result['test_case']['name']
                    target_angle = result['test_case']['fr2_angle']
                    metrics = result['metrics']
                    
                    target_angle_str = f"{np.degrees(target_angle):.1f}°" if target_angle is not None else "ref"
                    
                    print(f"{test_name:<20} {target_angle_str:<15} "
                          f"{metrics['final_position_error']:<15.4f} "
                          f"{metrics['trajectory_smoothness']:<15.4f}")
        
        return results

    def _compute_adaptation_metrics(self, trajectory, transformation_info, target_pos, target_angle):
        """Compute quality metrics for the adaptation"""
        
        metrics = {}
        
        # Final position error
        final_fr2_pos = trajectory[-1, 5:7]  # Last FR2 position
        metrics['final_position_error'] = np.linalg.norm(final_fr2_pos - target_pos)
        
        # Final orientation error (if target angle specified)
        if target_angle is not None:
            final_fr2_angle = trajectory[-1, 9]  # Last FR2 orientation
            angle_error = abs(self._angle_difference(final_fr2_angle, target_angle))
            metrics['final_orientation_error'] = angle_error
            metrics['final_orientation_error_deg'] = np.degrees(angle_error)
        
        # Trajectory smoothness (based on acceleration)
        fr2_pos = trajectory[:, 5:7]
        if len(fr2_pos) > 2:
            vel = np.gradient(fr2_pos, axis=0)
            accel = np.gradient(vel, axis=0)
            smoothness = np.mean(np.linalg.norm(accel, axis=1))
            metrics['trajectory_smoothness'] = smoothness
        
        # Path length
        if len(fr2_pos) > 1:
            distances = np.linalg.norm(np.diff(fr2_pos, axis=0), axis=1)
            metrics['path_length'] = np.sum(distances)
        
        # Orientation smoothness
        fr2_orientations = trajectory[:, 9]
        if len(fr2_orientations) > 1:
            orient_changes = np.abs(np.diff(fr2_orientations))
            # Handle angle wrapping
            orient_changes = np.minimum(orient_changes, 2*np.pi - orient_changes)
            metrics['orientation_smoothness'] = np.mean(orient_changes)
        
        return metrics

    def _print_metrics(self, metrics):
        """Print metrics in a readable format"""
        print("\nAdaptation Quality Metrics:")
        print("-" * 30)
        
        for key, value in metrics.items():
            if 'error' in key and 'deg' not in key:
                print(f"{key:<25}: {value:.4f}")
            elif 'deg' in key:
                print(f"{key:<25}: {value:.2f}°")
            elif 'smoothness' in key:
                print(f"{key:<25}: {value:.6f}")
            else:
                print(f"{key:<25}: {value:.4f}")

    def create_rotation_demonstration(self, base_demo_idx=0, rotation_angles=[0, 30, 60, 90]):
        """
        Create a demonstration of how rotations affect trajectories
        """
        print(f"\n=== Rotation Demonstration ===")
        
        # Use a base demonstration
        base_fr2_pos = self.frame_data['fr2']['origins'][base_demo_idx]
        base_fr2_angle = self.frame_data['fr2']['orientations'][base_demo_idx]
        
        print(f"Base position: {base_fr2_pos}")
        print(f"Base angle: {np.degrees(base_fr2_angle):.1f}°")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(rotation_angles)))
        
        for i, angle_deg in enumerate(rotation_angles):
            angle_rad = np.radians(angle_deg)
            target_angle = base_fr2_angle + angle_rad
            
            print(f"\nTesting rotation: {angle_deg}° (total: {np.degrees(target_angle):.1f}°)")
            
            try:
                adapted_trajectory, _ = self.adapt_to_target_pose(
                    target_fr2_pos=base_fr2_pos,  # Same position
                    target_fr2_angle=target_angle,  # Different angle
                    reference_demo_idx=base_demo_idx,
                    n_steps=200
                )
                
                # Plot in different subplots
                ax_idx = i % 4
                ax = axes[ax_idx]
                
                # FR1 trajectory
                fr1_pos = adapted_trajectory[:, 0:2]
                ax.plot(fr1_pos[:, 0], fr1_pos[:, 1], color=colors[i], 
                       linewidth=3, label=f'Rotation: {angle_deg}°')
                
                # Start and end points
                ax.scatter(fr1_pos[0, 0], fr1_pos[0, 1], color=colors[i], 
                          s=100, marker='o', edgecolor='black')
                ax.scatter(fr1_pos[-1, 0], fr1_pos[-1, 1], color=colors[i], 
                          s=100, marker='s', edgecolor='black')
                
                ax.set_title(f'FR1 Response to {angle_deg}° FR2 Rotation')
                ax.set_xlabel('X Position')
                ax.set_ylabel('Y Position')
                ax.grid(True, alpha=0.3)
                ax.legend()
                ax.axis('equal')
                
            except Exception as e:
                print(f"Failed for {angle_deg}°: {e}")
        
        plt.tight_layout()
        plt.suptitle(f'Effect of FR2 Rotations on FR1 Trajectories\nBase Demo: {base_demo_idx}', 
                     fontsize=16, y=1.02)
        
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/rotation_demonstration.png', dpi=300, bbox_inches='tight')
        print("✓ Rotation demonstration saved: plots/rotation_demonstration.png")
        plt.show()


def main():
    """Main execution for rotational TP-GMM testing"""
    
    # Configuration
    especific_path = '#39_16'
    model_file = f'data/tpgmm_gait_model{especific_path}.pkl'
    
    if not os.path.exists(model_file):
        print(f"✗ Model file not found: {model_file}")
        return
    
    try:
        # Initialize rotational TP-GMM
        rotational_tpgmm = RotationalTPGMM(model_file)
        
        # Run comprehensive rotational tests
        results = rotational_tpgmm.test_rotational_adaptations()

        # Create rotation demonstration
        rotational_tpgmm.create_rotation_demonstration()
        
        print(f"\n{'='*60}")
        print("ALL ROTATIONAL TESTING COMPLETED")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()