import joblib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class CorrectTPGMM:
    """
    Proper TP-GMM implementation using Product of Gaussians.
    """
    
    def __init__(self, model_path: str, reg_lambda: float = 1e-6):
        """
        Load pre-trained TP-GMM model.
        
        Args:
            model_path: Path to saved model (.pkl file)
            reg_lambda: Regularization parameter for numerical stability
        """
        self.reg_lambda = reg_lambda
        
        # Load model
        model_data = joblib.load(model_path)
        self.gmm = model_data['gmm_model']
        self.data_structure = model_data['data_structure']
        self.original_demos = model_data['individual_demos']
        
        # Define dimension indices for each frame
        self.time_dim = self.data_structure['time_dim']
        self.fr1_pos_dims = self.data_structure['position_dims']['fr1']  # FR1 position
        self.fr1_vel_dims = self.data_structure['velocity_dims']['fr1']  # FR1 velocity  
        self.fr2_pos_dims = self.data_structure['position_dims']['fr2']  # FR2 position
        
        # Full FR1 state (position + velocity + acceleration if available)
        self.fr1_dims = self.data_structure['fr1_dims']
        
        print(f"Loaded TP-GMM model: {self.gmm.n_components} components")
        print(f"Time dim: {self.time_dim}")
        print(f"FR1 dims: {self.fr1_dims}")
        print(f"FR2 pos dims: {self.fr2_pos_dims}")
    
    def gmr_single_frame(self, input_dims: list, output_dims: list, query: np.ndarray) -> tuple:
        """
        Perform GMR for a single frame (standard GMR).
        
        Args:
            input_dims: Input dimension indices
            output_dims: Output dimension indices  
            query: Query points [N x input_dim]
            
        Returns:
            mean: Regression mean [N x output_dim]
            cov: Regression covariance [N x output_dim x output_dim]
        """
        N = query.shape[0]
        K = self.gmm.n_components
        
        # Extract means and covariances for this frame
        mu_in = self.gmm.means_[:, input_dims]
        mu_out = self.gmm.means_[:, output_dims]
        
        # Compute regression parameters for each component
        reg_matrices = []
        cond_covs = []
        
        for k in range(K):
            sigma_in = self.gmm.covariances_[k][np.ix_(input_dims, input_dims)]
            sigma_out = self.gmm.covariances_[k][np.ix_(output_dims, output_dims)]
            sigma_in_out = self.gmm.covariances_[k][np.ix_(input_dims, output_dims)]
            
            # Regularization
            sigma_in_reg = sigma_in + self.reg_lambda * np.eye(sigma_in.shape[0])
            
            # Regression matrix: A_k = Σ_in_out^T * Σ_in^(-1)
            A_k = np.linalg.solve(sigma_in_reg, sigma_in_out).T
            reg_matrices.append(A_k)
            
            # Conditional covariance: Σ_out|in = Σ_out - A_k * Σ_in_out
            cond_cov = sigma_out - A_k @ sigma_in_out
            cond_covs.append(cond_cov + self.reg_lambda * np.eye(cond_cov.shape[0]))
        
        # Compute component responsibilities
        weights = np.zeros((N, K))
        for k in range(K):
            sigma_in_k = self.gmm.covariances_[k][np.ix_(input_dims, input_dims)]
            sigma_in_k_reg = sigma_in_k + self.reg_lambda * np.eye(sigma_in_k.shape[0])
            
            mvn = multivariate_normal(mu_in[k], sigma_in_k_reg)
            weights[:, k] = self.gmm.weights_[k] * mvn.pdf(query)
        
        # Normalize weights  
        weight_sum = np.sum(weights, axis=1, keepdims=True)
        weights = weights / (weight_sum + 1e-12)
        
        # Compute weighted prediction
        mean = np.zeros((N, len(output_dims)))
        cov = np.zeros((N, len(output_dims), len(output_dims)))
        
        for k in range(K):
            # Conditional mean
            cond_mean = mu_out[k] + (query - mu_in[k]) @ reg_matrices[k].T
            mean += weights[:, k:k+1] * cond_mean
            
            # Weighted covariance
            for i in range(N):
                cov[i] += weights[i, k] * cond_covs[k]
        
        return mean, cov
    
    def product_of_gaussians(self, means: list, covs: list) -> tuple:
        """
        Compute product of multiple Gaussians (TP-GMM core operation).
        
        Product of Gaussians: N(μ₁,Σ₁) × N(μ₂,Σ₂) = N(μ_prod, Σ_prod)
        
        Σ_prod = (Σ₁⁻¹ + Σ₂⁻¹)⁻¹  
        μ_prod = Σ_prod × (Σ₁⁻¹μ₁ + Σ₂⁻¹μ₂)
        
        Args:
            means: List of mean vectors [N x dim] for each frame
            covs: List of covariance matrices [N x dim x dim] for each frame
            
        Returns:
            prod_mean: Product mean [N x dim]
            prod_cov: Product covariance [N x dim x dim]
        """
        N = means[0].shape[0]
        dim = means[0].shape[1]
        
        prod_mean = np.zeros((N, dim))
        prod_cov = np.zeros((N, dim, dim))
        
        for i in range(N):
            # Compute precision matrices (inverse covariances)
            precisions = []
            precision_means = []
            
            for mean, cov in zip(means, covs):
                # Regularize covariance
                cov_reg = cov[i] + self.reg_lambda * np.eye(dim)
                precision = np.linalg.inv(cov_reg)
                precisions.append(precision)
                precision_means.append(precision @ mean[i])
            
            # Product precision (sum of precisions)
            prod_precision = sum(precisions)
            
            # Product covariance (inverse of sum of precisions)
            prod_cov[i] = np.linalg.inv(prod_precision + self.reg_lambda * np.eye(dim))
            
            # Product mean
            prod_mean[i] = prod_cov[i] @ sum(precision_means)
        
        return prod_mean, prod_cov
    
    def generate_trajectory(self, target_fr2: np.ndarray, n_steps: int = 100) -> tuple:
        """
        Generate TP-GMM trajectory using Product of Gaussians.
        
        Args:
            target_fr2: Target position in FR2 [x, y]
            n_steps: Number of trajectory points
            
        Returns:
            trajectory: Generated trajectory in FR1
            covariance: Trajectory covariance
        """
        # Create time vector
        t = np.linspace(0, 1, n_steps).reshape(-1, 1)
        
        # --- Frame 1: Time → FR1 (temporal evolution) ---
        # Query: time only
        query_fr1 = t
        input_dims_fr1 = [self.time_dim]
        output_dims_fr1 = self.fr1_dims
        
        mean_fr1, cov_fr1 = self.gmr_single_frame(input_dims_fr1, output_dims_fr1, query_fr1)
        
        # --- Frame 2: (Time + FR2) → FR1 (task constraint) ---
        # Query: [time, target_fr2_x, target_fr2_y]
        target_repeated = np.tile(target_fr2, (n_steps, 1))
        query_fr2 = np.hstack([t, target_repeated])
        input_dims_fr2 = [self.time_dim] + self.fr2_pos_dims
        output_dims_fr2 = self.fr1_dims
        
        mean_fr2, cov_fr2 = self.gmr_single_frame(input_dims_fr2, output_dims_fr2, query_fr2)
        
        # --- Product of Gaussians (TP-GMM Core) ---
        # Combine predictions from both frames
        prod_mean, prod_cov = self.product_of_gaussians([mean_fr1, mean_fr2], [cov_fr1, cov_fr2])
        
        print(f"✓ Generated trajectory using Product of Gaussians")
        print(f"  Frame 1 (temporal): {mean_fr1.shape}")
        print(f"  Frame 2 (task): {mean_fr2.shape}")
        print(f"  Product result: {prod_mean.shape}")
        
        return prod_mean, prod_cov
    
    def plot_comparison(self, trajectory: np.ndarray, target_fr2: np.ndarray):
        """
        Plot original demonstrations vs TP-GMM adapted trajectory.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot FR1 trajectories (robot frame)
        ax1.set_title('FR1: TP-GMM Trajectory (Product of Gaussians)')
        
        # Original demonstrations
        for i, demo in enumerate(self.original_demos):
            pos_fr1 = demo[:, self.fr1_pos_dims]
            ax1.plot(pos_fr1[:, 0], pos_fr1[:, 1], 'gray', alpha=0.4, linewidth=1,
                    label='Original Demos' if i == 0 else '')
        
        # TP-GMM adapted trajectory
        pos_adapted = trajectory[:, :2]  # Extract x, y position
        ax1.plot(pos_adapted[:, 0], pos_adapted[:, 1], 'red', linewidth=2, label='TP-GMM Adapted')
        ax1.scatter(pos_adapted[0, 0], pos_adapted[0, 1], c='green', s=80, label='Start', zorder=5)
        ax1.scatter(pos_adapted[-1, 0], pos_adapted[-1, 1], c='red', s=80, label='End', zorder=5)
        
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # Plot FR2 context (task frame)
        ax2.set_title('FR2: Task Frame & Target')
        
        # Original FR2 trajectories
        for demo in self.original_demos:
            pos_fr2 = demo[:, self.fr2_pos_dims]
            ax2.plot(pos_fr2[:, 0], pos_fr2[:, 1], 'gray', alpha=0.4, linewidth=1)
        
        # New target
        ax2.scatter(target_fr2[0], target_fr2[1], c='blue', s=150, marker='*', 
                   label='New Target', edgecolor='black', zorder=5)
        
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        plt.tight_layout()
        plt.show()


def main():
    """
    Demonstrate proper TP-GMM with Product of Gaussians.
    """
    # Configuration
    model_path = 'data/tpgmm_gait_model#39_16.pkl'
    new_target = np.array([1.0, 1.0])  # New target position in FR2
    
    try:
        # Initialize TP-GMM
        tpgmm = CorrectTPGMM(model_path, reg_lambda=1e-6)
        
        # Generate trajectory using Product of Gaussians
        trajectory, covariance = tpgmm.generate_trajectory(new_target, n_steps=100)
        
        # Visualize results
        tpgmm.plot_comparison(trajectory, new_target)
        
        print(f"✓ TP-GMM completed successfully")
        print(f"  Trajectory shape: {trajectory.shape}")
        print(f"  Used Product of Gaussians: ✓")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()