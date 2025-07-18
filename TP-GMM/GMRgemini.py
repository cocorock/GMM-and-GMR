import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class TPGMMReproducer:
    """
    Handles loading a trained TP-GMM model and reproducing trajectories
    for a new target frame.
    """
    def __init__(self, model_path: str):
        """
        Initializes the reproducer by loading a pre-trained model.

        Args:
            model_path (str): Path to the saved .pkl model file.
        """
        print(f"Loading TP-GMM model from: {model_path}")
        try:
            self.model_data = joblib.load(model_path)
            self.gmm = self.model_data['gmm_model']
            self.data_structure = self.model_data['data_structure']
            print("✓ Model loaded successfully.")
            print(f"  - Number of components: {self.gmm.n_components}")
            print(f"  - Original demonstrations: {len(self.model_data['individual_demos'])}")
        except FileNotFoundError:
            print(f"✗ Error: Model file not found at '{model_path}'")
            raise
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise

    def get_adapted_trajectory(self, target_pos_fr2: np.ndarray, n_steps: int = 100) -> (np.ndarray, np.ndarray):
        """
        Generates an adapted trajectory for FR1 based on a target in FR2.
        This is the core of the TP-GMM adaptation.

        Args:
            target_pos_fr2 (np.ndarray): The new target [x, y] position for FR2.
            n_steps (int): The number of points to generate for the trajectory.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
            - The mean of the reproduced trajectory for FR1 (shape: [n_steps, 5]).
            - The covariance matrices of the reproduced trajectory for FR1.
        """
        print(f"\nGenerating adapted trajectory for target FR2 position: {target_pos_fr2}")

        # 1. Define Input and Output Dimensions
        time_dim = [self.data_structure['time_dim']]
        fr2_pos_dims = self.data_structure['position_dims']['fr2']
        in_idx = time_dim + fr2_pos_dims
        out_idx = self.data_structure['fr1_dims']
        
        t = np.linspace(0, 1, n_steps).reshape(-1, 1)

        # --- Gaussian Mixture Regression ---
        
        mu_in = [self.gmm.means_[k, in_idx] for k in range(self.gmm.n_components)]
        mu_out = [self.gmm.means_[k, out_idx] for k in range(self.gmm.n_components)]
        
        reg_matrix = []
        for k in range(self.gmm.n_components):
            sigma_in = self.gmm.covariances_[k][np.ix_(in_idx, in_idx)]
            sigma_in_out = self.gmm.covariances_[k][np.ix_(in_idx, out_idx)]
            # FIX #1: The transpose (.T) was removed here. This was the cause of the ValueError.
            # The original line was: reg_matrix.append(np.linalg.solve(sigma_in, sigma_in_out).T)
            reg_matrix.append(np.linalg.solve(sigma_in, sigma_in_out))

        query = np.hstack([t, np.tile(target_pos_fr2, (n_steps, 1))])

        weights = np.zeros((n_steps, self.gmm.n_components))
        for k in range(self.gmm.n_components):
            sigma_in_k = self.gmm.covariances_[k][np.ix_(in_idx, in_idx)]
            mvn = np.exp(-0.5 * np.sum(np.dot(query - mu_in[k], 
                                             np.linalg.inv(sigma_in_k)) * (query - mu_in[k]), axis=1))
            prior = self.gmm.weights_[k]
            weights[:, k] = prior * mvn

        # FIX #2: Add a small epsilon for numerical stability to prevent division by zero.
        # This solves the RuntimeWarning.
        weight_sum = np.sum(weights, axis=1, keepdims=True)
        weights = weights / (weight_sum + 1e-9)

        repro_mean = np.zeros((n_steps, len(out_idx)))
        for k in range(self.gmm.n_components):
            # The matrix multiplication now works because reg_matrix[k] has the correct shape (3, 5)
            # and (query - mu_in[k]) has shape (100, 3). The result of the @ operation is (100, 5).
            repro_mean += weights[:, k:k+1] * (mu_out[k] + (query - mu_in[k]) @ reg_matrix[k])

        repro_cov = None 

        print("✓ Trajectory generation complete.")
        return repro_mean, repro_cov

    def plot_results(self, reproduced_trajectory: np.ndarray, target_pos_fr2: np.ndarray):
        """
        Visualizes the original and reproduced trajectories.
        """
        print("Plotting results...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('TP-GMM Trajectory Adaptation', fontsize=16)

        # --- Plot 1: FR1 (Robot Leg Frame) ---
        ax1.set_title('Adapted Trajectory in FR1 (Robot Leg)', fontweight='bold')
        
        for i, demo in enumerate(self.model_data['individual_demos']):
            pos_fr1 = demo[:, self.data_structure['position_dims']['fr1']]
            ax1.plot(pos_fr1[:, 0], pos_fr1[:, 1], color='gray', alpha=0.4, label='Original Demos' if i == 0 else "")

        repro_pos_fr1 = reproduced_trajectory[:, 0:2]
        ax1.plot(repro_pos_fr1[:, 0], repro_pos_fr1[:, 1], color='red', linewidth=3, label='Adapted Trajectory')
        ax1.scatter(repro_pos_fr1[0, 0], repro_pos_fr1[0, 1], c='red', marker='o', s=100, label='Start', zorder=5)
        ax1.scatter(repro_pos_fr1[-1, 0], repro_pos_fr1[-1, 1], c='red', marker='x', s=100, label='End', zorder=5)
        
        ax1.set_xlabel('Position X')
        ax1.set_ylabel('Position Y')
        ax1.grid(True, linestyle='--')
        ax1.legend()
        ax1.axis('equal')

        # --- Plot 2: FR2 (Task Frame) ---
        ax2.set_title('Target in FR2 (Task Frame)', fontweight='bold')

        for i, demo in enumerate(self.model_data['individual_demos']):
            pos_fr2 = demo[:, self.data_structure['position_dims']['fr2']]
            ax2.plot(pos_fr2[:, 0], pos_fr2[:, 1], color='gray', alpha=0.4, label='Original Demos' if i == 0 else "")

        ax2.scatter(target_pos_fr2[0], target_pos_fr2[1], c='blue', marker='*', s=250, 
                    edgecolor='black', label='New Target Position', zorder=10)

        ax2.set_xlabel('Position X')
        ax2.set_ylabel('Position Y')
        ax2.grid(True, linestyle='--')
        ax2.legend()
        ax2.axis('equal')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('plots/tpgmm_adapted_trajectory.png', dpi=300)
        plt.show()


def main():
    """
    Main execution function.
    """
    # --- Configuration ---
    especific_path = '#39_16'
    model_file = f'data/tpgmm_gait_model{especific_path}.pkl'
    
    # Try the original target again, or a new one
    new_target_fr2 = np.array([1, 1]) 
    
    trajectory_steps = 100
    
    # --- Execution ---
    try:
        reproducer = TPGMMReproducer(model_path=model_file)
        
        adapted_fr1_mean, _ = reproducer.get_adapted_trajectory(
            target_pos_fr2=new_target_fr2,
            n_steps=trajectory_steps
        )
        
        reproducer.plot_results(
            reproduced_trajectory=adapted_fr1_mean,
            target_pos_fr2=new_target_fr2
        )
        
        print("\n✓ Adaptation and plotting finished successfully.")

    except Exception as e:
        print(f"\n✗ An error occurred in the main pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()