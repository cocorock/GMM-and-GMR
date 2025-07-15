import json
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import joblib
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.colors as mcolors
from scipy.stats import chi2


class TPGMMGaitTrainer:
    def __init__(self):
        """
        TP-GMM Trainer for gait data with 2 frames of reference
        
        Frame dimensions:
        - Position: 2D (x, y)
        - Velocity: 2D (x, y) 
        - Orientation: 1D
        Total per frame: 5D
        """
        self.num_frames = 2  # FR1 + FR2
        
        # Dimensions per frame
        self.dims = {
            'position': 2,    # x, y
            'velocity': 2,    # vx, vy
            'orientation': 1  # theta
        }
        self.point_dim = sum(self.dims.values())  # 5D per frame
        self.total_dim = self.point_dim * self.num_frames  # 10D total
        
    def load_gait_demonstrations(self, json_file: str) -> List[np.ndarray]:
        """
        Load gait demonstrations from JSON file
        
        Args:
            json_file: Path to JSON file with gait data
            
        Returns:
            List of demonstration arrays
        """
        print(f"Loading gait demonstrations from: {json_file}")
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            demonstrations = []
            
            for demo_idx, demo_data in enumerate(data):
                print(f"Processing demonstration {demo_idx}")
                
                # Process demonstration data
                demo_array = self.process_gait_demo(demo_data)
                
                if demo_array is not None and len(demo_array) > 0:
                    demonstrations.append(demo_array)
                    print(f"✓ Demo {demo_idx}: {len(demo_array)} points")
                else:
                    print(f"✗ Demo {demo_idx}: Invalid or empty")
                    
            print(f"Successfully loaded {len(demonstrations)} demonstrations")
            return demonstrations
            
        except Exception as e:
            print(f"✗ Error loading {json_file}: {e}")
            return []
    
    def process_gait_demo(self, demo_data: Dict) -> np.ndarray:
        """
        Process single gait demonstration into TP-GMM format
        
        Args:
            demo_data: Dictionary with demonstration data
            
        Returns:
            Array with TP-GMM data [N x 11] = [time | 5D_FR1 | 5D_FR2]
        """
        try:
            # Extract time (already normalized)
            time_data = np.array(demo_data['time']).flatten()
            n_points = len(time_data)
            
            # Extract FR1 data (robot leg frame)
            pos_fr1 = np.array(demo_data['ankle_pos_FR1'])  # [N x 2]
            vel_fr1 = np.array(demo_data['ankle_pos_FR1_velocity'])  # [N x 2]
            orient_fr1 = np.array(demo_data['ankle_orientation_FR1']).flatten()  # [N x 1]
            
            # Extract FR2 data (task frame)
            pos_fr2 = np.array(demo_data['ankle_pos_FR2'])  # [N x 2]
            vel_fr2 = np.array(demo_data['ankle_pos_FR2_velocity'])  # [N x 2]
            orient_fr2 = np.array(demo_data['ankle_orientation_FR2']).flatten()  # [N x 1]
            
            # Combine FR1 data: [pos_x, pos_y, vel_x, vel_y, orient]
            fr1_data = np.column_stack([
                pos_fr1,           # 2D position
                vel_fr1,           # 2D velocity  
                orient_fr1         # 1D orientation
            ])
            
            # Combine FR2 data: [pos_x, pos_y, vel_x, vel_y, orient]
            fr2_data = np.column_stack([
                pos_fr2,           # 2D position
                vel_fr2,           # 2D velocity
                orient_fr2         # 1D orientation
            ])
            
            # Create TP-GMM data: [time | FR1 | FR2]
            tpgmm_data = np.column_stack([
                time_data.reshape(-1, 1),  # Time dimension
                fr1_data,                   # 5D FR1
                fr2_data                    # 5D FR2
            ])
            
            print(f"  Processed {n_points} points, shape: {tpgmm_data.shape}")
            return tpgmm_data
            
        except Exception as e:
            print(f"Error processing demonstration: {e}")
            return None
    
    def plot_first_demonstrations(self, demonstrations: List[np.ndarray], n_demos: int = 3):
        """
        Plot first n demonstrations showing trajectories for both frames
        
        Args:
            demonstrations: List of demonstration arrays
            n_demos: Number of demonstrations to plot
        """
        n_demos = min(n_demos, len(demonstrations))
        colors = plt.cm.Set1(np.linspace(0, 1, n_demos))
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'First {n_demos} Gait Demonstrations', fontsize=16)
        
        # Column titles
        axes[0, 0].set_title('Frame FR1 (Robot Leg)', fontsize=14, fontweight='bold')
        axes[0, 1].set_title('Frame FR2 (Task Frame)', fontsize=14, fontweight='bold')
        
        for demo_idx in range(n_demos):
            demo = demonstrations[demo_idx]
            time = demo[:, 0]
            color = colors[demo_idx]
            label = f'Demo {demo_idx + 1}'
            
            # FR1 data (columns 1-5)
            pos_fr1 = demo[:, 1:3]      # position x,y
            vel_fr1 = demo[:, 3:5]      # velocity x,y
            orient_fr1 = demo[:, 5]     # orientation
            
            # FR2 data (columns 6-10)
            pos_fr2 = demo[:, 6:8]      # position x,y
            vel_fr2 = demo[:, 8:10]     # velocity x,y
            orient_fr2 = demo[:, 10]    # orientation
            
            # Plot FR1 - Position
            axes[0, 0].plot(pos_fr1[:, 0], pos_fr1[:, 1], 
                           color=color, label=label, linewidth=2)
            axes[0, 0].set_xlabel('Position X')
            axes[0, 0].set_ylabel('Position Y')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
            
            # Plot FR2 - Position
            axes[0, 1].plot(pos_fr2[:, 0], pos_fr2[:, 1], 
                           color=color, label=label, linewidth=2)
            axes[0, 1].set_xlabel('Position X')
            axes[0, 1].set_ylabel('Position Y')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
            
            # Plot FR1 - Velocity
            axes[1, 0].plot(vel_fr1[:, 0], vel_fr1[:, 1], 
                           color=color, label=label, linewidth=2)
            axes[1, 0].set_xlabel('Velocity X')
            axes[1, 0].set_ylabel('Velocity Y')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
            
            # Plot FR2 - Velocity
            axes[1, 1].plot(vel_fr2[:, 0], vel_fr2[:, 1], 
                           color=color, label=label, linewidth=2)
            axes[1, 1].set_xlabel('Velocity X')
            axes[1, 1].set_ylabel('Velocity Y')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
            
            # Plot FR1 - Orientation
            axes[2, 0].plot(time, orient_fr1, 
                           color=color, label=label, linewidth=2)
            axes[2, 0].set_xlabel('Time')
            axes[2, 0].set_ylabel('Orientation (rad)')
            axes[2, 0].grid(True, alpha=0.3)
            axes[2, 0].legend()
            
            # Plot FR2 - Orientation
            axes[2, 1].plot(time, orient_fr2, 
                           color=color, label=label, linewidth=2)
            axes[2, 1].set_xlabel('Time')
            axes[2, 1].set_ylabel('Orientation (rad)')
            axes[2, 1].grid(True, alpha=0.3)
            axes[2, 1].legend()
        
        plt.tight_layout()
        plt.savefig(f'plots/gait_demonstrations_{n_demos}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def compute_and_plot_pca(self, demonstrations: List[np.ndarray]):
        """
        Compute and plot PCA for both frames
        
        Args:
            demonstrations: List of demonstration arrays
        """
        # Combine all demonstrations
        all_data = np.vstack(demonstrations)
        
        # Extract FR1 and FR2 data (without time)
        fr1_data = all_data[:, 1:6]   # 5D FR1 data
        fr2_data = all_data[:, 6:11]  # 5D FR2 data
        
        # Compute PCA
        pca_fr1 = PCA(n_components=2)
        pca_fr2 = PCA(n_components=2)
        
        fr1_transformed = pca_fr1.fit_transform(fr1_data)
        fr2_transformed = pca_fr2.fit_transform(fr2_data)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('PCA Analysis of Gait Data', fontsize=16)
        
        # Plot FR1 PCA
        scatter1 = axes[0].scatter(fr1_transformed[:, 0], fr1_transformed[:, 1], 
                                  c=all_data[:, 0], cmap='viridis', alpha=0.6)
        axes[0].set_title(f'FR1 PCA (Explained variance: {pca_fr1.explained_variance_ratio_.sum():.2%})')
        axes[0].set_xlabel(f'PC1 ({pca_fr1.explained_variance_ratio_[0]:.1%})')
        axes[0].set_ylabel(f'PC2 ({pca_fr1.explained_variance_ratio_[1]:.1%})')
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=axes[0], label='Time')
        
        # Plot FR2 PCA
        scatter2 = axes[1].scatter(fr2_transformed[:, 0], fr2_transformed[:, 1], 
                                  c=all_data[:, 0], cmap='viridis', alpha=0.6)
        axes[1].set_title(f'FR2 PCA (Explained variance: {pca_fr2.explained_variance_ratio_.sum():.2%})')
        axes[1].set_xlabel(f'PC1 ({pca_fr2.explained_variance_ratio_[0]:.1%})')
        axes[1].set_ylabel(f'PC2 ({pca_fr2.explained_variance_ratio_[1]:.1%})')
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=axes[1], label='Time')
        
        plt.tight_layout()
        plt.savefig(f'plots/pca_analysis_{len(demonstrations)}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print PCA components
        print("\n=== PCA Analysis Results ===")
        print(f"FR1 - Explained variance ratio: {pca_fr1.explained_variance_ratio_}")
        print(f"FR1 - Total explained variance: {pca_fr1.explained_variance_ratio_.sum():.2%}")
        print(f"FR2 - Explained variance ratio: {pca_fr2.explained_variance_ratio_}")
        print(f"FR2 - Total explained variance: {pca_fr2.explained_variance_ratio_.sum():.2%}")
        
        return pca_fr1, pca_fr2
    
    def optimize_n_components(self, data: np.ndarray, max_components: int = 15) -> int:
        """
        Optimize number of GMM components using BIC/AIC
        
        Args:
            data: Training data
            max_components: Maximum components to test
            
        Returns:
            Optimal number of components
        """
        n_components_range = range(2, min(max_components, len(data)//10) + 1)
        bic_scores = []
        aic_scores = []
        best_bic = np.inf
        best_n_components = 2
        
        print(f"Optimizing number of components (2 to {max(n_components_range)})...")
        
        for n in n_components_range:
            try:
                gmm = GaussianMixture(
                    n_components=n,
                    covariance_type='full',
                    reg_covar=1e-6,
                    random_state=42,
                    max_iter=100
                )
                gmm.fit(data)
                
                bic = gmm.bic(data)
                aic = gmm.aic(data)
                
                bic_scores.append(bic)
                aic_scores.append(aic)
                
                if bic < best_bic:
                    best_bic = bic
                    best_n_components = n
                    
                print(f"  n={n}: BIC={bic:.1f}, AIC={aic:.1f}")
                    
            except Exception as e:
                print(f"  n={n}: Error - {e}")
                break
        
        print(f"✓ Optimal components: {best_n_components} (BIC: {best_bic:.1f})")
        return best_n_components
    
    def plot_advanced_gmm_gaussians(self, gmm_model, data: np.ndarray, feature_indices: Tuple[int, int] = (1, 2)):
        """
        Advanced plot of GMM Gaussians with multiple confidence levels and contours
        
        Args:
            gmm_model: Trained GMM model
            data: Training data
            feature_indices: Which features to plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Advanced GMM Visualization (Features {feature_indices[0]} vs {feature_indices[1]})', fontsize=16)
        
        # Extract 2D features
        x_idx, y_idx = feature_indices
        X = data[:, [x_idx, y_idx]]
        
        # 1. Ellipses with confidence levels
        ax1 = axes[0, 0]
        confidence_levels = [0.39, 0.68, 0.95]  # 1, 2, 3 sigma equivalents
        chi2_vals = [chi2.ppf(conf, df=2) for conf in confidence_levels]
        
        # Plot data points
        scatter = ax1.scatter(X[:, 0], X[:, 1], c=data[:, 0], cmap='viridis', alpha=0.6, s=20)
        
        # Color map for components
        colors = plt.cm.Set1(np.linspace(0, 1, gmm_model.n_components))
        
        for i in range(gmm_model.n_components):
            # Get 2D mean and covariance
            mean_2d = gmm_model.means_[i, [x_idx, y_idx]]
            cov_2d = gmm_model.covariances_[i][[x_idx, y_idx]][:, [x_idx, y_idx]]
            
            # Eigendecomposition
            eigenvals, eigenvecs = np.linalg.eigh(cov_2d)
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
            
            # Plot ellipses for different confidence levels
            for j, (conf, chi2_val) in enumerate(zip(confidence_levels, chi2_vals)):
                width = 2 * np.sqrt(eigenvals[0] * chi2_val)
                height = 2 * np.sqrt(eigenvals[1] * chi2_val)
                
                alpha = 0.4 - j * 0.1
                ellipse = Ellipse(mean_2d, width, height, angle=angle,
                                facecolor=colors[i], alpha=alpha,
                                edgecolor=colors[i], linewidth=2-j*0.5)
                ax1.add_patch(ellipse)
            
            # Plot component center
            ax1.plot(mean_2d[0], mean_2d[1], 'o', color=colors[i],
                    markersize=10, markeredgecolor='black', markeredgewidth=2,
                    label=f'Comp {i+1} (w={gmm_model.weights_[i]:.2f})')
        
        ax1.set_xlabel(f'Feature {x_idx}')
        ax1.set_ylabel(f'Feature {y_idx}')
        ax1.set_title('Confidence Ellipses')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Probability contours
        ax2 = axes[0, 1]
        
        # Create grid
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        # Create grid points for probability calculation
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        full_grid = np.zeros((len(grid_points), data.shape[1]))
        full_grid[:, x_idx] = grid_points[:, 0]
        full_grid[:, y_idx] = grid_points[:, 1]
        
        # Set other dimensions to mean values
        other_dims = [i for i in range(data.shape[1]) if i not in [x_idx, y_idx]]
        for dim in other_dims:
            full_grid[:, dim] = np.mean(data[:, dim])
        
        # Calculate probabilities
        log_probs = gmm_model.score_samples(full_grid)
        probs = np.exp(log_probs).reshape(xx.shape)
        
        # Plot contours
        contours = ax2.contour(xx, yy, probs, levels=8, colors='red', alpha=0.8)
        ax2.clabel(contours, inline=True, fontsize=8)
        ax2.scatter(X[:, 0], X[:, 1], c=data[:, 0], cmap='viridis', alpha=0.6, s=20)
        ax2.set_xlabel(f'Feature {x_idx}')
        ax2.set_ylabel(f'Feature {y_idx}')
        ax2.set_title('Probability Contours')
        ax2.grid(True, alpha=0.3)
        
        # 3. Probability heatmap
        ax3 = axes[1, 0]
        im = ax3.imshow(probs, extent=[x_min, x_max, y_min, y_max], 
                       origin='lower', cmap='hot', alpha=0.8)
        plt.colorbar(im, ax=ax3, label='Probability Density')
        ax3.scatter(X[:, 0], X[:, 1], c='white', s=20, alpha=0.8, edgecolors='black')
        ax3.set_xlabel(f'Feature {x_idx}')
        ax3.set_ylabel(f'Feature {y_idx}')
        ax3.set_title('Probability Heatmap')
        
        # 4. Component responsibilities
        ax4 = axes[1, 1]
        responsibilities = gmm_model.predict_proba(data)
        
        # Show responsibilities for dominant component
        dominant_component = np.argmax(responsibilities, axis=1)
        max_responsibility = np.max(responsibilities, axis=1)
        
        scatter = ax4.scatter(X[:, 0], X[:, 1], c=dominant_component, 
                             s=max_responsibility*100, cmap='tab10', alpha=0.7)
        ax4.set_xlabel(f'Feature {x_idx}')
        ax4.set_ylabel(f'Feature {y_idx}')
        ax4.set_title('Component Assignments\n(Size = Responsibility)')
        ax4.grid(True, alpha=0.3)
        
        # Add colorbar for components
        cbar = plt.colorbar(scatter, ax=ax4, label='Component')
        cbar.set_ticks(range(gmm_model.n_components))
        
        plt.tight_layout()
        plt.savefig(f'plots/gmm_advanced_{gmm_model.n_components}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def check_sklearn_gmm_capabilities(self):
        """
        Display information about sklearn GMM visualization capabilities
        """
        print("\n=== Scikit-learn GMM Visualization Analysis ===")
        print("\n📊 Built-in Plotting Methods:")
        print("❌ sklearn.mixture.GaussianMixture has NO built-in plotting methods")
        
        print("\n✅ Available Data for Custom Visualization:")
        print("  • predict_proba() - Component responsibilities/soft assignments")
        print("  • score_samples() - Log-likelihood of data points")
        print("  • sample() - Generate new samples from the model")
        print("  • means_ - Component mean vectors")
        print("  • covariances_ - Component covariance matrices")
        print("  • weights_ - Component mixture weights")
        print("  • bic()/aic() - Model selection criteria")
        
        print("\n🎨 Recommended Visualization Approaches:")
        print("  1. Confidence ellipses using matplotlib.patches.Ellipse")
        print("  2. Probability contours using matplotlib.pyplot.contour")
        print("  3. Density heatmaps using matplotlib.pyplot.imshow")
        print("  4. Component scatter plots with color-coded responsibilities")
        print("  5. PCA projections for high-dimensional data")
        
        print("\n📈 This Implementation Provides:")
        print("  ✓ Multi-confidence level ellipses (1σ, 2σ, 3σ)")
        print("  ✓ Probability density contours and heatmaps")
        print("  ✓ Component responsibility visualization")
        print("  ✓ PCA analysis for dimensionality reduction")
        print("  ✓ Temporal evolution plots for time-series data")
    
    def train_tpgmm_model(self, demonstrations: List[np.ndarray]) -> Dict:
        """
        Train TP-GMM model on gait data
        
        Args:
            demonstrations: List of processed demonstrations
            
        Returns:
            Dictionary with trained model and metadata
        """
        print(f"\n=== Training TP-GMM on Gait Data ===")
        print(f"Demonstrations: {len(demonstrations)}")
        
        # Combine all demonstrations
        all_data = np.vstack(demonstrations)
        print(f"Total points: {len(all_data)}")
        print(f"Data dimension: {all_data.shape[1]} (expected: 11 = 1 time + 10 TP-GMM)")
        
        # Optimize number of components
        n_components = self.optimize_n_components(all_data, max_components=20)
        
        # Train final model
        print(f"\nTraining GMM with {n_components} components...")
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            reg_covar=1e-6,
            random_state=42,
            max_iter=300,
            init_params='kmeans'
        )
        
        gmm.fit(all_data)
        
        # Calculate metrics
        log_likelihood = gmm.score(all_data)
        bic = gmm.bic(all_data)
        aic = gmm.aic(all_data)
        
        print(f"✓ Model trained:")
        print(f"  Log-likelihood: {log_likelihood:.2f}")
        print(f"  BIC: {bic:.1f}")
        print(f"  AIC: {aic:.1f}")
        
        # Structure model data
        model_data = {
            'gmm_model': gmm,
            'training_data': all_data,
            'individual_demos': demonstrations,
            'n_components': n_components,
            'metrics': {
                'log_likelihood': log_likelihood,
                'bic': bic,
                'aic': aic
            },
            'data_structure': {
                'total_dim': self.total_dim + 1,  # +1 for time
                'time_dim': 0,
                'fr1_dims': list(range(1, self.point_dim + 1)),
                'fr2_dims': list(range(self.point_dim + 1, self.total_dim + 1)),
                'position_dims': {
                    'fr1': [1, 2],
                    'fr2': [6, 7]
                },
                'velocity_dims': {
                    'fr1': [3, 4],
                    'fr2': [8, 9]
                },
                'orientation_dims': {
                    'fr1': [5],
                    'fr2': [10]
                }
            },
            'frame_info': {
                'num_frames': self.num_frames,
                'dims_per_frame': self.point_dim
            }
        }
        
        return model_data
    
    def save_model(self, model_data: Dict, filename: str):
        """Save trained TP-GMM model"""
        try:
            joblib.dump(model_data, filename)
            print(f"✓ Model saved: {filename}")
            
            # Save readable info
            info_file = filename.replace('.pkl', '_info.txt')
            with open(info_file, 'w') as f:
                f.write("=== TP-GMM Gait Model Info ===\n\n")
                f.write(f"Components: {model_data['n_components']}\n")
                f.write(f"Total dimension: {model_data['data_structure']['total_dim']}\n")
                f.write(f"Demonstrations: {len(model_data['individual_demos'])}\n")
                f.write(f"Training points: {len(model_data['training_data'])}\n\n")
                f.write("Metrics:\n")
                for metric, value in model_data['metrics'].items():
                    f.write(f"  {metric}: {value:.2f}\n")
            
            print(f"✓ Info saved: {info_file}")
            
        except Exception as e:
            print(f"✗ Error saving model: {e}")

def main():
    """
    Main training pipeline for gait TP-GMM
    """
    # Initialize trainer
    trainer = TPGMMGaitTrainer()
    
    # Load gait data
    json_file = 'new_processed_gait_data#39_16.json'
    
    try:
        print("=== Loading Gait Demonstrations ===")
        demonstrations = trainer.load_gait_demonstrations(json_file)
        
        if len(demonstrations) == 0:
            print("✗ No valid demonstrations found!")
            return
        
        print(f"✓ Loaded {len(demonstrations)} demonstrations")
        
        # Check sklearn GMM capabilities
        trainer.check_sklearn_gmm_capabilities()
        
        # Plot first 3 demonstrations
        print("\n=== Plotting Trajectories ===")
        trainer.plot_first_demonstrations(demonstrations, n_demos=3)
        
        # Compute and plot PCA
        print("\n=== Computing PCA ===")
        pca_fr1, pca_fr2 = trainer.compute_and_plot_pca(demonstrations)
        
        # Train TP-GMM model
        print("\n=== Training TP-GMM ===")
        model_data = trainer.train_tpgmm_model(demonstrations)
        
        # Advanced GMM visualizations
        print("\n=== Advanced GMM Visualizations ===")
        
        # Plot for FR1 position features
        print("Visualizing FR1 position features...")
        trainer.plot_advanced_gmm_gaussians(model_data['gmm_model'], 
                                          model_data['training_data'], 
                                          feature_indices=(1, 2))
        
        # Plot for FR2 position features  
        print("Visualizing FR2 position features...")
        trainer.plot_advanced_gmm_gaussians(model_data['gmm_model'], 
                                          model_data['training_data'], 
                                          feature_indices=(6, 7))
        
        # Plot for FR1 velocity features
        print("Visualizing FR1 velocity features...")
        trainer.plot_advanced_gmm_gaussians(model_data['gmm_model'], 
                                          model_data['training_data'], 
                                          feature_indices=(3, 4))
        
        # Save model
        print("\n=== Saving Model ===")
        trainer.save_model(model_data, 'tpgmm_gait_model.pkl')
        
        print("\n✓ TP-GMM training completed successfully!")
        print(f"  Demonstrations: {len(demonstrations)}")
        print(f"  Components: {model_data['n_components']}")
        print(f"  Data dimension: {model_data['data_structure']['total_dim']}")
        print(f"  Feature mapping:")
        print(f"    Time: column 0")
        print(f"    FR1 (pos_x, pos_y, vel_x, vel_y, orient): columns 1-5")
        print(f"    FR2 (pos_x, pos_y, vel_x, vel_y, orient): columns 6-10")
        
    except Exception as e:
        print(f"✗ Error in training pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

class TPGMMGaitTrainer:
    def __init__(self):
        """
        TP-GMM Trainer for gait data with 2 frames of reference
        
        Frame dimensions:
        - Position: 2D (x, y)
        - Velocity: 2D (x, y) 
        - Orientation: 1D
        Total per frame: 5D
        """
        self.num_frames = 2  # FR1 + FR2
        
        # Dimensions per frame
        self.dims = {
            'position': 2,    # x, y
            'velocity': 2,    # vx, vy
            'orientation': 1  # theta
        }
        self.point_dim = sum(self.dims.values())  # 5D per frame
        self.total_dim = self.point_dim * self.num_frames  # 10D total
        
    def load_gait_demonstrations(self, json_file: str) -> List[np.ndarray]:
        """
        Load gait demonstrations from JSON file
        
        Args:
            json_file: Path to JSON file with gait data
            
        Returns:
            List of demonstration arrays
        """
        print(f"Loading gait demonstrations from: {json_file}")
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            demonstrations = []
            
            for demo_idx, demo_data in enumerate(data):
                print(f"Processing demonstration {demo_idx}")
                
                # Process demonstration data
                demo_array = self.process_gait_demo(demo_data)
                
                if demo_array is not None and len(demo_array) > 0:
                    demonstrations.append(demo_array)
                    print(f"✓ Demo {demo_idx}: {len(demo_array)} points")
                else:
                    print(f"✗ Demo {demo_idx}: Invalid or empty")
                    
            print(f"Successfully loaded {len(demonstrations)} demonstrations")
            return demonstrations
            
        except Exception as e:
            print(f"✗ Error loading {json_file}: {e}")
            return []
    
    def process_gait_demo(self, demo_data: Dict) -> np.ndarray:
        """
        Process single gait demonstration into TP-GMM format
        
        Args:
            demo_data: Dictionary with demonstration data
            
        Returns:
            Array with TP-GMM data [N x 11] = [time | 5D_FR1 | 5D_FR2]
        """
        try:
            # Extract time (already normalized)
            time_data = np.array(demo_data['time']).flatten()
            n_points = len(time_data)
            
            # Extract FR1 data (robot leg frame)
            pos_fr1 = np.array(demo_data['ankle_pos_FR1'])  # [N x 2]
            vel_fr1 = np.array(demo_data['ankle_pos_FR1_velocity'])  # [N x 2]
            orient_fr1 = np.array(demo_data['ankle_orientation_FR1']).flatten()  # [N x 1]
            
            # Extract FR2 data (task frame)
            pos_fr2 = np.array(demo_data['ankle_pos_FR2'])  # [N x 2]
            vel_fr2 = np.array(demo_data['ankle_pos_FR2_velocity'])  # [N x 2]
            orient_fr2 = np.array(demo_data['ankle_orientation_FR2']).flatten()  # [N x 1]
            
            # Combine FR1 data: [pos_x, pos_y, vel_x, vel_y, orient]
            fr1_data = np.column_stack([
                pos_fr1,           # 2D position
                vel_fr1,           # 2D velocity  
                orient_fr1         # 1D orientation
            ])
            
            # Combine FR2 data: [pos_x, pos_y, vel_x, vel_y, orient]
            fr2_data = np.column_stack([
                pos_fr2,           # 2D position
                vel_fr2,           # 2D velocity
                orient_fr2         # 1D orientation
            ])
            
            # Create TP-GMM data: [time | FR1 | FR2]
            tpgmm_data = np.column_stack([
                time_data.reshape(-1, 1),  # Time dimension
                fr1_data,                   # 5D FR1
                fr2_data                    # 5D FR2
            ])
            
            print(f"  Processed {n_points} points, shape: {tpgmm_data.shape}")
            return tpgmm_data
            
        except Exception as e:
            print(f"Error processing demonstration: {e}")
            return None
    
    def plot_first_demonstrations(self, demonstrations: List[np.ndarray], n_demos: int = 3):
        """
        Plot first n demonstrations showing trajectories for both frames
        
        Args:
            demonstrations: List of demonstration arrays
            n_demos: Number of demonstrations to plot
        """
        n_demos = min(n_demos, len(demonstrations))
        colors = plt.cm.Set1(np.linspace(0, 1, n_demos))
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'First {n_demos} Gait Demonstrations', fontsize=16)
        
        # Column titles
        axes[0, 0].set_title('Frame FR1 (Robot Leg)', fontsize=14, fontweight='bold')
        axes[0, 1].set_title('Frame FR2 (Task Frame)', fontsize=14, fontweight='bold')
        
        for demo_idx in range(n_demos):
            demo = demonstrations[demo_idx]
            time = demo[:, 0]
            color = colors[demo_idx]
            label = f'Demo {demo_idx + 1}'
            
            # FR1 data (columns 1-5)
            pos_fr1 = demo[:, 1:3]      # position x,y
            vel_fr1 = demo[:, 3:5]      # velocity x,y
            orient_fr1 = demo[:, 5]     # orientation
            
            # FR2 data (columns 6-10)
            pos_fr2 = demo[:, 6:8]      # position x,y
            vel_fr2 = demo[:, 8:10]     # velocity x,y
            orient_fr2 = demo[:, 10]    # orientation
            
            # Plot FR1 - Position
            axes[0, 0].plot(pos_fr1[:, 0], pos_fr1[:, 1], 
                           color=color, label=label, linewidth=2)
            axes[0, 0].set_xlabel('Position X')
            axes[0, 0].set_ylabel('Position Y')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
            
            # Plot FR2 - Position
            axes[0, 1].plot(pos_fr2[:, 0], pos_fr2[:, 1], 
                           color=color, label=label, linewidth=2)
            axes[0, 1].set_xlabel('Position X')
            axes[0, 1].set_ylabel('Position Y')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
            
            # Plot FR1 - Velocity
            axes[1, 0].plot(vel_fr1[:, 0], vel_fr1[:, 1], 
                           color=color, label=label, linewidth=2)
            axes[1, 0].set_xlabel('Velocity X')
            axes[1, 0].set_ylabel('Velocity Y')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
            
            # Plot FR2 - Velocity
            axes[1, 1].plot(vel_fr2[:, 0], vel_fr2[:, 1], 
                           color=color, label=label, linewidth=2)
            axes[1, 1].set_xlabel('Velocity X')
            axes[1, 1].set_ylabel('Velocity Y')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
            
            # Plot FR1 - Orientation
            axes[2, 0].plot(time, orient_fr1, 
                           color=color, label=label, linewidth=2)
            axes[2, 0].set_xlabel('Time')
            axes[2, 0].set_ylabel('Orientation (rad)')
            axes[2, 0].grid(True, alpha=0.3)
            axes[2, 0].legend()
            
            # Plot FR2 - Orientation
            axes[2, 1].plot(time, orient_fr2, 
                           color=color, label=label, linewidth=2)
            axes[2, 1].set_xlabel('Time')
            axes[2, 1].set_ylabel('Orientation (rad)')
            axes[2, 1].grid(True, alpha=0.3)
            axes[2, 1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def compute_and_plot_pca(self, demonstrations: List[np.ndarray]):
        """
        Compute and plot PCA for both frames
        
        Args:
            demonstrations: List of demonstration arrays
        """
        # Combine all demonstrations
        all_data = np.vstack(demonstrations)
        
        # Extract FR1 and FR2 data (without time)
        fr1_data = all_data[:, 1:6]   # 5D FR1 data
        fr2_data = all_data[:, 6:11]  # 5D FR2 data
        
        # Compute PCA
        pca_fr1 = PCA(n_components=2)
        pca_fr2 = PCA(n_components=2)
        
        fr1_transformed = pca_fr1.fit_transform(fr1_data)
        fr2_transformed = pca_fr2.fit_transform(fr2_data)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('PCA Analysis of Gait Data', fontsize=16)
        
        # Plot FR1 PCA
        scatter1 = axes[0].scatter(fr1_transformed[:, 0], fr1_transformed[:, 1], 
                                  c=all_data[:, 0], cmap='viridis', alpha=0.6)
        axes[0].set_title(f'FR1 PCA (Explained variance: {pca_fr1.explained_variance_ratio_.sum():.2%})')
        axes[0].set_xlabel(f'PC1 ({pca_fr1.explained_variance_ratio_[0]:.1%})')
        axes[0].set_ylabel(f'PC2 ({pca_fr1.explained_variance_ratio_[1]:.1%})')
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=axes[0], label='Time')
        
        # Plot FR2 PCA
        scatter2 = axes[1].scatter(fr2_transformed[:, 0], fr2_transformed[:, 1], 
                                  c=all_data[:, 0], cmap='viridis', alpha=0.6)
        axes[1].set_title(f'FR2 PCA (Explained variance: {pca_fr2.explained_variance_ratio_.sum():.2%})')
        axes[1].set_xlabel(f'PC1 ({pca_fr2.explained_variance_ratio_[0]:.1%})')
        axes[1].set_ylabel(f'PC2 ({pca_fr2.explained_variance_ratio_[1]:.1%})')
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=axes[1], label='Time')
        
        plt.tight_layout()
        plt.show()
        
        # Print PCA components
        print("\n=== PCA Analysis Results ===")
        print(f"FR1 - Explained variance ratio: {pca_fr1.explained_variance_ratio_}")
        print(f"FR1 - Total explained variance: {pca_fr1.explained_variance_ratio_.sum():.2%}")
        print(f"FR2 - Explained variance ratio: {pca_fr2.explained_variance_ratio_}")
        print(f"FR2 - Total explained variance: {pca_fr2.explained_variance_ratio_.sum():.2%}")
        
        return pca_fr1, pca_fr2
    
    def optimize_n_components(self, data: np.ndarray, max_components: int = 15) -> int:
        """
        Optimize number of GMM components using BIC/AIC
        
        Args:
            data: Training data
            max_components: Maximum components to test
            
        Returns:
            Optimal number of components
        """
        n_components_range = range(2, min(max_components, len(data)//10) + 1)
        bic_scores = []
        aic_scores = []
        best_bic = np.inf
        best_n_components = 2
        
        print(f"Optimizing number of components (2 to {max(n_components_range)})...")
        
        for n in n_components_range:
            try:
                gmm = GaussianMixture(
                    n_components=n,
                    covariance_type='full',
                    reg_covar=1e-6,
                    random_state=42,
                    max_iter=100
                )
                gmm.fit(data)
                
                bic = gmm.bic(data)
                aic = gmm.aic(data)
                
                bic_scores.append(bic)
                aic_scores.append(aic)
                
                if bic < best_bic:
                    best_bic = bic
                    best_n_components = n
                    
                print(f"  n={n}: BIC={bic:.1f}, AIC={aic:.1f}")
                    
            except Exception as e:
                print(f"  n={n}: Error - {e}")
                break
        
        print(f"✓ Optimal components: {best_n_components} (BIC: {best_bic:.1f})")
        return best_n_components
    
    def plot_gmm_gaussians(self, gmm_model, data: np.ndarray, feature_indices: Tuple[int, int] = (1, 2)):
        """
        Plot GMM Gaussians in 2D projection
        
        Args:
            gmm_model: Trained GMM model
            data: Training data
            feature_indices: Which features to plot (default: first 2 spatial features)
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Extract 2D features
        x_idx, y_idx = feature_indices
        X = data[:, [x_idx, y_idx]]
        
        # Plot data points colored by time
        scatter = ax.scatter(X[:, 0], X[:, 1], c=data[:, 0], cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter, label='Time')
        
        # Plot Gaussian ellipses
        colors = plt.cm.Set1(np.linspace(0, 1, gmm_model.n_components))
        
        for i in range(gmm_model.n_components):
            # Get mean and covariance for this component
            mean_2d = gmm_model.means_[i, [x_idx, y_idx]]
            cov_2d = gmm_model.covariances_[i][[x_idx, y_idx]][:, [x_idx, y_idx]]
            
            # Compute eigenvalues and eigenvectors
            eigenvals, eigenvecs = np.linalg.eigh(cov_2d)
            
            # Calculate ellipse parameters
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
            width = 2 * np.sqrt(eigenvals[0]) * 2  # 2 sigma
            height = 2 * np.sqrt(eigenvals[1]) * 2  # 2 sigma
            
            # Create ellipse
            ellipse = Ellipse(mean_2d, width, height, angle=angle, 
                            facecolor=colors[i], alpha=0.3, 
                            edgecolor=colors[i], linewidth=2,
                            label=f'Component {i+1}')
            ax.add_patch(ellipse)
            
            # Plot component center
            ax.plot(mean_2d[0], mean_2d[1], 'o', color=colors[i], 
                   markersize=8, markeredgecolor='black', markeredgewidth=1)
        
        ax.set_xlabel(f'Feature {x_idx}')
        ax.set_ylabel(f'Feature {y_idx}')
        ax.set_title(f'GMM Components (Features {x_idx} vs {y_idx})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def train_tpgmm_model(self, demonstrations: List[np.ndarray]) -> Dict:
        """
        Train TP-GMM model on gait data
        
        Args:
            demonstrations: List of processed demonstrations
            
        Returns:
            Dictionary with trained model and metadata
        """
        print(f"\n=== Training TP-GMM on Gait Data ===")
        print(f"Demonstrations: {len(demonstrations)}")
        
        # Combine all demonstrations
        all_data = np.vstack(demonstrations)
        print(f"Total points: {len(all_data)}")
        print(f"Data dimension: {all_data.shape[1]} (expected: 11 = 1 time + 10 TP-GMM)")
        
        # Optimize number of components
        n_components = self.optimize_n_components(all_data, max_components=20)
        
        # Train final model
        print(f"\nTraining GMM with {n_components} components...")
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            reg_covar=1e-6,
            random_state=42,
            max_iter=300,
            init_params='kmeans'
        )
        
        gmm.fit(all_data)
        
        # Calculate metrics
        log_likelihood = gmm.score(all_data)
        bic = gmm.bic(all_data)
        aic = gmm.aic(all_data)
        
        print(f"✓ Model trained:")
        print(f"  Log-likelihood: {log_likelihood:.2f}")
        print(f"  BIC: {bic:.1f}")
        print(f"  AIC: {aic:.1f}")
        
        # Structure model data
        model_data = {
            'gmm_model': gmm,
            'training_data': all_data,
            'individual_demos': demonstrations,
            'n_components': n_components,
            'metrics': {
                'log_likelihood': log_likelihood,
                'bic': bic,
                'aic': aic
            },
            'data_structure': {
                'total_dim': self.total_dim + 1,  # +1 for time
                'time_dim': 0,
                'fr1_dims': list(range(1, self.point_dim + 1)),
                'fr2_dims': list(range(self.point_dim + 1, self.total_dim + 1)),
                'position_dims': {
                    'fr1': [1, 2],
                    'fr2': [6, 7]
                },
                'velocity_dims': {
                    'fr1': [3, 4],
                    'fr2': [8, 9]
                },
                'orientation_dims': {
                    'fr1': [5],
                    'fr2': [10]
                }
            },
            'frame_info': {
                'num_frames': self.num_frames,
                'dims_per_frame': self.point_dim
            }
        }
        
        return model_data
    
    def save_model(self, model_data: Dict, filename: str):
        """Save trained TP-GMM model"""
        try:
            joblib.dump(model_data, filename)
            print(f"✓ Model saved: {filename}")
            
            # Save readable info
            info_file = filename.replace('.pkl', '_info.txt')
            with open(info_file, 'w') as f:
                f.write("=== TP-GMM Gait Model Info ===\n\n")
                f.write(f"Components: {model_data['n_components']}\n")
                f.write(f"Total dimension: {model_data['data_structure']['total_dim']}\n")
                f.write(f"Demonstrations: {len(model_data['individual_demos'])}\n")
                f.write(f"Training points: {len(model_data['training_data'])}\n\n")
                f.write("Metrics:\n")
                for metric, value in model_data['metrics'].items():
                    f.write(f"  {metric}: {value:.2f}\n")
            
            print(f"✓ Info saved: {info_file}")
            
        except Exception as e:
            print(f"✗ Error saving model: {e}")

def main():
    """
    Main training pipeline for gait TP-GMM
    """
    # Initialize trainer
    trainer = TPGMMGaitTrainer()
    
    # Load gait data
    json_file = 'data/new_processed_gait_data#39_16.json'
    
    try:
        print("=== Loading Gait Demonstrations ===")
        demonstrations = trainer.load_gait_demonstrations(json_file)
        
        if len(demonstrations) == 0:
            print("✗ No valid demonstrations found!")
            return
        
        print(f"✓ Loaded {len(demonstrations)} demonstrations")
        
        # Plot first 3 demonstrations
        print("\n=== Plotting Trajectories ===")
        trainer.plot_first_demonstrations(demonstrations, n_demos=10)
        
        # Compute and plot PCA
        print("\n=== Computing PCA ===")
        pca_fr1, pca_fr2 = trainer.compute_and_plot_pca(demonstrations)
        
        # Train TP-GMM model
        print("\n=== Training TP-GMM ===")
        model_data = trainer.train_tpgmm_model(demonstrations)
        
        # Plot GMM Gaussians
        print("\n=== Plotting GMM Components ===")
        # Plot for FR1 position
        trainer.plot_gmm_gaussians(model_data['gmm_model'], 
                                 model_data['training_data'], 
                                 feature_indices=(1, 2))
        
        # Plot for FR2 position  
        trainer.plot_gmm_gaussians(model_data['gmm_model'], 
                                 model_data['training_data'], 
                                 feature_indices=(6, 7))
        
        # Save model
        print("\n=== Saving Model ===")
        trainer.save_model(model_data, 'data/tpgmm_gait_model.pkl')

        print("\n✓ TP-GMM training completed successfully!")
        print(f"  Demonstrations: {len(demonstrations)}")
        print(f"  Components: {model_data['n_components']}")
        print(f"  Data dimension: {model_data['data_structure']['total_dim']}")
        
    except Exception as e:
        print(f"✗ Error in training pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
