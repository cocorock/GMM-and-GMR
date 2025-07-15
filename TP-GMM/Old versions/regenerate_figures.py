# Script to regenerate figures from a saved TP-GMM model
from tpgmm_gait_training import TPGMMGaitTrainer
import matplotlib.pyplot as plt

def regenerate_all_figures(model_file='tpgmm_gait_model_fixed.pkl'):
    """
    Regenerate all figures from a saved TP-GMM model
    """
    print("Regenerating figures from saved model...")
    
    # Initialize trainer
    trainer = TPGMMGaitTrainer()
    
    # Load the saved model
    print(f"Loading model from: {model_file}")
    model_data = trainer.load_tpgmm_model(model_file)
    
    if model_data is None:
        print("Failed to load model. Make sure the file exists and is valid.")
        return
    
    print("Model loaded successfully!")
    print(f"Model has {len(model_data['individual_demos'])} demonstrations")
    print(f"Total data points: {len(model_data['training_data'])}")
    
    # Generate all visualizations
    print("\n=== Generating Visualizations ===")
    
    print("1. Creating training data visualization...")
    trainer.visualize_training_data(model_data)
    
    print("2. Creating 2D trajectory visualization...")
    trainer.visualize_2d_trajectories(model_data)
    
    print("3. Performing PCA analysis and visualization...")
    trainer.analyze_latent_space_pca(model_data)
    trainer.visualize_pca_analysis(model_data)
    
    print("\n✓ All figures regenerated successfully!")
    print("Files saved:")
    print("  - tpgmm_training_data_analysis.png")
    print("  - tpgmm_2d_trajectories.png") 
    print("  - tpgmm_pca_comprehensive_analysis.png")

def create_specific_figure(model_file='tpgmm_gait_model_fixed.pkl', figure_type='all'):
    """
    Create a specific figure type
    
    Args:
        model_file: Path to saved model
        figure_type: 'training', '2d', 'pca', or 'all'
    """
    trainer = TPGMMGaitTrainer()
    model_data = trainer.load_tpgmm_model(model_file)
    
    if model_data is None:
        print("Failed to load model")
        return
    
    if figure_type == 'training' or figure_type == 'all':
        print("Creating training data visualization...")
        trainer.visualize_training_data(model_data)
    
    if figure_type == '2d' or figure_type == 'all':
        print("Creating 2D trajectory visualization...")
        trainer.visualize_2d_trajectories(model_data)
    
    if figure_type == 'pca' or figure_type == 'all':
        print("Creating PCA analysis...")
        trainer.analyze_latent_space_pca(model_data)
        trainer.visualize_pca_analysis(model_data)

# Example usage:
if __name__ == "__main__":
    # Regenerate all figures
    regenerate_all_figures()
    
    # Or create just one type:
    # create_specific_figure(figure_type='2d')  # Just 2D trajectories
    # create_specific_figure(figure_type='pca')  # Just PCA analysis
    # create_specific_figure(figure_type='training')  # Just training data