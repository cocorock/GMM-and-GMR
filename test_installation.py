import sys
import os

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    # Test imports based on the repository structure
    import numpy as np
    from tpgmm.tpgmm import TPGMM
    from tpgmm.gmr import GaussianMixtureRegression
    
    print("✓ TaskParameterizedGaussianMixtureModels imported successfully!")
    print("✓ All dependencies are working correctly!")
    
    # Test basic functionality
    print("\nTesting basic functionality...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    n_components = 3
    n_frames = 2
    n_vars = 5
    
    # Create a simple TPGMM instance
    tpgmm = TPGMM(n_components=n_components, n_frames=n_frames, n_vars=n_vars)
    print("✓ TPGMM instance created successfully!")
    
    print("✓ Library is ready to use!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please check the repository structure and file names.")
    
    # Alternative import attempt
    try:
        # Try to import from the source directory directly
        sys.path.append('./src')  # Common source directory
        from tpgmm import TPGMM
        print("✓ Alternative import successful!")
    except ImportError:
        print("❌ Alternative import also failed.")
        print("You may need to use the custom implementation instead.")

except Exception as e:
    print(f"❌ Error: {e}")
    print("The library structure might be different than expected.")