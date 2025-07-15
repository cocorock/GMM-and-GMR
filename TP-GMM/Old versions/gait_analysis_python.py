import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import warnings
from mpl_toolkits.mplot3d import Axes3D

def load_gait_data(filename='new_processed_gait_data.mat'):
    """
    Load the processed gait data from MATLAB .mat file
    """
    print("Loading gait data...")
    try:
        data = loadmat(filename)
        print("Available keys in .mat file:", list(data.keys()))
        
        # Handle the cell array structure properly
        processed_data = data['processed_gait_data']
        print(f"processed_gait_data shape: {processed_data.shape}")
        print(f"processed_gait_data dtype: {processed_data.dtype}")
        print(f"Number of cells in array: {processed_data.size}")
        
        # Show what's in each cell (more detailed inspection)
        for i in range(min(processed_data.size, 3)):  # Show first 3 cells max
            if processed_data.size == 1:
                cell_data = processed_data.item()
            else:
                cell_data = processed_data.flat[i].item()
            
            print(f"\nCell {i} detailed inspection:")
            print(f"  Type: {type(cell_data)}")
            
            if isinstance(cell_data, tuple):
                print(f"  Tuple length: {len(cell_data)}")
                for j, element in enumerate(cell_data):
                    print(f"    Element {j}: type={type(element)}")
                    if hasattr(element, 'shape'):
                        print(f"      Shape: {element.shape}")
                    if hasattr(element, 'dtype'):
                        print(f"      Dtype: {element.dtype}")
                        if hasattr(element.dtype, 'names') and element.dtype.names:
                            print(f"      Fields: {element.dtype.names}")
            elif hasattr(cell_data, 'dtype') and hasattr(cell_data.dtype, 'names'):
                print(f"  Fields: {cell_data.dtype.names}")
        
        return processed_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def debug_data_structure(gait_data):
    """
    Debug function to understand the data structure
    """
    print("\n=== DEBUGGING DATA STRUCTURE ===")
    
    field_names = gait_data.dtype.names
    for field_name in field_names:
        field_data = gait_data[field_name]
        print(f"\nField: {field_name}")
        print(f"  Type: {type(field_data)}")
        print(f"  Shape: {field_data.shape}")
        print(f"  Dtype: {field_data.dtype}")
        
        # Try to access the actual data
        try:
            if field_data.shape == (1, 1):
                actual_data = field_data[0, 0]
                print(f"  Actual data shape: {actual_data.shape}")
                print(f"  Actual data dtype: {actual_data.dtype}")
                if len(actual_data.shape) <= 2 and actual_data.size <= 10:
                    print(f"  Sample data: {actual_data}")
            else:
                print(f"  Direct access shape: {field_data.shape}")
                if field_data.size <= 10:
                    print(f"  Sample data: {field_data}")
        except Exception as e:
            print(f"  Error accessing data: {e}")
    
    print("=== END DEBUGGING ===\n")

def extract_data_fields(processed_data, cell_index=0):
    """
    Extract all relevant fields from the gait data structure
    """
    print(f"Extracting data fields from cell {cell_index}...")
    
    # Get the specific cell
    if processed_data.size == 1:
        cell_content = processed_data.item()
    else:
        cell_content = processed_data.flat[cell_index].item()
    
    print(f"Cell content type: {type(cell_content)}")
    
    if isinstance(cell_content, tuple):
        print(f"Cell contains tuple with {len(cell_content)} elements")
        
        # Based on the structure analysis, map tuple elements to data fields
        # From your original description and the shapes, the mapping appears to be:
        # Element 0: time (200, 1)
        # Element 1: pelvis_orientation (200, 1) 
        # Element 2: ankle_pos_FR1_velocity (200, 2)
        # Element 3: ankle_pos_FR1 (200, 2)
        # Element 4: ankle_pos_FR2_velocity (200, 2)
        # Element 5: ankle_pos_FR2 (200, 2)
        # Element 6: ankle_orientation_FR1 (200, 1)
        # Element 7: ankle_orientation_FR2 (200, 1)
        # Element 8: ankle_A_FR1 (200, 2, 2)
        # Element 9: ankle_b_FR1 (200, 2) - but dtype is uint8, might need conversion
        # Element 10: ankle_A_FR2 (200, 2, 2)
        # Element 11: ankle_b_FR2 (200, 2)
        
        time = cell_content[0].flatten()  # (200, 1) -> (200,)
        ankle_pos_FR1 = cell_content[3]   # (200, 2)
        ankle_pos_FR2 = cell_content[5]   # (200, 2)
        ankle_A_FR1 = cell_content[8]     # (200, 2, 2)
        ankle_b_FR1 = cell_content[9].astype(np.float64)  # (200, 2) - convert from uint8
        ankle_A_FR2 = cell_content[10]    # (200, 2, 2)
        ankle_b_FR2 = cell_content[11]    # (200, 2)
        
        print(f"Time points: {len(time)}")
        print(f"Ankle position FR1 shape: {ankle_pos_FR1.shape}")
        print(f"Ankle position FR2 shape: {ankle_pos_FR2.shape}")
        print(f"Transformation matrix A FR1 shape: {ankle_A_FR1.shape}")
        print(f"Translation vector b FR1 shape: {ankle_b_FR1.shape}")
        print(f"Transformation matrix A FR2 shape: {ankle_A_FR2.shape}")
        print(f"Translation vector b FR2 shape: {ankle_b_FR2.shape}")
        
        # Verify shapes match expected dimensions
        expected_shapes = {
            'time': (200,),
            'ankle_pos_FR1': (200, 2),
            'ankle_pos_FR2': (200, 2),
            'ankle_A_FR1': (200, 2, 2),
            'ankle_b_FR1': (200, 2),
            'ankle_A_FR2': (200, 2, 2),
            'ankle_b_FR2': (200, 2)
        }
        
        actual_shapes = {
            'time': time.shape,
            'ankle_pos_FR1': ankle_pos_FR1.shape,
            'ankle_pos_FR2': ankle_pos_FR2.shape,
            'ankle_A_FR1': ankle_A_FR1.shape,
            'ankle_b_FR1': ankle_b_FR1.shape,
            'ankle_A_FR2': ankle_A_FR2.shape,
            'ankle_b_FR2': ankle_b_FR2.shape
        }
        
        print("\nShape verification:")
        for key in expected_shapes:
            expected = expected_shapes[key]
            actual = actual_shapes[key]
            status = "✓" if expected == actual else "✗"
            print(f"  {key}: expected {expected}, got {actual} {status}")
        
        return time, ankle_pos_FR1, ankle_pos_FR2, ankle_A_FR1, ankle_b_FR1, ankle_A_FR2, ankle_b_FR2
        
    else:
        raise ValueError(f"Expected tuple in cell {cell_index}, got {type(cell_content)}")

def perform_inverse_transformations(ankle_pos_FR1, ankle_pos_FR2, ankle_A_FR1, ankle_b_FR1, ankle_A_FR2, ankle_b_FR2):
    """
    Perform inverse transformations for both reference frames
    """
    print("Performing inverse transformations...")
    
    n_points = ankle_pos_FR1.shape[0]
    ankle_pos_FR1_transformed = np.zeros((n_points, 2))
    ankle_pos_FR2_transformed = np.zeros((n_points, 2))
    
    print(f"Processing {n_points} time points...")
    
    # Perform inverse transformation for FR1
    for i in range(n_points):
        # Extract transformation matrix and translation vector for this time point
        # Handle different possible dimensions for A matrices
        if len(ankle_A_FR1.shape) == 3:
            A_FR1 = ankle_A_FR1[i, :, :]  # 2x2 matrix
        else:
            # If A is 2D, it might be stacked differently
            A_FR1 = ankle_A_FR1[i*2:(i+1)*2, :] if ankle_A_FR1.shape[0] > 2 else ankle_A_FR1
        
        b_FR1 = ankle_b_FR1[i, :].reshape(-1, 1)  # 2x1 vector
        
        # Original position in FR1
        pos_FR1 = ankle_pos_FR1[i, :].reshape(-1, 1)  # 2x1 vector
        
        print(f"Point {i}: A_FR1 shape: {A_FR1.shape}, b_FR1 shape: {b_FR1.shape}, pos_FR1 shape: {pos_FR1.shape}")
        
        # Apply inverse transformation: x_original = A^(-1) * (x_transformed - b)
        try:
            A_inv_FR1 = np.linalg.inv(A_FR1)
            ankle_pos_FR1_transformed[i, :] = (A_inv_FR1 @ (pos_FR1 - b_FR1)).flatten()
        except np.linalg.LinAlgError:
            # Handle singular matrix case
            warnings.warn(f"Singular matrix at time point {i} for FR1, using pseudo-inverse")
            A_pinv_FR1 = np.linalg.pinv(A_FR1)
            ankle_pos_FR1_transformed[i, :] = (A_pinv_FR1 @ (pos_FR1 - b_FR1)).flatten()
        except Exception as e:
            print(f"Error at point {i} for FR1: {e}")
            print(f"A_FR1: {A_FR1}")
            print(f"b_FR1: {b_FR1}")
            print(f"pos_FR1: {pos_FR1}")
            break
    
    # Perform inverse transformation for FR2
    for i in range(n_points):
        # Extract transformation matrix and translation vector for this time point
        # Handle different possible dimensions for A matrices
        if len(ankle_A_FR2.shape) == 3:
            A_FR2 = ankle_A_FR2[i, :, :]  # 2x2 matrix
        else:
            # If A is 2D, it might be stacked differently
            A_FR2 = ankle_A_FR2[i*2:(i+1)*2, :] if ankle_A_FR2.shape[0] > 2 else ankle_A_FR2
        
        b_FR2 = ankle_b_FR2[i, :].reshape(-1, 1)  # 2x1 vector
        
        # Original position in FR2
        pos_FR2 = ankle_pos_FR2[i, :].reshape(-1, 1)  # 2x1 vector
        
        # Apply inverse transformation: x_original = A^(-1) * (x_transformed - b)
        try:
            A_inv_FR2 = np.linalg.inv(A_FR2)
            ankle_pos_FR2_transformed[i, :] = (A_inv_FR2 @ (pos_FR2 - b_FR2)).flatten()
        except np.linalg.LinAlgError:
            # Handle singular matrix case
            warnings.warn(f"Singular matrix at time point {i} for FR2, using pseudo-inverse")
            A_pinv_FR2 = np.linalg.pinv(A_FR2)
            ankle_pos_FR2_transformed[i, :] = (A_pinv_FR2 @ (pos_FR2 - b_FR2)).flatten()
        except Exception as e:
            print(f"Error at point {i} for FR2: {e}")
            print(f"A_FR2: {A_FR2}")
            print(f"b_FR2: {b_FR2}")
            print(f"pos_FR2: {pos_FR2}")
            break
    
    print("Inverse transformations completed!")
    return ankle_pos_FR1_transformed, ankle_pos_FR2_transformed

def create_comparison_plots(time, pos_FR1_orig, pos_FR2_orig, pos_FR1_trans, pos_FR2_trans):
    """
    Create comprehensive comparison plots
    """
    print("Creating comparison plots...")
    
    # Create main comparison figure
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Gait Trajectory Comparison', fontsize=16, fontweight='bold')
    
    # Subplot 1: Original trajectories
    ax1 = plt.subplot(2, 3, 1)
    plt.plot(pos_FR1_orig[:, 0], pos_FR1_orig[:, 1], 'b-', linewidth=2, label='FR1 Original')
    plt.plot(pos_FR2_orig[:, 0], pos_FR2_orig[:, 1], 'r-', linewidth=2, label='FR2 Original')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Original Trajectories (Reference Frame Space)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Subplot 2: Transformed trajectories
    ax2 = plt.subplot(2, 3, 2)
    plt.plot(pos_FR1_trans[:, 0], pos_FR1_trans[:, 1], 'b-', linewidth=2, label='FR1 Transformed')
    plt.plot(pos_FR2_trans[:, 0], pos_FR2_trans[:, 1], 'r-', linewidth=2, label='FR2 Transformed')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Transformed Trajectories (Common Frame)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Subplot 3: Overlay comparison
    ax3 = plt.subplot(2, 3, 3)
    plt.plot(pos_FR1_trans[:, 0], pos_FR1_trans[:, 1], 'b-', linewidth=2, label='FR1 Transformed')
    plt.plot(pos_FR2_trans[:, 0], pos_FR2_trans[:, 1], 'r--', linewidth=2, label='FR2 Transformed')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Transformed Trajectories Overlay')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Subplot 4: X-component time series
    ax4 = plt.subplot(2, 3, 4)
    plt.plot(time, pos_FR1_trans[:, 0], 'b-', linewidth=2, label='FR1 X')
    plt.plot(time, pos_FR2_trans[:, 0], 'r--', linewidth=2, label='FR2 X')
    plt.xlabel('Time')
    plt.ylabel('X Position')
    plt.title('X-Component vs Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 5: Y-component time series
    ax5 = plt.subplot(2, 3, 5)
    plt.plot(time, pos_FR1_trans[:, 1], 'b-', linewidth=2, label='FR1 Y')
    plt.plot(time, pos_FR2_trans[:, 1], 'r--', linewidth=2, label='FR2 Y')
    plt.xlabel('Time')
    plt.ylabel('Y Position')
    plt.title('Y-Component vs Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 6: Difference between trajectories
    ax6 = plt.subplot(2, 3, 6)
    diff_x = pos_FR1_trans[:, 0] - pos_FR2_trans[:, 0]
    diff_y = pos_FR1_trans[:, 1] - pos_FR2_trans[:, 1]
    plt.plot(time, diff_x, 'g-', linewidth=2, label='X Difference')
    plt.plot(time, diff_y, 'm-', linewidth=2, label='Y Difference')
    plt.xlabel('Time')
    plt.ylabel('Position Difference')
    plt.title('Trajectory Differences (FR1 - FR2)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gait_trajectory_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create 3D trajectory comparison
    fig2 = plt.figure(figsize=(16, 8))
    fig2.suptitle('Gait Trajectory 3D Comparison', fontsize=16, fontweight='bold')
    
    # 3D trajectory plot
    ax_3d = fig2.add_subplot(1, 2, 1, projection='3d')
    ax_3d.plot(pos_FR1_trans[:, 0], pos_FR1_trans[:, 1], time, 'b-', linewidth=2, label='FR1')
    ax_3d.plot(pos_FR2_trans[:, 0], pos_FR2_trans[:, 1], time, 'r--', linewidth=2, label='FR2')
    ax_3d.set_xlabel('X Position')
    ax_3d.set_ylabel('Y Position')
    ax_3d.set_zlabel('Time')
    ax_3d.set_title('3D Trajectory Comparison')
    ax_3d.legend()
    
    # Euclidean distance plot
    ax_dist = fig2.add_subplot(1, 2, 2)
    euclidean_dist = np.sqrt(diff_x**2 + diff_y**2)
    plt.plot(time, euclidean_dist, 'k-', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Euclidean Distance')
    plt.title('Euclidean Distance Between Trajectories')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gait_trajectory_3d_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def calculate_trajectory_stats(pos_FR1_trans, pos_FR2_trans):
    """
    Calculate and display trajectory statistics
    """
    print("\n=== TRAJECTORY COMPARISON STATISTICS ===")
    
    # Calculate differences
    diff_x = pos_FR1_trans[:, 0] - pos_FR2_trans[:, 0]
    diff_y = pos_FR1_trans[:, 1] - pos_FR2_trans[:, 1]
    euclidean_dist = np.sqrt(diff_x**2 + diff_y**2)
    
    # X-component statistics
    print("X-Component Differences:")
    print(f"  Mean: {np.mean(diff_x):.4f}")
    print(f"  Std:  {np.std(diff_x):.4f}")
    print(f"  Max:  {np.max(np.abs(diff_x)):.4f}")
    
    # Y-component statistics
    print("Y-Component Differences:")
    print(f"  Mean: {np.mean(diff_y):.4f}")
    print(f"  Std:  {np.std(diff_y):.4f}")
    print(f"  Max:  {np.max(np.abs(diff_y)):.4f}")
    
    # Euclidean distance statistics
    print("Euclidean Distance:")
    print(f"  Mean: {np.mean(euclidean_dist):.4f}")
    print(f"  Std:  {np.std(euclidean_dist):.4f}")
    print(f"  Max:  {np.max(euclidean_dist):.4f}")
    print(f"  Min:  {np.min(euclidean_dist):.4f}")
    
    # Trajectory characteristics
    print("\nTrajectory Characteristics:")
    print("FR1 Trajectory:")
    print(f"  X Range: [{np.min(pos_FR1_trans[:, 0]):.4f}, {np.max(pos_FR1_trans[:, 0]):.4f}]")
    print(f"  Y Range: [{np.min(pos_FR1_trans[:, 1]):.4f}, {np.max(pos_FR1_trans[:, 1]):.4f}]")
    
    print("FR2 Trajectory:")
    print(f"  X Range: [{np.min(pos_FR2_trans[:, 0]):.4f}, {np.max(pos_FR2_trans[:, 0]):.4f}]")
    print(f"  Y Range: [{np.min(pos_FR2_trans[:, 1]):.4f}, {np.max(pos_FR2_trans[:, 1]):.4f}]")
    
    print("\n=== END STATISTICS ===")
    
    return {
        'diff_x': diff_x,
        'diff_y': diff_y,
        'euclidean_dist': euclidean_dist,
        'stats': {
            'x_mean': np.mean(diff_x),
            'x_std': np.std(diff_x),
            'x_max': np.max(np.abs(diff_x)),
            'y_mean': np.mean(diff_y),
            'y_std': np.std(diff_y),
            'y_max': np.max(np.abs(diff_y)),
            'euclidean_mean': np.mean(euclidean_dist),
            'euclidean_std': np.std(euclidean_dist),
            'euclidean_max': np.max(euclidean_dist),
            'euclidean_min': np.min(euclidean_dist)
        }
    }

def save_transformed_data(time, ankle_pos_FR1, ankle_pos_FR2, ankle_pos_FR1_transformed, ankle_pos_FR2_transformed):
    """
    Save the transformed data to a new .mat file
    """
    print("Saving transformed data...")
    
    data_dict = {
        'time': time,
        'ankle_pos_FR1_original': ankle_pos_FR1,
        'ankle_pos_FR2_original': ankle_pos_FR2,
        'ankle_pos_FR1_transformed': ankle_pos_FR1_transformed,
        'ankle_pos_FR2_transformed': ankle_pos_FR2_transformed
    }
    
    savemat('transformed_gait_data.mat', data_dict)
    print("Transformed data saved to 'transformed_gait_data.mat'")

def analyze_gait_data(filename='new_processed_gait_data#07.mat', cell_index=0):
    """
    Main function to analyze gait data
    """
    # Load data
    processed_data = load_gait_data(filename)
    if processed_data is None:
        return None
    
    print(f"\nAnalyzing cell {cell_index} of {processed_data.size} available cells")
    
    # Extract data fields from the specified cell
    time, ankle_pos_FR1, ankle_pos_FR2, ankle_A_FR1, ankle_b_FR1, ankle_A_FR2, ankle_b_FR2 = extract_data_fields(processed_data, cell_index)
    
    # Perform inverse transformations
    ankle_pos_FR1_transformed, ankle_pos_FR2_transformed = perform_inverse_transformations(
        ankle_pos_FR1, ankle_pos_FR2, ankle_A_FR1, ankle_b_FR1, ankle_A_FR2, ankle_b_FR2
    )
    
    # Create comparison plots
    create_comparison_plots(time, ankle_pos_FR1, ankle_pos_FR2, ankle_pos_FR1_transformed, ankle_pos_FR2_transformed)
    
    # Calculate and display statistics
    stats = calculate_trajectory_stats(ankle_pos_FR1_transformed, ankle_pos_FR2_transformed)
    
    # Save transformed data
    save_transformed_data(time, ankle_pos_FR1, ankle_pos_FR2, ankle_pos_FR1_transformed, ankle_pos_FR2_transformed)
    
    print(f"\nAnalysis completed successfully for cell {cell_index}!")
    
    return {
        'cell_index': cell_index,
        'time': time,
        'ankle_pos_FR1_original': ankle_pos_FR1,
        'ankle_pos_FR2_original': ankle_pos_FR2,
        'ankle_pos_FR1_transformed': ankle_pos_FR1_transformed,
        'ankle_pos_FR2_transformed': ankle_pos_FR2_transformed,
        'statistics': stats
    }

def analyze_all_cells(filename='new_processed_gait_data.mat'):
    """
    Analyze all cells in the processed_gait_data array
    """
    # Load data first to see how many cells we have
    processed_data = load_gait_data(filename)
    if processed_data is None:
        return None
    
    print(f"\nFound {processed_data.size} cells to analyze")
    
    all_results = []
    for i in range(processed_data.size):
        print(f"\n{'='*50}")
        print(f"ANALYZING CELL {i+1}/{processed_data.size}")
        print(f"{'='*50}")
        
        try:
            result = analyze_gait_data(filename, cell_index=i)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"Error analyzing cell {i}: {e}")
            continue
    
    return all_results

# Example usage
if __name__ == "__main__":
    # Option 1: Analyze just the first cell (default)
    print("Analyzing first cell...")
    results = analyze_gait_data()
    
    # Option 2: Analyze a specific cell (uncomment to use)
    # print("Analyzing cell 2...")
    # results = analyze_gait_data(cell_index=1)
    
    # Option 3: Analyze all cells (uncomment to use)
    # print("Analyzing all cells...")
    # all_results = analyze_all_cells()
    
    # Access results if needed
    if results:
        print("\nData analysis completed. Results are available in the 'results' dictionary.")
        print("Available keys:", list(results.keys()))