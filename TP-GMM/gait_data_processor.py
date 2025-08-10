import scipy.io
import numpy as np
import json
import pandas as pd
from pathlib import Path
import h5py
import warnings

class GaitDataProcessor:
    def __init__(self, mat_file_path):
        """
        Initialize the processor with the path to the .mat file
        
        Args:
            mat_file_path (str): Path to the .mat file containing processed_gait_data
        """
        self.mat_file_path = Path(mat_file_path)
        self.processed_data = None
        self.demonstrations = []
        
    def load_mat_file(self):
        """
        Load the .mat file and extract processed_gait_data
        Handles both old format (.mat v7.2 and earlier) and new format (.mat v7.3+)
        """
        try:
            # Try loading with scipy.io first (works for .mat v7.2 and earlier)
            mat_data = scipy.io.loadmat(self.mat_file_path)
            self.processed_data = mat_data['processed_gait_data']
            print(f"Successfully loaded .mat file using scipy.io")
            
        except NotImplementedError:
            # If scipy.io fails, try h5py (for .mat v7.3+)
            print("scipy.io failed, trying h5py for .mat v7.3+ format...")
            try:
                with h5py.File(self.mat_file_path, 'r') as f:
                    self.processed_data = f['processed_gait_data']
                    print(f"Successfully loaded .mat file using h5py")
            except Exception as e:
                raise Exception(f"Failed to load .mat file with both scipy.io and h5py: {e}")
                
        except Exception as e:
            raise Exception(f"Error loading .mat file: {e}")
    
    def extract_demonstrations(self):
        """
        Extract individual demonstrations from the processed_gait_data array
        """
        if self.processed_data is None:
            raise ValueError("No data loaded. Call load_mat_file() first.")
        
        self.demonstrations = []
        
        # Handle different possible structures
        if isinstance(self.processed_data, np.ndarray):
            # If it's a numpy array of structs
            for i, demo in enumerate(self.processed_data.flatten()):
                demo_dict = self._extract_struct_data(demo, i)
                if demo_dict:
                    self.demonstrations.append(demo_dict)
        
        print(f"Extracted {len(self.demonstrations)} demonstrations")
        return self.demonstrations
    
    def _extract_struct_data(self, struct_data, demo_index):
        """
        Extract data from a single MATLAB struct
        
        Args:
            struct_data: MATLAB struct containing gait data
            demo_index (int): Index of the demonstration
            
        Returns:
            dict: Dictionary containing extracted data
        """
        try:
            # Initialize the demonstration dictionary
            demo_dict = {
                'demonstration_index': demo_index,
                'time': None,
                'pelvis_orientation': None,
                'ankle_pos_FR1': None,
                'ankle_pos_FR1_velocity': None,
                'ankle_orientation_FR1': None,
                'ankle_pos_FR2': None,
                'ankle_pos_FR2_velocity': None,
                'ankle_orientation_FR2': None,
                'ankle_A_FR1': None,
                'ankle_b_FR1': None,
                'ankle_A_FR2': None,
                'ankle_b_FR2': None
            }
            
            # Extract each field from the struct
            field_names = ['time', 'pelvis_orientation', 'ankle_pos_FR1', 
                          'ankle_pos_FR1_velocity', 'ankle_orientation_FR1',
                          'ankle_pos_FR2', 'ankle_pos_FR2_velocity', 
                          'ankle_orientation_FR2', 'ankle_A_FR1', 'ankle_b_FR1',
                          'ankle_A_FR2', 'ankle_b_FR2']
            
            # Handle different struct formats
            if hasattr(struct_data, 'dtype') and struct_data.dtype.names:
                # Structured array format
                for field in field_names:
                    if field in struct_data.dtype.names:
                        data = struct_data[field][0, 0] if struct_data[field].ndim > 1 else struct_data[field]
                        demo_dict[field] = np.array(data)
            else:
                # Try accessing as object array
                for i, field in enumerate(field_names):
                    try:
                        if hasattr(struct_data, '__getitem__') and len(struct_data) > i:
                            demo_dict[field] = np.array(struct_data[i])
                    except:
                        continue
            
            return demo_dict
            
        except Exception as e:
            print(f"Error extracting demonstration {demo_index}: {e}")
            return None
    
    def convert_to_json(self, output_path=None):
        """
        Convert the demonstrations to JSON format
        
        Args:
            output_path (str, optional): Path to save JSON file
            
        Returns:
            str: JSON string representation of the data
        """
        if not self.demonstrations:
            raise ValueError("No demonstrations extracted. Call extract_demonstrations() first.")
        
        # Convert numpy arrays to lists for JSON serialization
        json_data = []
        for demo in self.demonstrations:
            json_demo = {}
            for key, value in demo.items():
                if isinstance(value, np.ndarray):
                    json_demo[key] = value.tolist()
                else:
                    json_demo[key] = value
            json_data.append(json_demo)
        
        json_string = json.dumps(json_data, indent=2)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(json_string)
            print(f"JSON data saved to {output_path}")
        
        return json_string
    
    def get_demonstration_summary(self):
        """
        Get a summary of the loaded demonstrations
        """
        if not self.demonstrations:
            return "No demonstrations loaded"
        
        summary = f"Total demonstrations: {len(self.demonstrations)}\n"
        
        if self.demonstrations:
            demo = self.demonstrations[0]
            summary += "\nFirst demonstration structure:\n"
            for key, value in demo.items():
                if isinstance(value, np.ndarray):
                    summary += f"  {key}: shape {value.shape}, dtype {value.dtype}\n"
                else:
                    summary += f"  {key}: {type(value)}\n"
        
        return summary
    
    def get_demonstration(self, index):
        """
        Get a specific demonstration by index
        
        Args:
            index (int): Index of the demonstration to retrieve
            
        Returns:
            dict: Dictionary containing the demonstration data
        """
        if not self.demonstrations:
            raise ValueError("No demonstrations loaded")
        
        if index < 0 or index >= len(self.demonstrations):
            raise IndexError(f"Demonstration index {index} out of range (0-{len(self.demonstrations)-1})")
        
        return self.demonstrations[index]
    
    def plot_data_verification(self, demo_index=0, save_plots=False):
        """
        Comprehensive plotting to verify data was loaded correctly
        
        Args:
            demo_index (int): Index of demonstration to plot
            save_plots (bool): Whether to save plots as PNG files
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available. Install it with: pip install matplotlib")
            return
        
        if demo_index >= len(self.demonstrations):
            raise IndexError(f"Demo index {demo_index} out of range")
        
        demo = self.demonstrations[demo_index]
        time_data = demo['time']
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(f'Data Verification - Demonstration {demo_index}', fontsize=16)
        
        # 1. Time series plot
        ax1 = plt.subplot(4, 3, 1)
        if time_data is not None:
            plt.plot(time_data, 'b-', linewidth=2)
            plt.title('Time Vector')
            plt.xlabel('Sample Index')
            plt.ylabel('Time')
            plt.grid(True)
        
        # 2. Pelvis orientation
        ax2 = plt.subplot(4, 3, 2)
        if demo['pelvis_orientation'] is not None:
            plt.plot(time_data, demo['pelvis_orientation'], 'r-', linewidth=2)
            plt.title('Pelvis Orientation')
            plt.xlabel('Time')
            plt.ylabel('Angle (rad)')
            plt.grid(True)
        
        # 3. Ankle positions FR1 (Hip frame)
        ax3 = plt.subplot(4, 3, 3)
        if demo['ankle_pos_FR1'] is not None:
            pos_fr1 = demo['ankle_pos_FR1']
            plt.plot(time_data, pos_fr1[:, 0], 'g-', label='X', linewidth=2)
            plt.plot(time_data, pos_fr1[:, 1], 'b-', label='Y', linewidth=2)
            plt.title('Ankle Position FR1 (Hip Frame)')
            plt.xlabel('Time')
            plt.ylabel('Position')
            plt.legend()
            plt.grid(True)
        
        # 4. Ankle positions FR2 (Global frame)
        ax4 = plt.subplot(4, 3, 4)
        if demo['ankle_pos_FR2'] is not None:
            pos_fr2 = demo['ankle_pos_FR2']
            plt.plot(time_data, pos_fr2[:, 0], 'g-', label='X', linewidth=2)
            plt.plot(time_data, pos_fr2[:, 1], 'b-', label='Y', linewidth=2)
            plt.title('Ankle Position FR2 (Global Frame)')
            plt.xlabel('Time')
            plt.ylabel('Position')
            plt.legend()
            plt.grid(True)
        
        # 5. Ankle velocities FR1
        ax5 = plt.subplot(4, 3, 5)
        if demo['ankle_pos_FR1_velocity'] is not None:
            vel_fr1 = demo['ankle_pos_FR1_velocity']
            plt.plot(time_data, vel_fr1[:, 0], 'g-', label='Vx', linewidth=2)
            plt.plot(time_data, vel_fr1[:, 1], 'b-', label='Vy', linewidth=2)
            plt.title('Ankle Velocity FR1')
            plt.xlabel('Time')
            plt.ylabel('Velocity')
            plt.legend()
            plt.grid(True)
        
        # 6. Ankle velocities FR2
        ax6 = plt.subplot(4, 3, 6)
        if demo['ankle_pos_FR2_velocity'] is not None:
            vel_fr2 = demo['ankle_pos_FR2_velocity']
            plt.plot(time_data, vel_fr2[:, 0], 'g-', label='Vx', linewidth=2)
            plt.plot(time_data, vel_fr2[:, 1], 'b-', label='Vy', linewidth=2)
            plt.title('Ankle Velocity FR2')
            plt.xlabel('Time')
            plt.ylabel('Velocity')
            plt.legend()
            plt.grid(True)
        
        # 7. Ankle orientations
        ax7 = plt.subplot(4, 3, 7)
        if demo['ankle_orientation_FR1'] is not None:
            plt.plot(time_data, demo['ankle_orientation_FR1'], 'r-', label='FR1', linewidth=2)
        if demo['ankle_orientation_FR2'] is not None:
            plt.plot(time_data, demo['ankle_orientation_FR2'], 'b-', label='FR2', linewidth=2)
        plt.title('Ankle Orientations')
        plt.xlabel('Time')
        plt.ylabel('Angle (rad)')
        plt.legend()
        plt.grid(True)
        
        # 8. X-Y Trajectory comparison
        ax8 = plt.subplot(4, 3, 8)
        if demo['ankle_pos_FR1'] is not None and demo['ankle_pos_FR2'] is not None:
            pos_fr1 = demo['ankle_pos_FR1']
            pos_fr2 = demo['ankle_pos_FR2']
            plt.plot(pos_fr1[:, 0], pos_fr1[:, 1], 'g-', label='FR1 (Hip)', linewidth=2)
            plt.plot(pos_fr2[:, 0], pos_fr2[:, 1], 'b-', label='FR2 (Global)', linewidth=2)
            plt.title('X-Y Trajectories Comparison')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.legend()
            plt.grid(True)
            plt.axis('equal')
        
        # 9. Transformation matrix A determinants (to check validity)
        ax9 = plt.subplot(4, 3, 9)
        if demo['ankle_A_FR1'] is not None:
            A_fr1 = demo['ankle_A_FR1']
            det_fr1 = np.array([np.linalg.det(A_fr1[i]) for i in range(len(A_fr1))])
            plt.plot(time_data, det_fr1, 'g-', label='A_FR1 det', linewidth=2)
        if demo['ankle_A_FR2'] is not None:
            A_fr2 = demo['ankle_A_FR2']
            det_fr2 = np.array([np.linalg.det(A_fr2[i]) for i in range(len(A_fr2))])
            plt.plot(time_data, det_fr2, 'b-', label='A_FR2 det', linewidth=2)
        plt.title('Transformation Matrix Determinants')
        plt.xlabel('Time')
        plt.ylabel('Determinant')
        plt.legend()
        plt.grid(True)
        
        # 10. Translation vectors b
        ax10 = plt.subplot(4, 3, 10)
        if demo['ankle_b_FR1'] is not None:
            b_fr1 = demo['ankle_b_FR1']
            plt.plot(time_data, b_fr1[:, 0], 'g-', label='b_FR1_x', linewidth=2)
            plt.plot(time_data, b_fr1[:, 1], 'g--', label='b_FR1_y', linewidth=2)
        if demo['ankle_b_FR2'] is not None:
            b_fr2 = demo['ankle_b_FR2']
            plt.plot(time_data, b_fr2[:, 0], 'b-', label='b_FR2_x', linewidth=2)
            plt.plot(time_data, b_fr2[:, 1], 'b--', label='b_FR2_y', linewidth=2)
        plt.title('Translation Vectors b')
        plt.xlabel('Time')
        plt.ylabel('Translation')
        plt.legend()
        plt.grid(True)
        
                
        # 12. Velocity magnitude comparison
        ax12 = plt.subplot(4, 3, 12)
        if demo['ankle_pos_FR1_velocity'] is not None:
            vel_fr1 = demo['ankle_pos_FR1_velocity']
            vel_mag_fr1 = np.sqrt(vel_fr1[:, 0]**2 + vel_fr1[:, 1]**2)
            plt.plot(time_data, vel_mag_fr1, 'g-', label='FR1 Vel Mag', linewidth=2)
        if demo['ankle_pos_FR2_velocity'] is not None:
            vel_fr2 = demo['ankle_pos_FR2_velocity']
            vel_mag_fr2 = np.sqrt(vel_fr2[:, 0]**2 + vel_fr2[:, 1]**2)
            plt.plot(time_data, vel_mag_fr2, 'b-', label='FR2 Vel Mag', linewidth=2)
        plt.title('Velocity Magnitudes')
        plt.xlabel('Time')
        plt.ylabel('Velocity Magnitude')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'plots/gait_data_verification_demo_{demo_index}.png', dpi=300, bbox_inches='tight')
            print(f"Plot saved as gait_data_verification_demo_{demo_index}.png")
        
        plt.show()
        
        fig = plt.figure(figsize=(4, 6))
        fig.suptitle(f'Statidistics', fontsize=16)
        # 11. Data statistics summary
        stats_text = self._get_demo_statistics(demo)
        plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def _get_demo_statistics(self, demo):
        """Generate statistics text for a demonstration"""
        stats = []
        stats.append("Data Statistics:")
        stats.append("-" * 15)
        
        for key, value in demo.items():
            if key == 'demonstration_index':
                continue
            if value is not None and isinstance(value, np.ndarray):
                stats.append(f"{key}:")
                stats.append(f"  Shape: {value.shape}")
                stats.append(f"  Min: {value.min():.3f}")
                stats.append(f"  Max: {value.max():.3f}")
                stats.append(f"  Mean: {value.mean():.3f}")
                stats.append("")
        
        return "\n".join(stats)
    
    def plot_all_demonstrations_overview(self, max_demos=5, save_plots=False):
        """
        Plot overview of multiple demonstrations for comparison
        
        Args:
            max_demos (int): Maximum number of demonstrations to plot
            save_plots (bool): Whether to save plots as PNG files
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available. Install it with: pip install matplotlib")
            return
        
        if not self.demonstrations:
            print("No demonstrations available")
            return
        
        n_demos = min(len(self.demonstrations), max_demos)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Overview of {n_demos} Demonstrations', fontsize=16)
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_demos))
        
        for i in range(n_demos):
            demo = self.demonstrations[i]
            time_data = demo['time']
            color = colors[i]
            
            # Plot ankle positions FR2 (Global frame)
            if demo['ankle_pos_FR2'] is not None:
                pos_fr2 = demo['ankle_pos_FR2']
                axes[0, 0].plot(time_data, pos_fr2[:, 0], color=color, label=f'Demo {i}', alpha=0.7)
                axes[0, 1].plot(time_data, pos_fr2[:, 1], color=color, label=f'Demo {i}', alpha=0.7)
                axes[0, 2].plot(pos_fr2[:, 0], pos_fr2[:, 1], color=color, label=f'Demo {i}', alpha=0.7)
            
            # Plot velocities
            if demo['ankle_pos_FR2_velocity'] is not None:
                vel_fr2 = demo['ankle_pos_FR2_velocity']
                vel_mag = np.sqrt(vel_fr2[:, 0]**2 + vel_fr2[:, 1]**2)
                axes[1, 0].plot(time_data, vel_mag, color=color, label=f'Demo {i}', alpha=0.7)
            
            # Plot orientations
            if demo['ankle_orientation_FR2'] is not None:
                axes[1, 1].plot(time_data, demo['ankle_orientation_FR2'], color=color, label=f'Demo {i}', alpha=0.7)
            
            # Plot pelvis orientation
            if demo['pelvis_orientation'] is not None:
                axes[1, 2].plot(time_data, demo['pelvis_orientation'], color=color, label=f'Demo {i}', alpha=0.7)
        
        # Set titles and labels
        axes[0, 0].set_title('Ankle Position X (Global)')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Position X')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        axes[0, 1].set_title('Ankle Position Y (Global)')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Position Y')
        axes[0, 1].grid(True)
        
        axes[0, 2].set_title('X-Y Trajectories (Global)')
        axes[0, 2].set_xlabel('Position X')
        axes[0, 2].set_ylabel('Position Y')
        axes[0, 2].grid(True)
        axes[0, 2].axis('equal')
        
        axes[1, 0].set_title('Velocity Magnitude')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Velocity Magnitude')
        axes[1, 0].grid(True)
        
        axes[1, 1].set_title('Ankle Orientation (Global)')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Angle (rad)')
        axes[1, 1].grid(True)
        
        axes[1, 2].set_title('Pelvis Orientation')
        axes[1, 2].set_xlabel('Time')
        axes[1, 2].set_ylabel('Angle (rad)')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('plots/gait_data_overview.png', dpi=300, bbox_inches='tight')
            print("Overview plot saved as gait_data_overview.png")
        
        plt.show()
    
    def check_data_integrity(self):
        """
        Check data integrity and print detailed report
        """
        if not self.demonstrations:
            print("No demonstrations to check")
            return
        
        print("=" * 60)
        print("DATA INTEGRITY REPORT")
        print("=" * 60)
        
        expected_fields = ['time', 'pelvis_orientation', 'ankle_pos_FR1', 
                          'ankle_pos_FR1_velocity', 'ankle_orientation_FR1',
                          'ankle_pos_FR2', 'ankle_pos_FR2_velocity', 
                          'ankle_orientation_FR2', 'ankle_A_FR1', 'ankle_b_FR1',
                          'ankle_A_FR2', 'ankle_b_FR2']
        
        print(f"Total demonstrations: {len(self.demonstrations)}")
        print(f"Expected fields: {len(expected_fields)}")
        
        # Check each demonstration
        for i, demo in enumerate(self.demonstrations):
            print(f"\nDemonstration {i}:")
            print("-" * 20)
            
            missing_fields = []
            invalid_shapes = []
            
            for field in expected_fields:
                # print(f"Checking field: {field}")
                if field not in demo or demo[field] is None:
                    missing_fields.append(field)
                else:
                    data = demo[field]
                    # print(f"  Found data with shape: {data.shape if isinstance(data, np.ndarray) else type(data)}")
                    if isinstance(data, np.ndarray):
                        # Check expected shapes
                        if field == 'time' and data.shape != (200, 1):
                            invalid_shapes.append(f"{field}: {data.shape} (expected (200, 1))")
                        elif field == 'pelvis_orientation' and data.shape != (200, 1):
                            invalid_shapes.append(f"{field}: {data.shape} (expected (200, 1))")
                        elif 'pos_' in field and data.shape != (200, 2):
                            invalid_shapes.append(f"{field}: {data.shape} (expected (200, 2))")
                        elif 'velocity' in field and data.shape != (200, 2):
                            invalid_shapes.append(f"{field}: {data.shape} (expected (200, 2))")
                        elif 'orientation' in field and data.shape != (200, 1):
                            invalid_shapes.append(f"{field}: {data.shape} (expected (200, 1))")
                        elif 'ankle_A_' in field and data.shape != (200, 2, 2):
                            invalid_shapes.append(f"{field}: {data.shape} (expected (200, 2, 2))")
                        elif 'ankle_b_' in field and data.shape != (200, 2):
                            invalid_shapes.append(f"{field}: {data.shape} (expected (200, 2))")
            
            if missing_fields:
                print(f"  Missing fields: {missing_fields}")
            if invalid_shapes:
                print(f"  Invalid shapes: {invalid_shapes}")
            if not missing_fields and not invalid_shapes:
                print("  ✓ All fields present with correct shapes")
        
        print("\n" + "=" * 60)

# Example usage
def main():
    # Initialize the processor
    fpath = 'data/new_processed_gait_data#35_1_p50.mat'
    if not Path(fpath).exists():
        print(f"File {fpath} does not exist. Please check the path.")
        return
    processor = GaitDataProcessor(fpath)

    try:
        # Load the .mat file
        processor.load_mat_file()
        
        # Extract demonstrations
        demonstrations = processor.extract_demonstrations()
        
        # Print summary
        print(processor.get_demonstration_summary())
        
        # Check data integrity
        processor.check_data_integrity()
        
        # Plot comprehensive verification for first demonstration
        if demonstrations:
            print("\nGenerating comprehensive verification plots...")
            processor.plot_data_verification(0, save_plots=True)
            
            # Plot overview of multiple demonstrations
            print("\nGenerating overview plots...")
            processor.plot_all_demonstrations_overview(max_demos=5, save_plots=True)
            
            # Access individual demonstrations
            demo_0 = processor.get_demonstration(0)
            print(f"\nFirst demonstration keys: {list(demo_0.keys())}")
            
            # Convert to JSON (optional)
            json_output = processor.convert_to_json(f'{fpath[:-4]}.json')
            print("Data converted to JSON format")
        
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
