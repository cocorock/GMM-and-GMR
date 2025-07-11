import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.io

# Constants
L1 = 0.4135  # Femur length (m)
L2 = 0.39    # Tibia length (m)

def load_gait_data(filename):
    """Load gait data from MATLAB file"""
    try:
        mat_data = scipy.io.loadmat(filename)
        output_struct_array = mat_data['output_struct_array']
        return output_struct_array
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def forward_kinematics(hip_angle_deg, knee_angle_deg, l1=L1, l2=L2):
    """
    Calculate forward kinematics for a 2-link leg
    
    Parameters:
    hip_angle_deg: Hip joint angle (degrees)
    knee_angle_deg: Knee joint angle (degrees)
    l1: Femur length
    l2: Tibia length
    
    Returns:
    hip_pos: Hip position (origin at 0,0)
    knee_pos: Knee position
    ankle_pos: Ankle position
    """
    # Convert degrees to radians for calculations
    hip_angle = np.radians(hip_angle_deg)
    knee_angle = np.radians(knee_angle_deg)
    
    # Hip is at origin
    hip_pos = np.array([0, 0])
    
    # Knee position
    knee_x = l1 * np.cos(hip_angle)
    knee_y = l1 * np.sin(hip_angle)
    knee_pos = np.array([knee_x, knee_y])
    
    # Ankle position (knee angle is relative to femur)
    ankle_x = knee_x + l2 * np.cos(hip_angle + knee_angle)
    ankle_y = knee_y + l2 * np.sin(hip_angle + knee_angle)
    ankle_pos = np.array([ankle_x, ankle_y])
    
    return hip_pos, knee_pos, ankle_pos

def calculate_euler_trajectory(hip_angles_deg, knee_angles_deg):
    """Calculate the trajectory of the ankle (end effector) using forward kinematics"""
    trajectory = []
    
    for i in range(len(hip_angles_deg)):
        _, _, ankle_pos = forward_kinematics(hip_angles_deg[i], knee_angles_deg[i])
        trajectory.append(ankle_pos)
    
    return np.array(trajectory)

def plot_gait_analysis(gait_data, sample_idx=0):
    """
    Plot gait analysis including trajectory and animated leg segments
    
    Parameters:
    gait_data: Loaded gait data structure
    sample_idx: Index of the gait sample to analyze
    """
    if gait_data is None:
        print("No gait data available")
        return
    
    # Extract data for the specified sample
    sample = gait_data[0, sample_idx]
    hip_pos_raw = sample['hip_pos'][0, 0].flatten()
    knee_pos_raw = sample['knee_pos'][0, 0].flatten()
    hip_vel_raw = sample['hip_vel'][0, 0].flatten()
    knee_vel_raw = sample['knee_vel'][0, 0].flatten()
    
    # Process hip position: invert signal and subtract 90 degrees
    hip_pos = hip_pos_raw - 90
    
    # Process knee position and velocity: invert signals
    knee_pos = -knee_pos_raw
    hip_vel = hip_vel_raw
    knee_vel = -knee_vel_raw
    
    print(f"Analyzing gait sample {sample_idx + 1}")
    print(f"Data points: {len(hip_pos)}")
    print(f"Hip angle range (processed): {np.min(hip_pos):.1f} to {np.max(hip_pos):.1f} degrees")
    print(f"Hip angle range (original): {np.min(hip_pos_raw):.1f} to {np.max(hip_pos_raw):.1f} degrees")
    print(f"Knee angle range (processed): {np.min(knee_pos):.1f} to {np.max(knee_pos):.1f} degrees")
    print(f"Knee angle range (original): {np.min(knee_pos_raw):.1f} to {np.max(knee_pos_raw):.1f} degrees")
    
    # Calculate ankle trajectory (angles are already in degrees)
    ankle_trajectory = calculate_euler_trajectory(hip_pos, knee_pos)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Joint angles over time
    ax1 = plt.subplot(2, 3, 1)
    time_points = np.linspace(0, 100, len(hip_pos))  # Assuming 100% gait cycle
    plt.plot(time_points, hip_pos, 'b-', label='Hip Angle (Processed)', linewidth=2)
    plt.plot(time_points, knee_pos, 'r-', label='Knee Angle (Processed)', linewidth=2)
    plt.xlabel('Gait Cycle (%)')
    plt.ylabel('Angle (degrees)')
    plt.title('Joint Angles vs Gait Cycle\n(Hip: inverted and -90°, Knee: inverted)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Joint velocities over time
    ax2 = plt.subplot(2, 3, 2)
    plt.plot(time_points, hip_vel, 'b-', label='Hip Velocity (Processed)', linewidth=2)
    plt.plot(time_points, knee_vel, 'r-', label='Knee Velocity (Processed)', linewidth=2)
    plt.xlabel('Gait Cycle (%)')
    plt.ylabel('Angular Velocity (deg/s)')
    plt.title('Joint Velocities vs Gait Cycle\n(Both signals inverted)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Ankle trajectory (Euler trajectory)
    ax3 = plt.subplot(2, 3, 3)
    plt.plot(ankle_trajectory[:, 0], ankle_trajectory[:, 1], 'g-', linewidth=2, label='Ankle Trajectory')
    plt.scatter(ankle_trajectory[0, 0], ankle_trajectory[0, 1], color='green', s=100, marker='o', label='Start')
    plt.scatter(ankle_trajectory[-1, 0], ankle_trajectory[-1, 1], color='red', s=100, marker='x', label='End')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Ankle Trajectory (Forward Kinematics)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Plot 4: Leg configuration animation
    ax4 = plt.subplot(2, 3, (4, 6))
    
    # Plot every 20th point for leg segments
    step_size = 20
    indices = range(0, len(hip_pos), step_size)
    
    # Set up the plot limits
    x_min, x_max = np.min(ankle_trajectory[:, 0]) - 0.1, np.max(ankle_trajectory[:, 0]) + 0.1
    y_min, y_max = np.min(ankle_trajectory[:, 1]) - 0.1, np.max(ankle_trajectory[:, 1]) + 0.1
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    # Plot ankle trajectory as background
    plt.plot(ankle_trajectory[:, 0], ankle_trajectory[:, 1], 'g--', alpha=0.5, linewidth=1, label='Ankle Path')
    
    # Plot leg segments for selected points
    colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))
    
    for i, idx in enumerate(indices):
        hip, knee, ankle = forward_kinematics(hip_pos[idx], knee_pos[idx])
        
        # Plot femur (hip to knee)
        plt.plot([hip[0], knee[0]], [hip[1], knee[1]], 
                color=colors[i], linewidth=3, alpha=0.7)
        
        # Plot tibia (knee to ankle)
        plt.plot([knee[0], ankle[0]], [knee[1], ankle[1]], 
                color=colors[i], linewidth=3, alpha=0.7)
        
        # Plot joints
        plt.plot(hip[0], hip[1], 'ko', markersize=8)  # Hip
        plt.plot(knee[0], knee[1], 'ro', markersize=6)  # Knee
        plt.plot(ankle[0], ankle[1], 'go', markersize=4)  # Ankle
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title(f'Leg Segments Every {step_size} Points\n(Femur: {L1}m, Tibia: {L2}m)')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend()
    
    # Add colorbar to show progression
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                              norm=plt.Normalize(vmin=0, vmax=100))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax4, shrink=0.8)
    cbar.set_label('Gait Cycle (%)')
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print(f"\nTrajectory Statistics:")
    print(f"Max X displacement: {np.max(ankle_trajectory[:, 0]):.4f} m")
    print(f"Min X displacement: {np.min(ankle_trajectory[:, 0]):.4f} m")
    print(f"X Range: {np.max(ankle_trajectory[:, 0]) - np.min(ankle_trajectory[:, 0]):.4f} m")
    print(f"Max Y displacement: {np.max(ankle_trajectory[:, 1]):.4f} m")
    print(f"Min Y displacement: {np.min(ankle_trajectory[:, 1]):.4f} m")
    print(f"Y Range: {np.max(ankle_trajectory[:, 1]) - np.min(ankle_trajectory[:, 1]):.4f} m")

def analyze_all_samples(gait_data):
    """Analyze all gait samples and plot overlaid trajectories"""
    if gait_data is None:
        return
    
    plt.figure(figsize=(12, 8))
    
    num_samples = gait_data.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, num_samples))
    
    for i in range(num_samples):
        sample = gait_data[0, i]
        hip_pos_raw = sample['hip_pos'][0, 0].flatten()
        knee_pos_raw = sample['knee_pos'][0, 0].flatten()
        
        # Process hip position: invert signal and subtract 90 degrees
        hip_pos = hip_pos_raw - 90
        
        # Process knee position: invert signal
        knee_pos = -knee_pos_raw
        
        # Calculate ankle trajectory (angles are already in degrees)
        ankle_trajectory = calculate_euler_trajectory(hip_pos, knee_pos)
        
        plt.plot(ankle_trajectory[:, 0], ankle_trajectory[:, 1], 
                color=colors[i], linewidth=2, alpha=0.7, label=f'Sample {i+1}')
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title(f'Ankle Trajectories - All {num_samples} Gait Samples')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load the gait data
    filename = "demo_gait_data_angular_10_samples.mat"
    gait_data = load_gait_data(filename)
    
    if gait_data is not None:
        # Analyze the first sample in detail
        plot_gait_analysis(gait_data, sample_idx=0)
        
        # Analyze all samples together
        analyze_all_samples(gait_data)
        
        # You can analyze other samples by changing the sample_idx
        # For example, to analyze the second sample:
        # plot_gait_analysis(gait_data, sample_idx=1)
    else:
        print("Could not load gait data. Please check the file path and format.")
