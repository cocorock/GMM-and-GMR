
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt

def load_mat_to_dataframe(file_path):
    """
    Loads a .mat file and converts its content to a pandas DataFrame.
    Assumes the .mat file contains a single array that can be directly converted.
    """
    mat_data = scipy.io.loadmat(file_path)
    data_key = [key for key in mat_data if not key.startswith('__')][0]
    data_array = mat_data[data_key]
    return pd.DataFrame(data_array)

# --- Main Script ---
file_path = "C:/Users/quepe/Documents/GitHub/GMM-and-GMR/right_leg_linear_kinematics.mat"
df = load_mat_to_dataframe(file_path)

# Assuming the rows are:
# Row 0: time
# Row 1: right ankle x position
# Row 2: right ankle y position
# Row 3: left ankle x position
# Row 4: left ankle y position
# Row 5: right ankle x velocity
# Row 6: right ankle y velocity
# Row 7: left ankle x velocity
# Row 8: left ankle y velocity

# Extract data for plotting
# Positions
right_ankle_pos_x = df.iloc[1, :]
right_ankle_pos_y = df.iloc[3, :]
left_ankle_pos_x = df.iloc[2, :]
left_ankle_pos_y = df.iloc[4, :]

# Velocities
right_ankle_vel_x = df.iloc[5, :]
right_ankle_vel_y = df.iloc[7, :]
left_ankle_vel_x = df.iloc[6, :]
left_ankle_vel_y = df.iloc[8, :]

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Ankle Position and Velocity Trajectories (X vs Y)', fontsize=16)

# Plot Right Ankle Position
axes[0, 0].plot(right_ankle_pos_x, right_ankle_pos_y, label='Right Ankle Position')
axes[0, 0].set_title('Right Ankle Position (X vs Y)')
axes[0, 0].set_xlabel('X Position')
axes[0, 0].set_ylabel('Y Position')
axes[0, 0].grid(True)
axes[0, 0].set_aspect('equal', adjustable='box')

# Plot Left Ankle Position
axes[0, 1].plot(left_ankle_pos_x, left_ankle_pos_y, color='orange', label='Left Ankle Position')
axes[0, 1].set_title('Left Ankle Position (X vs Y)')
axes[0, 1].set_xlabel('X Position')
axes[0, 1].set_ylabel('Y Position')
axes[0, 1].grid(True)
axes[0, 1].set_aspect('equal', adjustable='box')

# Plot Right Ankle Velocity
axes[1, 0].plot(right_ankle_vel_x, right_ankle_vel_y, color='green', label='Right Ankle Velocity')
axes[1, 0].set_title('Right Ankle Velocity (X vs Y)')
axes[1, 0].set_xlabel('X Velocity')
axes[1, 0].set_ylabel('Y Velocity')
axes[1, 0].grid(True)
axes[1, 0].set_aspect('equal', adjustable='box')

# Plot Left Ankle Velocity
axes[1, 1].plot(left_ankle_vel_x, left_ankle_vel_y, color='red', label='Left Ankle Velocity')
axes[1, 1].set_title('Left Ankle Velocity (X vs Y)')
axes[1, 1].set_xlabel('X Velocity')
axes[1, 1].set_ylabel('Y Velocity')
axes[1, 1].grid(True)
axes[1, 1].set_aspect('equal', adjustable='box')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('left_leg_linear_kinematics.png')
plt.show()
