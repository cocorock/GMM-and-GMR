import json
import numpy as np
from tpgmm.tpgmm.tpgmm import TPGMM
from tpgmm.gmr.gmr import GaussianMixtureRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

# Create a directory for plots if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

# Loading gait data json
with open("data/new_processed_gait_data#39_16.json", "r") as f:
    data = json.load(f)

# Trajectories manipulation
# Create separate lists for trajectories from each frame of reference
trajectories_fr1 = []
trajectories_fr2 = []

for demo in data:
    # For FR1, stack spatial features first, then time
    traj1 = np.hstack([
        demo['ankle_pos_FR1'],
        demo['ankle_pos_FR1_velocity'],
        demo['ankle_orientation_FR1'],
        demo['time']
    ])
    trajectories_fr1.append(traj1)

    # For FR2, stack spatial features first, then time
    traj2 = np.hstack([
        demo['ankle_pos_FR2'],
        demo['ankle_pos_FR2_velocity'],
        demo['ankle_orientation_FR2'],
        demo['time']
    ])
    trajectories_fr2.append(traj2)

# Convert lists to numpy arrays
trajectories_fr1 = np.array(trajectories_fr1)
trajectories_fr2 = np.array(trajectories_fr2)

# --- Plotting ---
# Create a figure with two subplots, side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot trajectories for Frame of Reference 1
ax1.set_title('Trajectories in Frame of Reference 1')
ax1.set_xlabel('X Position')
ax1.set_ylabel('Y Position')
for traj in trajectories_fr1:
    # The first two columns are x and y ankle positions
    x_pos = traj[:, 0]
    y_pos = traj[:, 1]
    ax1.plot(x_pos, y_pos, marker='.', linestyle='-', markersize=2)
ax1.grid(True)
ax1.axis('equal')

# Plot trajectories for Frame of Reference 2
ax2.set_title('Trajectories in Frame of Reference 2')
ax2.set_xlabel('X Position')
ax2.set_ylabel('Y Position')
for traj in trajectories_fr2:
    # The first two columns are x and y ankle positions
    x_pos = traj[:, 0]
    y_pos = traj[:, 1]
    ax2.plot(x_pos, y_pos, marker='.', linestyle='-', markersize=2)
ax2.grid(True)
ax2.axis('equal')

# Save the plot
fig.tight_layout()
plt.savefig('plots/trajectories_fr1_fr2.png')
plt.close(fig)

# Create the figure and subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plot X vs Time
axs[0].plot(trajectories_fr1[0][:, -1],  trajectories_fr1[0][:, 0], label='FR1', color='blue')
axs[0].plot(trajectories_fr2[0][:, -1],  trajectories_fr2[0][:, 0], label='FR2', color='orange')
axs[0].set_title('X vs Time')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('X')
axs[0].legend()
axs[0].grid()

# Plot Y vs Time
axs[1].plot(trajectories_fr1[0][:, -1],  trajectories_fr1[0][:, 1], label='FR1', color='blue')
axs[1].plot(trajectories_fr2[0][:, -1],  trajectories_fr2[0][:, 1], label='FR2', color='orange')
axs[1].set_title('Y vs Time')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Y')
axs[1].legend()
axs[1].grid()

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('plots/xy_vs_time.png')
plt.close(fig)

# Plot velocity trajectories
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot trajectories for Frame of Reference 1
ax1.set_title('Velocity trajectories in Frame of Reference 1')
ax1.set_xlabel('X Velocity')
ax1.set_ylabel('Y Velocity')
for traj in trajectories_fr1:
    x_pos = traj[:, 2]
    y_pos = traj[:, 3]
    ax1.plot(x_pos, y_pos, marker='.', linestyle='-', markersize=2)
ax1.grid(True)
ax1.axis('equal')

# Plot trajectories for Frame of Reference 2
ax2.set_title('Trajectories in Frame of Reference 2')
ax2.set_xlabel('X Velocity')
ax2.set_ylabel('Y Velocity')
for traj in trajectories_fr2:
    x_pos = traj[:, 2]
    y_pos = traj[:, 3]
    ax2.plot(x_pos, y_pos, marker='.', linestyle='-', markersize=2)
ax2.grid(True)
ax2.axis('equal')

# Save the plots
plt.tight_layout()
plt.savefig('plots/velocity_trajectories.png')
plt.close(fig)

# Plot velocity vs time
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Plot X vs Time
axs[0].plot(trajectories_fr1[0][:, -1],  trajectories_fr1[0][:, 2], label='FR1', color='blue')
axs[0].plot(trajectories_fr2[0][:, -1],  trajectories_fr2[0][:, 2], label='FR2', color='orange')
axs[0].set_title('X vel vs Time')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('X vel')
axs[0].legend()
axs[0].grid()

# Plot Y vs Time
axs[1].plot(trajectories_fr1[0][:, -1],  trajectories_fr1[0][:, 3], label='FR1', color='blue')
axs[1].plot(trajectories_fr2[0][:, -1],  trajectories_fr2[0][:, 3], label='FR2', color='orange')
axs[1].set_title('Y vel vs Time')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Yvvel')
axs[1].legend()
axs[1].grid()

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('plots/velocity_vs_time.png')
plt.close(fig)

# Plot orientation trajectories
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot trajectories for Frame of Reference 1
ax1.set_title('Orientation trajectories in Frame of Reference 1')
ax1.set_xlabel('X Orientation')
ax1.set_ylabel('Y Orientation')
for traj in trajectories_fr1:
    orient = traj[:, 4]
    time = traj[:, 5]
    ax1.plot(time, orient, marker='.', linestyle='-', markersize=2)
ax1.grid(True)

# Plot trajectories for Frame of Reference 2
ax2.set_title('Trajectories in Frame of Reference 2')
ax2.set_xlabel('X Orientation')
ax2.set_ylabel('Y Orientation')
for traj in trajectories_fr2:
    orient = traj[:, 4]
    time = traj[:, 5]
    ax2.plot(time, orient, marker='.', linestyle='-', markersize=2)
ax2.grid(True)

# Save the plots
plt.tight_layout()
plt.savefig('plots/orientation_trajectories.png')
plt.close(fig)

# Plot orientation vs time
fig, ax = plt.subplots(figsize=(8, 6))

# Plot X vs Time
ax.plot(trajectories_fr1[0][:, -1],  np.degrees(trajectories_fr1[0][:, 4]), label='FR1', color='blue')
ax.plot(trajectories_fr2[0][:, -1],  np.degrees(trajectories_fr2[0][:, 4]), label='FR2', color='orange')
ax.set_title('Orientation vs Time')
ax.set_xlabel('Time (s)')
ax.set_ylabel('degrees')
ax.legend()
ax.grid()

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('plots/orientation_vs_time.png')
plt.close(fig)

# TPGMM
# Reshape the data for TPGMM
num_trajectories, num_samples, num_features = trajectories_fr1.shape
reshaped_trajectories = np.stack([trajectories_fr1, trajectories_fr2], axis=0)
reshaped_trajectories = reshaped_trajectories.reshape(2, num_trajectories * num_samples, num_features)
print(f"reshaped trajectories shape: {reshaped_trajectories.shape}")

# Fitting TPGMM
best_n_components = None
lowest_bic_score = float('inf')

# Loop through n_components from 5 to 30
for n_components in range(5, 31):
    print(f'Fitting TPGMM with n_components={n_components}...')
    tpgmm = TPGMM(n_components=n_components, verbose=False, reg_factor=1e-5)
    print('Data check passed. Fitting the model...')
    tpgmm.fit(reshaped_trajectories)
    print('BIC calculation...')
    bic_score = tpgmm.bic(reshaped_trajectories)
    print(f'BIC score for n_components={n_components}: {bic_score}')
    if bic_score < lowest_bic_score:
        lowest_bic_score = bic_score
        best_n_components = n_components

print(f'\nBest n_components: {best_n_components}')
print(f'Lowest BIC score: {lowest_bic_score}')

print(f'\nDefining TPGMM with the optimal n_components={best_n_components}...')
tpgmm = TPGMM(n_components=best_n_components, verbose=False, reg_factor=1e-5)
print('Data check passed. Fitting the model...')
tpgmm.fit(reshaped_trajectories)

# Second Part - GMR
# === Step 2: Properly extract reference frames data ===
sample_trajectory_idx = 6
sample_trajectory = data[sample_trajectory_idx]

traj_fr1 = np.hstack([
    sample_trajectory['ankle_pos_FR1'],
    sample_trajectory['ankle_pos_FR1_velocity'],
    sample_trajectory['ankle_orientation_FR1'],
    sample_trajectory['time']
])

traj_fr2 = np.hstack([
    sample_trajectory['ankle_pos_FR2'],
    sample_trajectory['ankle_pos_FR2_velocity'],
    sample_trajectory['ankle_orientation_FR2'],
    sample_trajectory['time']
])
print(f"Trajectory shape: {traj_fr1.shape}")

# === Step 2:  extract reference frames data ===
num_timesteps = traj_fr1.shape[0]

# === Step 3: Create GMR instance ===
gmr = GaussianMixtureRegression.from_tpgmm(tpgmm, input_idx=[0, 1])
print(f"TPGMM means shape: {tpgmm.means_.shape}")
print(f"TPGMM covariances shape: {tpgmm.covariances_.shape}")
print(f"GMR input dimensions: {gmr.input_idx}")
print(f"GMR output dimensions: {gmr.output_idx}")

# === Step 5: Try GMR fitting ===
identity_translation = np.zeros((1, 4))
identity_translation = identity_translation.repeat(2, axis=0)
identity_rotation = np.array([np.eye(4)])
identity_rotation = identity_rotation.repeat(2, axis=0)
print(f"identity_translation shape: {identity_translation.shape}")
print(f"identity_rotation shape: {identity_rotation.shape}")
gmr.fit(translation=identity_translation, rotation_matrix=identity_rotation)
print("-"*50)

gmr_transformed = GaussianMixtureRegression.from_tpgmm(tpgmm, input_idx=[0, 1])
angle_transform = np.radians(15)
rot_2d_transform = np.array([
    [np.cos(angle_transform), -np.sin(angle_transform)],
    [np.sin(angle_transform), np.cos(angle_transform)]
])
rot_4d_transform = np.eye(4)
rot_4d_transform[0:2, 0:2] = rot_2d_transform
print(f"rot_4d_transform shape: \n {rot_4d_transform}")
transform_translation = np.array([[0, 0, 0, 0], [0, 0, angle_transform, 0]])
transform_rotation = np.array([np.eye(4), rot_4d_transform])
gmr_transformed.fit(translation=transform_translation, rotation_matrix=transform_rotation)

# === Step 6: Plot the obtained trajectory ===
input_positions = traj_fr1[:, :2]
reference_vel = traj_fr1[:, 2:4]
reference_ori = traj_fr1[:, 4]
ref_time = traj_fr1[:, 5]

pred_original_mean, pred_original_cov = gmr.predict(input_positions)
pred_transformed_mean, pred_transformed_cov = gmr_transformed.predict(input_positions)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(ref_time, reference_vel[: , 0], 'k--', label='referece', linewidth=2)
axes[0, 0].plot(ref_time, pred_original_mean[:, 0], 'b--', label='GMR Original X_vel', alpha=0.7)
axes[0, 0].plot(ref_time, pred_transformed_mean[:, 0], 'r--', label='GMR Transformed X_vel', alpha=0.7)
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('X Velocity')
axes[0, 0].set_title('X Velocity Prediction')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(ref_time,reference_vel[: , 1], 'k--', label='referece', linewidth=2)
axes[0, 1].plot(ref_time, pred_original_mean[:, 1], 'b--', label='GMR Original Y_vel', alpha=0.7)
axes[0, 1].plot(ref_time, pred_transformed_mean[:, 1], 'r--', label='GMR Transformed Y_vel', alpha=0.7)
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('Y Velocity')
axes[0, 1].set_title('Y Velocity Prediction')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(input_positions[:, 0], input_positions[:, 1], 'k--', label='Input Trajectory', linewidth=2)
axes[1, 0].set_xlabel('X Position')
axes[1, 0].set_ylabel('Y Position')
axes[1, 0].set_title('Input X-Y Trajectory')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axis('equal')

axes[1, 1].plot(ref_time, pred_original_mean[:, 2], 'b-', label='GMR Original Orientation', alpha=0.7)
axes[1, 1].plot(ref_time, pred_transformed_mean[:, 2], 'r-', label='GMR Transformed Orientation', alpha=0.7)
axes[1, 1].plot(ref_time, reference_ori, 'k--', label='Reference Orientation', alpha=0.7)
axes[1, 1].set_xlabel('Time')
axes[1, 1].set_ylabel('Orientation')
axes[1, 1].set_title('Orientation Prediction')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/gmr_predictions.png')
plt.close(fig)

# Plot uncertainty
print(f"\nPrediction shapes:")
print(f"Input shape: {input_positions.shape}")
print(f"Original prediction mean shape: {pred_original_mean.shape}")
print(f"Transformed prediction mean shape: {pred_transformed_mean.shape}")

fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
std_original = np.sqrt(np.diagonal(pred_original_cov, axis1=1, axis2=2))
std_transformed = np.sqrt(np.diagonal(pred_transformed_cov, axis1=1, axis2=2))
time = ref_time
ax.plot(time, pred_original_mean[:, 0], 'b-', label='Original X_vel')
ax.fill_between(time,
                pred_original_mean[:, 0] - std_original[:, 0],
                pred_original_mean[:, 0] + std_original[:, 0],
                alpha=0.3, color='blue')
ax.plot(time, pred_transformed_mean[:, 0], 'r-', label='Transformed X_vel')
ax.fill_between(time,
                pred_transformed_mean[:, 0] - std_transformed[:, 0],
                pred_transformed_mean[:, 0] + std_transformed[:, 0],
                alpha=0.3, color='red')
ax.set_xlabel('Time')
ax.set_ylabel('X Velocity')
ax.set_title('X Velocity Predictions with Uncertainty')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('plots/gmr_uncertainty.png')
plt.close(fig2)

print("Script finished. Plots are saved in the 'plots' directory.")
