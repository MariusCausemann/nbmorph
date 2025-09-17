import numpy as np
import numba
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbmorph
numba.set_num_threads(1)

# Load the data
print("Loading data...")
img = np.load("data/dense_cells.npz")["arr_0"][:,:,19:20]
print(f"Data shape: {img.shape}, dtype: {img.dtype}")
#img = nbmorph.erode_labels_spherical(img, radius=1)
# Take a central slice for plotting
slice_index = 0
img_slice = img[:,:,slice_index].copy()

# Define the grid of parameters
radii = [1, 2, 3, 4]
iterations = [1, 2, 4, 8]

# Create a 4x4 plot
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
fig.suptitle('Effect of Morphological Smoothing', fontsize=16)

# Plot the original slice for reference in the first subplot
axes[0, 0].imshow(img_slice, cmap='viridis')
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')


for i, radius in enumerate(radii):
    for j, iteration in enumerate(iterations):
        print(f"Processing radius={radius}, iterations={iteration}...")
        
        # Apply morphological smoothing
        smoothed_img = nbmorph.smooth_labels_spherical(
            img, radius=radius, iterations=iteration)

        #smoothed_img = nbmorph.dilate_labels_spherical(smoothed_img, radius=5)
        #smoothed_img = nbmorph.erode_labels_spherical(smoothed_img, radius=1)

        # Get the same slice from the smoothed image
        smoothed_slice = smoothed_img[:,:,slice_index]
        
        # Plot the result
        ax = axes[i, j]
        cmap = plt.cm.PiYG
        cmap.set_under("black")
        ax.imshow(smoothed_slice, cmap=cmap, vmin=1, vmax=img.max())
        ax.set_title(f'Radius: {radius}, Iterations: {iteration}')
        #ax.axis('off')

# Adjust layout and save the plot
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('img/smoothing_effect.png')
print("Plot saved to smoothing_effect.png")
