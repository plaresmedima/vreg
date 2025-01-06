"""
===================
Plot align rotation
===================
"""

#%%
# Setup
# -----
# Import packages and get test data

import numpy as np
import vreg

# Set this to False to show the results
OFF_SCREEN = True

#%%
# Generate test data
# ------------------
# Generate reference volume
input_data, input_affine = vreg.generate('triple ellipsoid', markers=False)

#%%
# Define rotation
angle = 0.2 * (np.pi/2)
axis = [1,1,0]
rotation = angle * np.array(axis)/np.linalg.norm(axis)

#%%
# Define output geometry
output_shape = input_data.shape
output_affine = input_affine

#%%
# Perform rotation
output_data = vreg.rotate(
    input_data, input_affine, 
    output_shape, output_affine, 
    rotation)

#%% 
# Plot test data
# --------------
# The grey volume is the original shape, and the red surface represents the 
# same shape after rotating.

vreg.plot_affine_transformed(
    input_data, input_affine, 
    output_data, output_affine, 
    vreg.affine_matrix(rotation=rotation), 
    off_screen=OFF_SCREEN)

#%% 
# Find the rotation
# -----------------
rotation_estimate = vreg.align(
    moving=input_data, moving_affine=input_affine, 
    static=output_data, static_affine=output_affine, 
    parameters=np.array([0, 0, 0], dtype=np.float32), 
    resolutions=[4,2,1], transformation=vreg.rotate,
    metric=vreg.sum_of_squares,
)
#%% 
# Check accuracy
# --------------
# Compare translation against ground truth
err = np.linalg.norm(rotation_estimate-rotation)
err = 100*err/np.linalg.norm(rotation)
print('Ground truth parameter: ', rotation)
print('Parameter estimate: ', rotation_estimate)
print('Parameter error (%): ', err)

#%% 
# Display result
# --------------
# Rotate the original volume
output_data_estimate = vreg.rotate(
    input_data, input_affine, 
    input_data.shape, input_affine, 
    rotation_estimate)

#%% 
# Compare rotated volume and target
pl = vreg.plot_affine_resliced(
    output_data_estimate, input_affine, 
    output_data, output_affine, 
    off_screen=OFF_SCREEN)
pl.show()

