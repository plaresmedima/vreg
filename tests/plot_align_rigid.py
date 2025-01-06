"""
================
Plot align rigid
================
"""

#%%
# Setup
# -----
# Import packages and get test data

import numpy as np
import vreg

# Set this to False to show the results
OFF_SCREEN = False


#%%
# Generate test data
# ------------------
# Generate reference volume
input_data, input_affine = vreg.generate('triple ellipsoid', markers=False)

#%%
# Define rigid transformation
angle = 0.1 * (np.pi/2)
axis = [1,1,0]
translation = np.array([0, 10, -10]) # mm
rotation = angle * np.array(axis)/np.linalg.norm(axis)
parameters = np.concatenate((rotation, translation))

#%%
# Define output_geometry
output_shape = input_data.shape
output_affine = input_affine

#%%
# Perform rigid transformation   
output_data = vreg.rigid(
    input_data, input_affine, 
    output_shape, output_affine, 
    parameters)

#%% 
# Plot test data
# --------------
# The grey volume is the original shape, and the red surface represents the 
# same shape after rotating.

vreg.plot_affine_transformed(
    input_data, input_affine, 
    output_data, output_affine, 
    vreg.affine_matrix(rotation=rotation, translation=translation), 
    off_screen=OFF_SCREEN)

#%% 
# Find the rigid transformation
# -----------------------------
parameters_estimate = vreg.align(
    moving=input_data, moving_affine=input_affine, 
    static=output_data, static_affine=output_affine, 
    parameters=np.zeros(6, dtype=np.float32), 
    resolutions=[4,2,1], transformation=vreg.rigid,
    metric=vreg.sum_of_squares,
)

#%% 
# Check accuracy
# --------------
# Compare translation against ground truth
err = np.linalg.norm(parameters_estimate-parameters)
err = 100*err/np.linalg.norm(parameters)
print('Ground truth parameter: ', parameters)
print('Parameter estimate: ', parameters_estimate)
print('Parameter error (%): ', err)

#%% 
# Display result
# --------------
# Rotate the original volume
output_data_estimate = vreg.rigid(
    input_data, input_affine, 
    input_data.shape, input_affine, 
    parameters_estimate)

#%% 
# Compare rotated volume and target
pl = vreg.plot_affine_resliced(
    output_data_estimate, input_affine, 
    output_data, output_affine, 
    off_screen=OFF_SCREEN)
pl.show()

