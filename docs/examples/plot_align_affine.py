"""
=================
Plot align affine
=================
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
# Define affine transformation
stretch = [1.0, 1.5, 1.5]
translation = np.array([30, -80, -20]) # mm
rotation_angle = 0.20 * (np.pi/2)
rotation_axis = [0,0,1]
rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
parameters = np.concatenate((rotation, translation, stretch))

#%%
# Define output_geometry
output_shape = input_data.shape
output_affine = input_affine

#%%
# Apply affine
output_data = vreg.affine(
    input_data, input_affine, 
    output_shape, output_affine, 
    parameters)

#%% 
# Plot test data
# --------------
# The grey volume is the original shape, and the red surface represents the 
# same shape after affine transformation.

vreg.plot_affine_transformed(
    input_data, input_affine, 
    output_data, output_affine, 
    vreg.affine_matrix(rotation=rotation, translation=translation, pixel_spacing=stretch), 
    off_screen=OFF_SCREEN)

#%% 
# Find the affine transformation
# ------------------------------
bounds = (
    [-np.inf] * 6 + [0] * 3, 
    np.inf,
)
parameters_estimate = vreg.align(
    moving=input_data, moving_affine=input_affine, 
    static=output_data, static_affine=output_affine, 
    parameters=np.array([0] * 6 + [1] * 3, dtype=np.float32), 
    resolutions=[4,2,1], transformation=vreg.affine,
    metric=vreg.sum_of_squares,
    options={'bounds':bounds},
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
output_data_estimate = vreg.affine(
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


