"""
==================
Plot align stretch
==================
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
# Define transformation
stretch_factor = [1.0, 1.1, 0.9]

#%%
# Define output geometry
output_shape = input_data.shape
output_affine = input_affine

#%%
# Stretch and reslice reference volume
output_data = vreg.stretch(
    input_data, input_affine, 
    output_shape, output_affine, 
    stretch_factor) 

#%% 
# Plot test data
# --------------
# The grey volume is the original shape, and the red surface represents the 
# same shape after stretching and reslicing to the red slab.

vreg.plot_affine_transformed(
    input_data, input_affine, 
    output_data, output_affine, 
    vreg.affine_matrix(pixel_spacing=stretch_factor), 
    off_screen=OFF_SCREEN)

#%% 
# Find the stretch factor
# -----------------------

stretch_factor_estimate = vreg.align(
    moving=input_data, moving_affine=input_affine, 
    static=output_data, static_affine=output_affine, 
    parameters=np.array([1, 1, 1], dtype=np.float32), 
    resolutions=[4,2,1], transformation=vreg.stretch,
    metric=vreg.sum_of_squares,
    options={'bounds':(0, np.inf)},
)

#%% 
# Check accuracy
# --------------
# Compare translation against ground truth

err = np.linalg.norm(stretch_factor_estimate-stretch_factor)
err = 100*err/np.linalg.norm(stretch_factor)
print('Ground truth parameter: ', stretch_factor)
print('Parameter estimate: ', stretch_factor_estimate)
print('Parameter error (%): ', err)


#%% 
# Display result
# --------------
# Stretch the original volume

output_data_estimate = vreg.stretch(
    input_data, input_affine, 
    input_data.shape, input_affine, 
    stretch_factor_estimate)

#%% 
# Compare stretched volume and target

pl = vreg.plot_affine_resliced(
    output_data_estimate, input_affine, 
    output_data, output_affine, 
    off_screen=OFF_SCREEN)
pl.show()

