"""
======================
Plot align translation
======================
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

input_data, input_affine, output_data, output_affine, translation = vreg.generate_translated_data_2()

#%% 
# Plot test data
# --------------
# The grey volume is the original shape, and the red surface represents the 
# same shape after translation and reslicing to the red slab.

vreg.plot_affine_transformed(
    input_data, input_affine, 
    output_data, output_affine, 
    vreg.affine_matrix(translation=translation), 
    off_screen=OFF_SCREEN)

#%% 
# Find the translation
# --------------------
# Our task is to find the translation that maps the grey volume 
# onto the red area.

translation_estimate = vreg.align(
    moving=input_data, moving_affine=input_affine, 
    static=output_data, static_affine=output_affine, 
    parameters=np.array([0, 0, 0], dtype=np.float32), 
    resolutions=[4,2,1], 
    transformation=vreg.translate, metric=vreg.sum_of_squares,
)

#%% 
# Check accuracy
# --------------
# Compare translation against ground truth

err = np.linalg.norm(translation_estimate-translation)
err = 100*err/np.linalg.norm(translation)
print('Ground truth parameter: ', translation)
print('Parameter estimate: ', translation_estimate)
print('Parameter error (%): ', err)


#%% 
# Display result
# --------------
# Translate the original volume

output_data_estimate = vreg.translate(
    input_data, input_affine, 
    input_data.shape, input_affine, 
    translation_estimate)

#%% 
# Compare translated volume and target

pl = vreg.plot_affine_resliced(
    output_data_estimate, input_affine, 
    output_data, output_affine, 
    off_screen=OFF_SCREEN)
pl.show()






        