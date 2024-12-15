"""
===========
Plot rotate
===========
"""

#%%
# Setup

import numpy as np
import vreg

# Set this to False to show the results
OFF_SCREEN = True

#%%
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
# Display results
vreg.plot_affine_transformed(
    input_data, input_affine, 
    output_data, output_affine, 
    vreg.affine_matrix(rotation=rotation), 
    off_screen=OFF_SCREEN)
