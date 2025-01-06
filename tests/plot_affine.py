"""
===========
Plot affine
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
# Display results
vreg.plot_affine_transformed(
    input_data, input_affine, 
    output_data, output_affine, 
    vreg.affine_matrix(rotation=rotation, translation=translation, pixel_spacing=stretch), 
    off_screen=OFF_SCREEN)
