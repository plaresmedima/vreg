"""
==========
Plot rigid
==========
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
# Define rigid transformation
angle = 0.1 * (np.pi/2)
axis = [1,1,0]
translation = np.array([0, 20, -20]) # mm
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
# Display results
vreg.plot_affine_transformed(
    input_data, input_affine, 
    output_data, output_affine, 
    vreg.affine_matrix(rotation=rotation, translation=translation), 
    off_screen=OFF_SCREEN)

