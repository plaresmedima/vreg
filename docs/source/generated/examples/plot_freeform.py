"""
=============
Plot freeform
=============
"""

#%%
# Setup
import numpy as np
import vreg

# Set this to False to show the results
OFF_SCREEN = True

#%%
# Define geometry of source data
input_shape = np.array([300, 250, 25])   # mm
pixel_spacing = np.array([1.25, 1.25, 5.0]) # mm
translation = np.array([0, 0, 0]) # mm
rotation_angle = 0.2 * (np.pi/2) # radians
rotation_axis = [1,0,0]

#%%
# Generate source data
rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
input_affine = vreg.affine_matrix(rotation=rotation, translation=translation, pixel_spacing=pixel_spacing)
input_data, input_affine = vreg.generate('triple ellipsoid', shape=input_shape, affine=input_affine, markers=False)

#%%
# Define affine transformation
stretch = [1.0, 1.5, 1.5]
translation = np.array([30, -80, -20]) # mm
rotation_angle = 0.50 * (np.pi/2)
rotation_axis = [0,0,1]
rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)

#%%
# Define output_volume
output_shape = list(input_data.shape)
output_affine = input_affine.copy()

window=False
if window:
    output_shape[0] = 100
    output_shape[1] = 100
    output_affine[0,3] = output_affine[0,3] + 80
    output_affine[1,3] = output_affine[1,3] + 80
    output_affine[2,3] = output_affine[2,3] + 40

#%%
# Get exact results for affine transformation
parameters = np.concatenate((rotation, translation, stretch))
exact_output_data = vreg.affine(input_data, input_affine, output_shape, output_affine, parameters)
affine_matrix = vreg.affine_matrix(rotation=rotation, translation=translation, pixel_spacing=stretch)

#%%
# Show affine transform
vreg.plot_affine_transformed(
    input_data, input_affine, 
    exact_output_data, output_affine, 
    affine_matrix, off_screen=OFF_SCREEN)

#%%
# Apply freeform deformation derived from affine transformation
output_data = vreg.affine_freeform(
        input_data, input_affine, 
        output_shape, output_affine, 
        parameters, nodes=16)
    
#%%
# Display results
error = np.linalg.norm(output_data-exact_output_data)/np.linalg.norm(exact_output_data)
print('Error (%): ', 100*error)
vreg.plot_affine_transformed(
    input_data, input_affine, 
    output_data, output_affine, 
    affine_matrix, 
    off_screen=OFF_SCREEN)



