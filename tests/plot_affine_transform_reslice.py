"""
=============================
Plot affine transform reslice
=============================
"""

import numpy as np
import vreg



# Define geometry of input data
input_shape = np.array([400, 300, 120])   # mm
pixel_spacing = np.array([1.0, 1.0, 1.0]) # mm
translation = np.array([0, 0, 0]) # mm
rotation_angle = -0.2 * (np.pi/2) # radians
rotation_axis = [1,0,0]
rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
input_affine = vreg.affine_matrix(rotation=rotation, translation=translation, pixel_spacing=pixel_spacing)

# Define geometry of output data
output_shape = np.array([350, 300, 10])
pixel_spacing = np.array([1.25, 1.25, 5.0]) # mm
translation = np.array([100, -30, -40]) # mm
rotation_angle = 0.0 * (np.pi/2) # radians
rotation_axis = [1,0,0]
rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
output_affine = vreg.affine_matrix(rotation=rotation, translation=translation, pixel_spacing=pixel_spacing)

# Define affine transformation
stretch = [1.25, 1, 1.0]
translation = np.array([20, 0, 0]) # mm
rotation_angle = 0.1 * (np.pi/2)
rotation_axis = [1,0,0]
rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
transformation = vreg.affine_matrix(rotation=rotation, translation=translation, pixel_spacing=stretch)

# Generate input data
input_data, input_affine = vreg.generate('triple ellipsoid', shape=input_shape, affine=input_affine)

# Calculate affine transform
output_data = vreg.affine_transform_and_reslice(input_data, input_affine, output_shape, output_affine, transformation)

# Display results
vreg.plot_affine_transform_reslice(input_data, input_affine, output_data, output_affine, transformation, off_screen=True)

