"""
=====================
Plot affine transform
=====================
"""

import time
import numpy as np
import vreg


# Define geometry of source data
input_shape = np.array([300, 250, 25])   # mm
pixel_spacing = np.array([1.25, 1.25, 5.0]) # mm
translation = np.array([0, 0, 0]) # mm
rotation_angle = 0.2 * (np.pi/2) # radians
rotation_axis = [1,0,0]

# Generate source volume data
rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
input_affine = vreg.affine_matrix(rotation=rotation, translation=translation, pixel_spacing=pixel_spacing)
input_data, input_affine = vreg.generate('triple ellipsoid', shape=input_shape, affine=input_affine)

# Define affine transformation
stretch = [1.0, 1, 2.0]
translation = np.array([0, 20, 0]) # mm
rotation_angle = 0.20 * (np.pi/2)
rotation_axis = [0,0,1]

# Perform affine transformation
start_time = time.time()
rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
transformation = vreg.affine_matrix(rotation=rotation, translation=translation, pixel_spacing=stretch)
output_data, output_affine = vreg.affine_transform(input_data, input_affine, transformation)
end_time = time.time()

# Display results
print('Computation time (sec): ', end_time-start_time)
vreg.plot_affine_transformed(input_data, input_affine, output_data, output_affine, transformation, off_screen=True)

