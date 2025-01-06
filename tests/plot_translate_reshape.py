"""
======================
Plot translate reshape
======================
"""


import time
import numpy as np
import vreg




# Define geometry of source data
input_shape = np.array([300, 250, 12])   # mm
pixel_spacing = np.array([1.25, 1.25, 10.0]) # mm
rotation_angle = 0.0 * (np.pi/2) # radians
rotation_axis = [1,0,0]
translation = np.array([0, 0, 0]) # mm

# Generate reference volume
rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
input_affine = vreg.affine_matrix(rotation=rotation, translation=translation, pixel_spacing=pixel_spacing)
input_data, input_affine = vreg.generate('triple ellipsoid', shape=input_shape, affine=input_affine)

# Perform translation with reshaping
translation = np.array([0, -30, 0]) # mm
start_time = time.time()
ouput_data, output_affine = vreg.translate_reshape(input_data, input_affine, translation)
end_time = time.time()

# Display results
print('Computation time (sec): ', end_time-start_time)
transformation = vreg.affine_matrix(translation=translation)
vreg.plot_affine_transformed(input_data, input_affine, ouput_data, output_affine, transformation, off_screen=True)
