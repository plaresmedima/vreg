"""
==================
Plot rotate around
==================
"""

import time
import numpy as np

import vreg


# Generate reference volume
input_data, input_affine = vreg.generate('ellipsoid', markers=False)

# Define rotation
rotation = 0.5 * np.pi/2 * np.array([1, 0, 0]) # radians
com = vreg.center_of_mass(input_data, input_affine)

# Define output_volume
output_shape = input_data.shape
output_affine = input_affine

# Perform rotation
start_time = time.time()
parameters = np.concatenate((rotation, com))
output_data = vreg.rotate_around(input_data, input_affine, output_shape, output_affine, parameters)
end_time = time.time()

# Display results
print('Computation time (sec): ', end_time-start_time)
transformation = vreg.affine_matrix(rotation=rotation, center=com)
vreg.plot_affine_transformed(input_data, input_affine, output_data, output_affine, transformation, off_screen=True)

