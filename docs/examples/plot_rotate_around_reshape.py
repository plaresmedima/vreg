"""
==========================
Plot rotate around reshape
==========================
"""

import time
import numpy as np

import vreg

# Generate reference volume
input_data, input_affine = vreg.generate('ellipsoid', markers=False)

# Define rotation
rotation = 0.5 * np.pi/2 * np.array([1, 0, 0]) # radians
com = vreg.center_of_mass(input_data, input_affine)

# Perform rotation
start_time = time.time()
output_data, output_affine = vreg.rotate_around_reshape(input_data, input_affine, rotation, com)
end_time = time.time()

# Display results
print('Computation time (sec): ', end_time-start_time)
transformation = vreg.affine_matrix(rotation=rotation, center=com)
vreg.plot_affine_transformed(input_data, input_affine, output_data, output_affine, transformation, off_screen=True)


        