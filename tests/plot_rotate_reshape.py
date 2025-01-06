"""
===================
Plot rotate reshape
===================
"""


import time
import numpy as np

import vreg

# Generate reference volume
input_data, input_affine = vreg.generate('triple ellipsoid')

# Define rotation
angle = 0.1 * (np.pi/2)
axis = [1,0,0]
rotation = angle * np.array(axis)/np.linalg.norm(axis)

# Perform rotation
start_time = time.time()
output_data, output_affine = vreg.rotate_reshape(input_data, input_affine, rotation) # not logical that this returns the output_affine
end_time = time.time()

# Display results
print('Computation time (sec): ', end_time-start_time)
transformation = vreg.affine_matrix(rotation=rotation)
vreg.plot_affine_transformed(input_data, input_affine, output_data, output_affine, transformation, off_screen=True)

