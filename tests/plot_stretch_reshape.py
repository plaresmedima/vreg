"""
====================
Plot stretch reshape
====================
"""


import time
import numpy as np

import vreg

# Generate reference volume
input_data, input_affine = vreg.generate('triple ellipsoid', markers=False)

# Define transformation
stretch_factor = np.array([2.0, 2.5, 0.5])

# Perform transformation
start_time = time.time()
output_data, output_affine = vreg.stretch_reshape(input_data, input_affine, stretch_factor) # specifying outputshape and affine should not be required
end_time = time.time()

# Display results

print('Computation time (sec): ', end_time-start_time)
transformation = vreg.affine_matrix(pixel_spacing=stretch_factor)
vreg.plot_affine_transformed(input_data, input_affine, output_data, output_affine, transformation, off_screen=True)


