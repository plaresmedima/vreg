"""
==================
Plot rigid reshape
==================
"""


import time
import numpy as np

import vreg


# Generate input data
input_data, input_affine = vreg.generate('ellipsoid', markers=False)

# Define rigid transformation
angle = 0.5 * (np.pi/2)
axis = [1,0,0]
translation = np.array([0, 60, -40]) # mm
rotation = angle * np.array(axis)/np.linalg.norm(axis)

# Perform rigid transformation   
start_time = time.time()
output_data, output_affine = vreg.rigid_reshape(input_data, input_affine, rotation, translation)
end_time = time.time()

# Display results
print('Computation time (sec): ', end_time-start_time)
transformation = vreg.affine_matrix(rotation=rotation, translation=translation)
vreg.plot_affine_transformed(input_data, input_affine, output_data, output_affine, transformation, off_screen=True)
