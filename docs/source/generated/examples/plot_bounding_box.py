"""
=================
Plot bounding box
=================
"""

import numpy as np

import vreg



# Define geometry of source data
input_shape = np.array([300, 250, 12])   # mm
pixel_spacing = np.array([1.25, 1.25, 10]) # mm
rotation_angle = 0.5 * (np.pi/2) # radians
rotation_axis = [1,0,0]
translation = np.array([0, -40, 180]) # mm
margin = 10 # mm

# Generate reference volume
rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
input_affine = vreg.affine_matrix(rotation=rotation, translation=translation, pixel_spacing=pixel_spacing)
#input_data, input_affine = vreg.fake.generate('double ellipsoid', shape=input_shape, affine=input_affine, markers=False)
input_data, input_affine = vreg.generate('triple ellipsoid', shape=input_shape, affine=input_affine, markers=False)

# Perform translation with reshaping
output_data, output_affine = vreg.bounding_box(input_data, input_affine, margin=margin)

# Display results
vreg.plot_bounding_box(input_data, input_affine, output_data.shape, output_affine, off_screen=True)

