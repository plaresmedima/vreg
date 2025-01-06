"""
============
Plot stretch
============
"""
#%%
# Setup
import vreg

# Set this to False to show the results
OFF_SCREEN = True 

#%%
# Generate reference volume
input_data, input_affine = vreg.generate('triple ellipsoid', markers=False)

#%%
# Define transformation
stretch_factor = [1.0, 1.5, 0.9]

#%%
# Define output geometry
output_shape = input_data.shape
output_affine = input_affine

#%%
# Perform stretch
output_data = vreg.stretch(
    input_data, input_affine, 
    output_shape, output_affine, 
    stretch_factor) 

#%%
# Display results
transformation = vreg.affine_matrix(pixel_spacing=stretch_factor)
vreg.plot_affine_transformed(
    input_data, input_affine, 
    output_data, output_affine, 
    transformation, off_screen=OFF_SCREEN)

