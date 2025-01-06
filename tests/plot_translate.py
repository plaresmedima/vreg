"""
==============
Plot translate
==============
"""

#%%
# Setup
import vreg

# Set this to False to show the results
OFF_SCREEN = True  

#%%
# Generate original and translated data
n=2
if n==1:
    input_data, input_affine, output_data, output_affine, translation = vreg.generate_translated_data_1()
elif n==2:
    input_data, input_affine, output_data, output_affine, translation = vreg.generate_translated_data_2()
elif n==3:
    input_data, input_affine, output_data, output_affine, translation = vreg.generate_translated_data_3()

#%%
# Display results
transformation = vreg.affine_matrix(translation=translation)
vreg.plot_affine_transform_reslice(
    input_data, input_affine, 
    output_data, output_affine, 
    transformation, off_screen=OFF_SCREEN)


