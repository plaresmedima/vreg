"""
==============
Align freeform
==============
"""

#%%
# Setup
import numpy as np
import vreg

# Set this to False to show the results
OFF_SCREEN = True

#%%
# Define geometry of source data
input_shape = np.array([300, 250, 25])   # mm
pixel_spacing = np.array([1.25, 1.25, 5.0]) # mm
translation = np.array([0, 0, 0]) # mm
rotation_angle = 0.2 * (np.pi/2) # radians
rotation_axis = [1,0,0]

#%%
# Generate source data
rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
input_affine = vreg.affine_matrix(rotation=rotation, translation=translation, pixel_spacing=pixel_spacing)
input_data, input_affine = vreg.generate('triple ellipsoid', shape=input_shape, affine=input_affine, markers=False)

#%%
# Define affine transformation
stretch = [1.0, 1.5, 1.5]
translation = np.array([30, -80, -20]) # mm
rotation_angle = 0.50 * (np.pi/2)
rotation_axis = [0,0,1]
rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)

#%%
# Define output_volume
output_shape = list(input_data.shape)
output_affine = input_affine.copy()

window=False
if window:
    output_shape[0] = 100
    output_shape[1] = 100
    output_affine[0,3] = output_affine[0,3] + 80
    output_affine[1,3] = output_affine[1,3] + 80
    output_affine[2,3] = output_affine[2,3] + 40

#%%
# Apply freeform deformation derived from affine transformation
affine_parameters = np.concatenate((rotation, translation, stretch))
output_data = vreg.affine_freeform(
    input_data, input_affine, 
    output_shape, output_affine, 
    affine_parameters, nodes=16)
    
#%%
# Display results
vreg.plot_affine_transformed(
    input_data, input_affine, 
    output_data, output_affine, 
    vreg.affine_matrix(rotation=rotation, translation=translation, pixel_spacing=stretch), 
    off_screen=OFF_SCREEN)


#%% 
# Find the freeform deformation
# -----------------------------
nodes = 2
defo_init = np.zeros(vreg.deformation_field_shape(output_shape, nodes))
defo_estimate = vreg.align(
    moving=input_data, moving_affine=input_affine, 
    static=output_data, static_affine=output_affine,  
    resolutions=[4,2,1], parameters=defo_init.flatten(),
    transformation=vreg.freeform, metric=vreg.sum_of_squares,
    options={'xtol':1e-2},
)

#%% 
# Check accuracy
# --------------
# Compare translation against ground truth
defo = vreg.affine_deformation_field(
    input_affine, output_shape, output_affine, affine_parameters, nodes=nodes)
defo = defo.flatten()
err = np.linalg.norm(defo_estimate-defo)
err = 100*err/np.linalg.norm(defo)
print('Ground truth parameter: ', defo)
print('Parameter estimate: ', defo_estimate)
print('Parameter error (%): ', err)

#%% 
# Display result
# --------------
# Rotate the original volume
output_data_estimate = vreg.freeform(
    input_data, input_affine, 
    input_data.shape, input_affine, 
    defo_estimate)

#%% 
# Compare rotated volume and target
pl = vreg.plot_affine_resliced(
    output_data_estimate, input_affine, 
    output_data, output_affine, 
    off_screen=OFF_SCREEN)
pl.show()


        