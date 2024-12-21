"""
=======================================================
2D in-slice translation: 2D multislice image to 3D mask
=======================================================
"""

#%%
# Setup
import vreg


# Set this to False to show the results
OFF_SCREEN = True


#%%
# Get data

#dixon = vreg.fetch('Dixon_water')
mask = vreg.fetch('left_kidney')
#mask = vreg.fetch('right_kidney')
multislice = vreg.fetch('T1')

#%%
# Reslice left kidney ROI to the multislice image (slice 0)

mask_slices = [
    vreg.affine_reslice(
        mask[0], mask[1], oneslice[1], output_shape=oneslice[0].shape,
    ) for oneslice in multislice
]
vreg.plot_overlay_2d(
    [v[0] for v in multislice], 
    [v[0] for v in mask_slices],
    title='Original data',
    off_screen=OFF_SCREEN,
)

#%% 
# Let's see if we can find the optimal translation automatically by 
# fitting for the translation vector. 
translation = [
    vreg.align(
        moving=mask[0], moving_affine=mask[1],
        static=oneslice[0], static_affine=oneslice[1],
        transformation=vreg.translate_passive_inslice,
        metric=vreg.mi_grad, 
        #metric=vreg.mutual_information, 
        optimize='brute', options={'grid':2*[[-20,20,20]]},
    ) for oneslice in multislice
]
# Apply the translation that we found
mask_slices = [
    vreg.translate_passive_inslice(
        mask[0], mask[1], oneslice[0].shape, oneslice[1], translation[z],
    ) for z, oneslice in enumerate(multislice)
]
# Plot the result
vreg.plot_overlay_2d([v[0] for v in multislice], mask_slices, 
                     title='In-slice translation')





