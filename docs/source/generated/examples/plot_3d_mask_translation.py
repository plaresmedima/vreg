"""
===================================
3D translation: 3D image to 3D mask
===================================
"""

#%%
# Setup
import vreg


# Set this to False to show the results
OFF_SCREEN = True


#%%
# Get data

#mask = vreg.fetch('left_kidney')
mask = vreg.fetch('right_kidney')
slab = vreg.fetch('MTR')

#%%
# Reslice mask to the MTR image

mask_slab, _ = vreg.affine_reslice(mask[0], mask[1], slab[1], 
                                output_shape=slab[0].shape)
vreg.plot_overlay_2d(slab[0], mask_slab, title='Original data', 
                     off_screen=OFF_SCREEN)

#%%
# 3D translation in slice coords

grid = (
    [-20, 20, 20],
    [-20, 20, 20],
    [-5, 5, 5],
) 
translation = vreg.align(
    moving=mask[0], moving_affine=mask[1],
    static=slab[0], static_affine=slab[1],
    transformation=vreg.translate_passive_ortho,
    metric=vreg.mutual_information,
    optimize='brute', options={'grid':grid},
) 
# Apply the translation that we found
mask_slab = vreg.translate_passive_ortho(mask[0], mask[1], slab[0].shape, 
                                         slab[1], translation) 
# Plot the result
vreg.plot_overlay_2d(slab[0], mask_slab, title='3D translation')



