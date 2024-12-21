"""
==============================================
3D translation: multislice 2D image to 3D mask
==============================================
"""

#%%
# Setup
import vreg
from tqdm import tqdm


# Set this to False to show the results
OFF_SCREEN = True


#%%
# Get data

dixon = vreg.Volume(*vreg.fetch('Dixon_water'))
lk = vreg.fetch('left_kidney')
rk = vreg.fetch('right_kidney')
lk = vreg.affine_reslice(lk[0], lk[1], dixon[1], output_shape=dixon[0].shape)
rk = vreg.affine_reslice(rk[0], rk[1], dixon[1], output_shape=dixon[0].shape)
mask = (lk[0]+rk[0], dixon[1])
dixon = vreg.mask_volume(dixon[0], dixon[1], mask[0], mask[1], 20)
multislice = vreg.fetch('T2star')
#multislice = vreg.fetch('T1')

#%%
# Reslice left kidney ROI to the T1-map (slice 0)

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

exit()


#%%
# 3D translation in slice coords

grid = (
    [-20, 20, 20],
    [-20, 20, 20],
    [-5, 5, 5],
) 
translation = [
    vreg.align(
        moving=mask[0], moving_affine=mask[1],
        #moving=dixon[0], moving_affine=dixon[1],
        static=oneslice[0], static_affine=oneslice[1],
        transformation=vreg.translate_passive_ortho,
        metric=vreg.mutual_information,
        optimize='brute', 
        options={
            'grid':grid, 
            'desc':'Translating ' + str(z) + ' out of ' + str(len(multislice))
        },
    ) for z, oneslice in enumerate(multislice)
]
# Apply the translation that we found
mask_slices = [
    vreg.translate_passive_ortho(mask[0], mask[1], oneslice[0].shape, 
                                 oneslice[1], translation[z]) 
    for z, oneslice in enumerate(multislice)
]
# Plot the result
vreg.plot_overlay_2d([v[0] for v in multislice], mask_slices, 
                     title='3D translation')


#%%
# Fine tune left kidney with a rigid transformation

grids = [
    (
        [-0.2, 0.2, 4],
        [-0.2, 0.2, 4],
        [-0.2, 0.2, 4],
        [t[0]-2, t[0]+2, 4],
        [t[1]-2, t[1]+2, 4],
        [t[2]-2, t[2]+2, 4],
    ) 
for t in translation]

# bounds = [
#     (
#         [g[0] for g in grid],
#         [g[1] for g in grid],
#     )
# for grid in grids]

# Find the transformation
params = [
    vreg.align(
        moving=lk[0], moving_affine=lk[1],
        #moving=dixon[0], moving_affine=dixon[1],
        static=oneslice[0], static_affine=oneslice[1],
        transformation=vreg.rigid_passive_com_ortho,
        metric=vreg.mutual_information,
        # optimize='LS',
        # parameters=[0]*3 + list(translation[z]),
        # options={
        #     'bounds': bounds[z],
        #     'abs_step': [0.01]*3+[0.1]*3,
        # },
        optimize='brute', 
        options={
            'grid':grids[z],
            'desc':'Transforming ' + str(z) + ' out of ' + str(len(multislice))
        },
    ) for z, oneslice in tqdm(enumerate(multislice), desc='Left kidney alignment')
]
# Apply the transformation to the mask
mask_slices = [
    vreg.rigid_passive_com_ortho(lk[0], lk[1], oneslice[0].shape, 
                                 oneslice[1], params[z]) 
    for z, oneslice in enumerate(multislice)
]
# Plot the result
vreg.plot_overlay_2d([v[0] for v in multislice], mask_slices, 
                     title='3D rigid')

#%%
# Fine tune right kidney with a rigid transformation

# Find the transformation
params = [
    vreg.align(
        moving=rk[0], moving_affine=rk[1],
        #moving=dixon[0], moving_affine=dixon[1],
        static=oneslice[0], static_affine=oneslice[1],
        transformation=vreg.rigid_passive_com_ortho,
        metric=vreg.mutual_information,
        # parameters=[0]*3 + list(translation[z]),
        # optimize='LS',
        # options={
        #     'bounds': bounds[z],
        #     'abs_step': [0.01]*3+[0.1]*3,
        # },
        optimize='brute', 
        options={
            'grid':grids[z],
            'desc':'Transforming ' + str(z) + ' out of ' + str(len(multislice))
        },
    ) for z, oneslice in tqdm(enumerate(multislice), desc='Right kidney alignment')
]
# Apply the transformation to the mask
mask_slices = [
    vreg.rigid_passive_com_ortho(rk[0], rk[1], oneslice[0].shape, 
                                 oneslice[1], params[z]) 
    for z, oneslice in enumerate(multislice)
]
# Plot the result
vreg.plot_overlay_2d([v[0] for v in multislice], mask_slices, 
                     title='3D rigid')



