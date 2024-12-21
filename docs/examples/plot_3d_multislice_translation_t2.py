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
dixon = vreg.fetch('Dixon_water')
lk = vreg.fetch('left_kidney')
rk = vreg.fetch('right_kidney')
multislice = vreg.fetch('T2star')
#multislice = vreg.fetch('T1')

mask = lk.slice_like(dixon).add(rk)
#dixon = dixon.bounding_box(mask, margin=20)


#%%
# 3D translation in slice coords

grid = (
    [-20, 20, 20],
    [-20, 20, 20],
    [-5, 5, 5],
) 
ms_trans = [
    #oneslice.translate_to(
    oneslice.coreg_to(
        mask, optimize='brute', return_params=True,
        options={
            'grid':grid, 
            'desc':'Translating ' + str(z+1) + ' out of ' + str(len(multislice))
        },
    ) for z, oneslice in enumerate(multislice)
]

vols = [v[0] for v in ms_trans]
translation = [v[1] for v in ms_trans]

# Plot the result
vreg.plot_overlay_2d(
    [v.values for v in vols], 
    [mask.slice_like(v).values for v in vols],  
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

ms_trans = [
    oneslice.coreg_to(
        lk, optimize='brute', transform=vreg.rigid_passive_com_ortho,
        options={
            'grid':grids[z], 
            'desc':'Transforming ' + str(z+1) + ' out of ' + str(len(multislice))
        },
    ) for z, oneslice in enumerate(multislice)
]

# Plot the result
vreg.plot_overlay_2d(
    [v.values for v in ms_trans], 
    [lk.slice_like(v).values for v in ms_trans],  
    title='3D translation')

#%%
# Fine tune right kidney with a rigid transformation

ms_trans = [
    oneslice.coreg_to(
        rk, optimize='brute', transform=vreg.rigid_passive_com_ortho,
        options={
            'grid':grids[z], 
            'desc':'Transforming ' + str(z+1) + ' out of ' + str(len(multislice))
        },
    ) for z, oneslice in enumerate(multislice)
]


# Plot the result
vreg.plot_overlay_2d(
    [v.values for v in ms_trans], 
    [rk.slice_like(v).values for v in ms_trans],  
    title='3D translation')




