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

dixon = vreg.fetch('Dixon_water')
lk = vreg.fetch('left_kidney')
rk = vreg.fetch('right_kidney')
mtr = vreg.fetch('MTR')
mask = lk.slice_like(dixon).add(rk)

#%%
# Reslice mask to the MTR image
#mtr = mtr.translate([20,40,0])

vreg.plot_overlay_2d(mtr.values, 
                     mask.slice_like(mtr).values, 
                     title='Original data', 
                     off_screen=OFF_SCREEN)

#%%
# 3D translation in slice coords

grid = (
    [-20, 20, 20],
    [-20, 20, 20],
    [-5, 5, 5],
) 

mtr = mtr.coreg_to(mask, optimize='brute', options={'grid':grid})

vreg.plot_overlay_2d(mtr.values, 
                     mask.slice_like(mtr).values, 
                     title='3D translation')



