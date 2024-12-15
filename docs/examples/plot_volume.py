"""
===========
Plot volume
===========
"""

# sphinx_gallery_start_ignore
# PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT = True
# sphinx_gallery_end_ignore

# %%
import vreg

n=1
if n==1:
    input_data, input_affine = vreg.generate_plot_data_1()


# %%
pl = vreg.plot_volume(input_data, input_affine, off_screen=True)
pl.show_grid()
pl.show()

