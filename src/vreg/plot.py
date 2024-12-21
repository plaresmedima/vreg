import numpy as np
import pyvista as pv
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

import vreg.utils
import vreg.mod_affine


def plot_overlay_2d(under:np.ndarray | list, over:np.ndarray | list, 
                    alpha=0.25, title=None, off_screen=False):

    if isinstance(under, np.ndarray):
        if under.ndim==2:
            fig, ax = plt.subplots()
            fig.suptitle(title)
            _plot_overlay_2d(ax, under, over, alpha)
        else:
            plot_overlay_2d(
                [under[:,:,z] for z in range(under.shape[-1])], 
                [over[:,:,z] for z in range(over.shape[-1])],
                alpha, title, off_screen,
            )
        plt.show()
    elif isinstance(under, list):
        n = len(under)
        ny = np.ceil(np.sqrt(n)).astype(int)
        nx = np.ceil(n/ny).astype(int)
        fig, ax = plt.subplots(nx, ny, figsize=(ny*3, nx*3+0.5))
        fig.suptitle(title)
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=0.90, 
                            bottom=0)
        k=0
        for i in range(nx):
            for j in range(ny):
                if k < n:
                    _plot_overlay_2d(ax[i,j], under[k], over[k], alpha)
                else:
                    ax[i,j].axis('off')
                    #ax[i,j].set_xticks([])  
                    #ax[i,j].set_yticks([]) 
                k+=1
        plt.show()
    else:
        raise ValueError('Arguments must be either lists or arrays.')



def _plot_overlay_2d(ax, under, over, alpha=0.25):

    # Show the first image
    ax.imshow(under.T, cmap='gray', alpha=1.0)  # Adjust alpha for transparency

    # Create a mask for image2 where values are non-zero
    mask = over > 0

    # Show the second image only where the mask is True, with full opacity for non-zero values
    # Use np.where to set the alpha values based on the mask
    alpha_im = np.zeros_like(over)  # Create an alpha channel for image2
    alpha_im[mask] = alpha  # Set alpha to 1 for non-zero values

    # Display image2 with the mask applied
    # Using 'jet' colormap for image2, but only where mask is True
    ax.imshow(over.T, cmap='jet', alpha=alpha_im.T)  # Non-zero values will be opaque

    # Optional: Add a colorbar
    #plt.colorbar(ax.imshow(image2, cmap='jet', alpha=0.5), ax=ax)

    ax.set_xticks([])  
    ax.set_yticks([]) 


def pv_contour(values, data, affine, surface=False):

    # For display of the surface, interpolate from volume to surface array
    surf_shape = 1 + np.array(data.shape)
    r = vreg.utils.surface_coordinates(data.shape)
    r = vreg.utils.extend_border(r, data.shape)
    surf_data = ndi.map_coordinates(data, r.T, order=3)
    surf_data = np.reshape(surf_data, surf_shape)

    rotation, translation, pixel_spacing = vreg.utils.affine_components(affine)
    grid = pv.ImageData(dimensions=surf_shape, spacing=pixel_spacing)
    surf = grid.contour(values, surf_data.flatten(order="F"), method='marching_cubes')
    surf = surf.rotate_vector(rotation, np.linalg.norm(rotation)*180/np.pi, inplace=False)
    surf = surf.translate(translation, inplace=False)
    if surface:
        surf = surf.reconstruct_surface()
    return surf


def plot_volume(volume, affine, off_screen=False):
    """Plot

    Args:
        volume (_type_): _description_
        affine (_type_): _description_
        off_screen (bool, optional): _description_. Defaults to False.

    Returns:
        pyvista.Plotter: _description_

    Example:

    .. pyvista-plot::
        :caption: A volume
        :include-source: True

        >>> import vreg
        >>> arr, aff = vreg.generate_plot_data_1()
        >>> pl = vreg.plot_volume(arr, aff, off_screen=True)
        >>> out = pl.show()
    """

    clr, op = (255,255,255), 0.5
    #clr, op = (255,0,0), 1.0

    pl = pv.Plotter(off_screen=off_screen)
    pl.add_axes()

    # Plot the surface
    surf = pv_contour([0.5], volume, affine)
    if len(surf.points) == 0:
        print('Cannot plot the reference surface. It has no points inside the volume. ')
    else:
        pl.add_mesh(surf,
            color=clr, 
            opacity=op,
            show_edges=False, 
            smooth_shading=True, 
            specular=0, 
            show_scalar_bar=False,        
        )

    # Plot wireframe around edges of reference volume
    vertices, faces = vreg.utils.parallellepid(volume.shape, affine=affine)
    box = pv.PolyData(vertices, faces)
    pl.add_mesh(box,
        style='wireframe',
        opacity=op,
        color=clr, 
    )
    
    return pl


def plot_affine_resliced(volume, affine, volume_resliced, affine_resliced, off_screen=False):

    clr, op = (255,0,0), 1.0

    pl = plot_volume(volume, affine, off_screen=off_screen)

    # Plot the resliced surface
    surf = pv_contour([0.5], volume_resliced, affine_resliced)
    if len(surf.points) == 0:
        print('Cannot plot the resliced surface. It has no points inside the volume. ')
    else:
        pl.add_mesh(surf,
            color=clr, 
            opacity=op,
            show_edges=False, 
            smooth_shading=True, 
            specular=0, 
            show_scalar_bar=False,        
        )

    # Plot wireframe around edges of resliced volume
    vertices, faces = vreg.utils.parallellepid(volume_resliced.shape, affine=affine_resliced)
    box = pv.PolyData(vertices, faces)
    pl.add_mesh(box,
        style='wireframe',
        opacity=op,
        color=clr, 
    ) 

    return pl


def plot_affine_transformed(input_data, input_affine, output_data, output_affine, transformation, off_screen=False):

    pl = plot_affine_resliced(input_data, input_affine, output_data, output_affine, off_screen=off_screen)

    # Plot the reference surface
    surf = pv_contour([0.5], output_data, output_affine)
    if len(surf.points) == 0:
        print('Cannot plot the surface. It has no points inside the volume. ')
    else:
        pl.add_mesh(surf,
            color=(0,0,255), 
            opacity=0.25,
            show_edges=False, 
            smooth_shading=True, 
            specular=0, 
            show_scalar_bar=False,        
        )
    
    # Create blue reference box showing transformation
    vertices, faces = vreg.utils.parallellepid(input_data.shape, affine=np.dot(transformation, input_affine)) 
    box = pv.PolyData(vertices, faces)
    pl.add_mesh(box,
        style='wireframe',
        color=(0,0,255),
        opacity=0.5,
    )            
        
    pl.show()


def plot_bounding_box(input_data, input_affine, output_shape, output_affine, off_screen=False):

    pl = plot_volume(input_data, input_affine, off_screen=off_screen)

    # Plot wireframe around edges of resliced volume
    vertices, faces = vreg.utils.parallellepid(output_shape, affine=output_affine)
    box = pv.PolyData(vertices, faces)
    pl.add_mesh(box,
        style='wireframe',
        opacity=1.0,
        color=(255,0,0), 
    )
    pl.show()


def plot_freeform_transformed(input_data, input_affine, output_data, output_affine, off_screen=False):

    pl = plot_affine_resliced(input_data, input_affine, output_data, output_affine, off_screen=off_screen)

    # Plot the reference surface
    surf = pv_contour([0.5], output_data, output_affine)
    if len(surf.points) == 0:
        print('Cannot plot the surface. It has no points inside the volume. ')
    else:
        pl.add_mesh(surf,
            color=(0,0,255), 
            opacity=0.25,
            show_edges=False, 
            smooth_shading=True, 
            specular=0, 
            show_scalar_bar=False,        
        )           
        
    pl.show()


def plot_affine_transform_reslice(input_data, input_affine, output_data, output_affine, transformation, off_screen=False):
    """Visualise the result of an affine transformation with reslicing.

    The original volume is shown in grey, and the resulting volume in red. In 
    blue is shown the transformed volume before reslicing to the final shape.

    Args:
        input_data (numpy.ndarray): Volume array before transformation.
        input_affine (numpy.ndarray): Volume affine before transformation.
        output_data (numpy.ndarray): Volume array after transformation.
        output_affine (numpy.ndarray): Volume affine after transformation.
        transformation (numpy.ndarray): affine matrix of the transformation.
        off_screen (bool, optional): Show the result (True) or not (False). 
          Defaults to False.

    Example:

    .. pyvista-plot::
        :caption: Active affine transform with reslicing.
        :include-source: True

        >>> import vreg
        >>> iarr, iaff, oarr, oaff, transl = vreg.generate_translated_data_2()
        >>> transfo = vreg.affine_matrix(translation=transl)
        >>> vreg.plot_affine_transform_reslice(iarr, iaff, oarr, oaff, transfo, off_screen=True)
    """

    # Plot reslice
    pl = plot_affine_resliced(input_data, input_affine, output_data, output_affine, off_screen=off_screen)
    
    # Show in blue transparent the transformation without reslicing
    output_data, output_affine = vreg.mod_affine.affine_transform(input_data, input_affine, transformation, reshape=True)

    # Plot the reference surface
    surf = pv_contour([0.5], output_data, output_affine)
    if len(surf.points) == 0:
        print('Cannot plot the surface. It has no points inside the volume. ')
    else:
        pl.add_mesh(surf,
            color=(0,0,255), 
            opacity=0.25,
            show_edges=False, 
            smooth_shading=True, 
            specular=0, 
            show_scalar_bar=False,        
        )

    # Create blue reference box showing transformation
    vertices, faces = vreg.utils.parallellepid(input_data.shape, affine=np.dot(transformation, input_affine))
    box = pv.PolyData(vertices, faces)
    pl.add_mesh(box,
        style='wireframe',
        color=(0,0,255),
        opacity=0.5,
    )  

    pl.show()