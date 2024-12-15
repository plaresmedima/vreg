import numpy as np

import vreg.optimize
import vreg.fake
import vreg.utils
import vreg.plot
import vreg.mod_freeform
import vreg.mod_affine


# default metrics
# ---------------

def interaction(static, transformed, nan=None):
    # static is here a mask image.
    if nan is not None:
        i = np.where(transformed != nan)
        msk, img = static[i], transformed[i]
    else:
        msk, img = static, transformed
    return np.std(img[np.where(msk>0.5)])
    return 1/np.mean(np.abs(msk*img))
    return np.exp(-np.mean(np.abs(msk*img)))
    i = img[np.where(msk>0.5)]
    return np.exp(-np.mean(i**2))


def sum_of_squares(static, transformed, nan=None):
    if nan is not None:
        i = np.where(transformed != nan)
        st, tr = static[i], transformed[i]
    else:
        st, tr = static, transformed
    return np.sum(np.square(st-tr))
    

def mutual_information(static, transformed, nan=None):

    # Mask if needed
    if nan is not None:
        i = np.where(transformed != nan)
        st, tr = static[i], transformed[i]
    else:
        st, tr = static, transformed
    # Calculate 2d histogram
    hist_2d, _, _ = np.histogram2d(st.ravel(), tr.ravel(), bins=20)
    # Convert bins counts to probability values
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return -np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))





def goodness_of_alignment(params, transformation, metric, moving, moving_affine, static, static_affine, coord, moving_mask, static_mask_ind):

    # Transform the moving image
    nan = 2**16-2 #np.nan does not work
    transformed = transformation(moving, moving_affine, static.shape, static_affine, params, output_coordinates=coord, cval=nan)
    
    # If a moving mask is provided, this needs to be transformed in the same way
    if moving_mask is None:
        moving_mask_ind = None
    else:
        mask_transformed = transformation(moving_mask, moving_affine, static.shape, static_affine, params, output_coordinates=coord, cval=nan)
        moving_mask_ind = np.where(mask_transformed >= 0.5)

    # Calculate matric in indices exposed by the mask(s)
    if static_mask_ind is None and moving_mask_ind is None:
        return metric(static, transformed, nan=nan)
    if static_mask_ind is None and moving_mask_ind is not None:
        return metric(static[moving_mask_ind], transformed[moving_mask_ind], nan=nan)
    if static_mask_ind is not None and moving_mask_ind is None:
        return metric(static[static_mask_ind], transformed[static_mask_ind], nan=nan)
    if static_mask_ind is not None and moving_mask_ind is not None:
        ind = static_mask_ind or moving_mask_ind
        return metric(static[ind], transformed[ind], nan=nan)
    




def align_freeform(nodes=[2,4,8], static=None, parameters=None, optimization=None, **kwargs):

    transformation = vreg.mod_freeform.freeform
    if parameters is None:
        # Initialise the inverse deformation field
        dim = vreg.mod_freeform.deformation_field_shape(static.shape, nodes[0])
        parameters = np.zeros(dim)

    for i in range(nodes):

        if optimization['method'] == 'GD':
            # Define the step size
            step = np.full(parameters.shape, 0.1)
            optimization['options']['gradient step'] = step

        # Coregister for nr nodes
        try:
            parameters = align(
                static= static,
                parameters = parameters, 
                optimization = optimization,
                transformation = transformation,
                **kwargs,
            )
        except:
            print('Failed to align volumes. Returning zeros as best guess..')
            dim = vreg.mod_freeform.deformation_field_shape(static.shape, nodes[0])
            return np.zeros(dim)
        
        if i+1 < len(nodes):
            dim = vreg.mod_freeform.deformation_field_shape(static.shape, nodes[i+1])
            parameters = vreg.mod_freeform.upsample_deformation_field(parameters, dim[:3])

    return parameters
    
    
def align(
        moving = None, 
        static = None, 
        parameters = None, 
        moving_affine = None, 
        static_affine = None, 
        transformation = vreg.mod_affine.translate,
        metric = sum_of_squares,
        optimize = 'LS',
        options = {},
        resolutions = [1], 
        static_mask = None,
        static_mask_affine = None, 
        moving_mask = None,
        moving_mask_affine = None):
    
    """Find the affine transformation between two volumes.

    Args:
        moving (_type_, optional): _description_. Defaults to None.
        static (_type_, optional): _description_. Defaults to None.
        parameters (_type_, optional): _description_. Defaults to None.
        moving_affine (_type_, optional): _description_. Defaults to None.
        static_affine (_type_, optional): _description_. Defaults to None.
        transformation (_type_, optional): _description_. Defaults to vreg.mod_affine.translate.
        metric (_type_, optional): _description_. Defaults to sum_of_squares.
        optimize (str, optional): Optimization method to use. Options are 
          'GD', 'brute', 'ibrute', 'LS' or 'min'. Defaults to 'LS'.
        options (dict, optional): keyword arguments for the optimization function.
        resolutions (list, optional): _description_. Defaults to [1].
        static_mask (_type_, optional): _description_. Defaults to None.
        static_mask_affine (_type_, optional): _description_. Defaults to None.
        moving_mask (_type_, optional): _description_. Defaults to None.
        moving_mask_affine (_type_, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    
    # Set defaults
    if moving is None:
        msg = 'The moving volume is a required argument for alignment'
        raise ValueError(msg)
    if static is None:
        msg = 'The static volume is a required argument for alignment'
        raise ValueError(msg)
    if moving.ndim == 2: # If 2d array, add a 3d dimension of size 1
        moving = np.expand_dims(moving, axis=-1)
    if static.ndim == 2: # If 2d array, add a 3d dimension of size 1
        static = np.expand_dims(static, axis=-1)
    if moving_affine is None:
        moving_affine = np.eye(1 + moving.ndim)
    if static_affine is None:
        static_affine = np.eye(1 + static.ndim)
    if moving_mask is not None:
        if moving_mask_affine is None:
            moving_mask_affine = moving_affine
    if static_mask is not None:
        if static_mask_affine is None:
            static_mask_affine = static_affine

    # Perform multi-resolution loop
    for res in resolutions:
        print('DOWNSAMPLE BY FACTOR: ', res)

        if res == 1:
            moving_resampled, moving_resampled_affine = moving, moving_affine
            static_resampled, static_resampled_affine = static, static_affine
        else:
            # Downsample moving data
            r, t, p = vreg.utils.affine_components(moving_affine)
            moving_resampled_affine = vreg.utils.affine_matrix(rotation=r, translation=t, pixel_spacing=p*res)
            moving_resampled, moving_resampled_affine = vreg.mod_affine.affine_reslice(moving, moving_affine, moving_resampled_affine)
            #moving_resampled_data, moving_resampled_affine = moving_data, moving_affine

            # Downsample static data
            r, t, p = vreg.utils.affine_components(static_affine)
            static_resampled_affine = vreg.utils.affine_matrix(rotation=r, translation=t, pixel_spacing=p*res)
            static_resampled, static_resampled_affine = vreg.mod_affine.affine_reslice(static, static_affine, static_resampled_affine)

        # resample the masks on the geometry of the target volumes
        if moving_mask is None:
            moving_mask_resampled = None
        else:
            moving_mask_resampled, _ = vreg.mod_affine.affine_reslice(moving_mask, moving_mask_affine, moving_resampled_affine, moving_resampled.shape)
        if static_mask is None:
            static_mask_resampled_ind = None
        else:
            static_mask_resampled, _ = vreg.mod_affine.affine_reslice(static_mask, static_mask_affine, static_resampled_affine, static_resampled.shape)
            static_mask_resampled_ind = np.where(static_mask_resampled >= 0.5)

        coord = vreg.utils.volume_coordinates(static_resampled.shape) 
        # Here we need a generic precomputation step:
        # prec = precompute(moving_resampled, moving_resampled_affine, static_resampled, static_resampled_affine)
        # args = (transformation, metric, moving_resampled, moving_resampled_affine, static_resampled, static_resampled_affine, coord, prec)
        args = (transformation, metric, moving_resampled, moving_resampled_affine, static_resampled, static_resampled_affine, coord, moving_mask_resampled, static_mask_resampled_ind)
        parameters = vreg.optimize.minimize(goodness_of_alignment, parameters, args=args, method=optimize, **options)

    return parameters


def align_slice_by_slice(
        moving = None, 
        static = None, 
        parameters = None, 
        moving_affine = None, 
        static_affine = None, 
        transformation = vreg.mod_affine.translate,
        metric = sum_of_squares,
        optimization = {'method':'GD', 'options':{}},
        resolutions = [1],
        slice_thickness = None,
        progress = None,
        static_mask = None,
        static_mask_affine = None, 
        moving_mask = None,
        moving_mask_affine = None):
    
    # If a single slice thickness is provided, turn it into a list.
    nz = moving.shape[2]
    if slice_thickness is not None:
        if not isinstance(slice_thickness, list):
            slice_thickness = [slice_thickness]*nz
    if not isinstance(parameters, list):
        parameters = [parameters]*nz

    estimate = []
    for z in range(nz):

        print('SLICE: ', z)

        if progress is not None:
            progress(z, nz)
        
        # Get the slice and its affine
        moving_z, moving_affine_z = vreg.mod_affine.extract_slice(moving, moving_affine, z, slice_thickness)
        if moving_mask is None:
            moving_mask_z, moving_mask_affine_z = None, None
        else:
            moving_mask_z, moving_mask_affine_z = vreg.mod_affine.extract_slice(moving_mask, moving_mask_affine, z, slice_thickness)

        # Align volumes
        try:
            estimate_z = align(
                moving = moving_z, 
                moving_affine = moving_affine_z, 
                static = static, 
                static_affine = static_affine, 
                parameters = parameters[z], 
                resolutions = resolutions, 
                transformation = transformation,
                metric = metric,
                optimization = optimization,
                static_mask = static_mask,
                static_mask_affine = static_mask_affine, 
                moving_mask = moving_mask_z,
                moving_mask_affine = moving_mask_affine_z,
            )
        except:
            estimate_z = parameters[z]

        estimate.append(estimate_z)

    return estimate