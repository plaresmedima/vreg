import numpy as np

import vreg.optimize
import vreg.utils
import vreg.mod_affine
import vreg.mod_freeform
import vreg.obsolete_transforms
import vreg.metrics



def goodness_of_alignment_passive(
        params, transformation, metric, moving, moving_affine, static, 
        static_affine, coord, moving_mask, static_mask_ind):

    # Transform the moving image
    nan = 2**16-2 #np.nan does not work
    static_transformed = transformation(moving.shape, moving_affine, static, static_affine, params)
    
    # If a moving mask is provided, this needs to be transformed in the same way
    if moving_mask is None:
        moving_mask_ind = None
    else:
        moving_mask_ind = np.where(moving_mask >= 0.5)

    # Calculate matric in indices exposed by the mask(s)
    if static_mask_ind is None and moving_mask_ind is None:
        return metric(static_transformed, moving, nan=nan)
    if static_mask_ind is None and moving_mask_ind is not None:
        return metric(static_transformed[moving_mask_ind], moving[moving_mask_ind], nan=nan)
    if static_mask_ind is not None and moving_mask_ind is None:
        return metric(static_transformed[static_mask_ind], moving[static_mask_ind], nan=nan)
    if static_mask_ind is not None and moving_mask_ind is not None:
        ind = static_mask_ind or moving_mask_ind
        return metric(static_transformed[ind], moving[ind], nan=nan)


def goodness_of_alignment(
        params, transformation, metric, moving, moving_affine, static, 
        static_affine, coord, moving_mask, static_mask_ind):

    # Transform the moving image
    nan = 2**16-2 #np.nan does not work
    moving_transformed = transformation(
        moving, moving_affine, static.shape, static_affine, params, 
        output_coordinates=coord, cval=nan)
    
    # If a moving mask is provided, this needs to be transformed in the same way
    if moving_mask is None:
        moving_mask_ind = None
    else:
        mask_transformed = transformation(moving_mask, moving_affine, static.shape, static_affine, params, output_coordinates=coord, cval=nan)
        moving_mask_ind = np.where(mask_transformed >= 0.5)

    # Calculate matric in indices exposed by the mask(s)
    if static_mask_ind is None and moving_mask_ind is None:
        return metric(static, moving_transformed, nan=nan)
    if static_mask_ind is None and moving_mask_ind is not None:
        return metric(static[moving_mask_ind], moving_transformed[moving_mask_ind], nan=nan)
    if static_mask_ind is not None and moving_mask_ind is None:
        return metric(static[static_mask_ind], moving_transformed[static_mask_ind], nan=nan)
    if static_mask_ind is not None and moving_mask_ind is not None:
        ind = static_mask_ind or moving_mask_ind
        return metric(static[ind], moving_transformed[ind], nan=nan)
    




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
        transformation = vreg.obsolete_transforms.translate,
        metric = vreg.metrics.mutual_information,
        optimize = 'LS',
        options = {},
        resolutions = [1], 
        static_mask = None,
        static_mask_affine = None, 
        moving_mask = None,
        moving_mask_affine = None):
    
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
        #print('DOWNSAMPLE BY FACTOR: ', res)

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
        if vreg.obsolete_transforms.is_passive(transformation):
            goa = goodness_of_alignment_passive
        else:
            goa = goodness_of_alignment
        parameters = vreg.optimize.minimize(goa, parameters, args=args, method=optimize, **options)

    return parameters


def align_slice_by_slice(
        moving = None, 
        static = None, 
        parameters = None, 
        moving_affine = None, 
        static_affine = None, 
        transformation = vreg.obsolete_transforms.translate,
        metric = vreg.metrics.sum_of_squares,
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
        moving_z, moving_affine_z = vreg.utils.extract_slice(moving, moving_affine, z, slice_thickness)
        if moving_mask is None:
            moving_mask_z, moving_mask_affine_z = None, None
        else:
            moving_mask_z, moving_mask_affine_z = vreg.utils.extract_slice(moving_mask, moving_mask_affine, z, slice_thickness)

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