import numpy as np
import scipy.ndimage as ndi

import vreg.utils



def apply_affine(affine, coord):
    """Apply affine transformation to an array of coordinates"""

    nd = affine.shape[0]-1
    matrix = affine[:nd,:nd]
    offset = affine[:nd, nd]
    return np.dot(coord, matrix.T) + offset
    #return np.dot(matrix, co.T).T + offset


def apply_inverse_affine(
        input_data, inverse_affine, 
        output_shape, output_coordinates=None, 
        order=1, **kwargs):

    # Create an array of all coordinates in the output volume
    if output_coordinates is None:
        output_coordinates = vreg.utils.volume_coordinates(output_shape)

    # Apply affine transformation to all coordinates in the output volume
    # nd = inverse_affine.shape[0]-1
    # matrix = inverse_affine[:nd,:nd]
    # offset = inverse_affine[:nd, nd]
    # input_coordinates = np.dot(output_coordinates, matrix.T) + offset
    # #co = np.dot(matrix, co.T).T + offset
    input_coordinates = apply_affine(inverse_affine, output_coordinates)

    # Extend with constant value for half a voxel outside of the boundary
    input_coordinates = vreg.utils.extend_border(
        input_coordinates, input_data.shape)

    # Interpolate the volume in the transformed coordinates
    #output_data = ndi.map_coordinates(input_data, input_coordinates.T, **kwargs)
    output_data = ndi.map_coordinates(
        input_data, input_coordinates.T, order=order, **kwargs)
    output_data = np.reshape(output_data, output_shape)

    return output_data


def affine_reslice(
        input_data, input_affine:np.ndarray, 
        output_affine:np.ndarray, output_shape=None, 
        **kwargs):

    # If the output shape is 2d, add a 3d dimension of size 1 for the calculation
    output_2d = False
    if output_shape is not None:
        output_2d = np.size(output_shape)==2
        if output_2d:
            output_shape = output_shape + (1,)

    # If 2d array, add a 3d dimension of size 1
    if input_data.ndim == 2: 
        input_data = np.expand_dims(input_data, axis=-1)

    # If no output shape is provided, retain the physical volume of the input datas
    if output_shape is None:

        # Get field of view from input data
        _, _, input_pixel_spacing = vreg.utils.affine_components(input_affine)
        field_of_view = np.multiply(np.array(input_data.shape), input_pixel_spacing)

        # Find output shape for the same field of view
        output_rotation, output_translation, output_pixel_spacing = vreg.utils.affine_components(output_affine)
        output_shape = np.around(np.divide(field_of_view, output_pixel_spacing)).astype(np.int16)
        output_shape[np.where(output_shape==0)] = 1

        # Adjust output pixel spacing to fit the field of view
        output_pixel_spacing = np.divide(field_of_view, output_shape)
        output_affine = vreg.utils.affine_matrix(
            rotation=output_rotation, translation=output_translation, 
            pixel_spacing=output_pixel_spacing)

    # Reslice input data to output geometry
    transform = np.linalg.inv(input_affine).dot(output_affine) # Ai B
    output_data = apply_inverse_affine(input_data, transform, output_shape, **kwargs)

    # If a 2d output shape was requested, return a 2d array
    if output_2d:
        output_data = output_data[:,:,0]

    return output_data, output_affine


# TODO This needs to become a private helper function
def affine_transform(input_data, input_affine, transformation, reshape=False, **kwargs):

    # If 2d array, add a 3d dimension of size 1
    if input_data.ndim == 2: 
        input_data = np.expand_dims(input_data, axis=-1)

    if reshape:
        output_shape, output_affine = vreg.utils.affine_output_geometry(input_data.shape, input_affine, transformation)
    else:
        output_shape, output_affine = input_data.shape, input_affine.copy()

    # Perform the inverse transformation
    affine_transformed = transformation.dot(input_affine)
    inverse = np.linalg.inv(affine_transformed).dot(output_affine) # Ainv Tinv B 
    output_data = apply_inverse_affine(input_data, inverse, output_shape, **kwargs)

    return output_data, output_affine


# This needs a reshape option to expand to the envelope in the new reference frame
def affine_transform_and_reslice(input_data, input_affine, output_shape, output_affine, transformation, **kwargs):

    input_data = vreg.utils.to_3d(input_data)
    affine_transformed = transformation.dot(input_affine)
    inverse = np.linalg.inv(affine_transformed).dot(output_affine) # Ai Ti B
    output_data = apply_inverse_affine(input_data, inverse, output_shape, **kwargs)

    return output_data


def affine_reslice_slice_by_slice(input_data, input_affine, output_affine, output_shape=None, slice_thickness=None, mask=False, label=False, **kwargs):
    # generalizes affine_reslice - also works with multislice volumes where slice thickness is less than slice gap

    # If 3D volume - do normal affine_reslice
    if slice_thickness is None:
        output_data, output_affine = affine_reslice(input_data, input_affine, output_affine, output_shape=output_shape, **kwargs)
    # If slice thickness equals slice spacing:
    # then its a 3D volume - do normal affine_reslice 
    elif slice_thickness == np.linalg.norm(input_affine[:3,2]):
        output_data, output_affine = affine_reslice(input_data, input_affine, output_affine, output_shape=output_shape, **kwargs)
    # If multislice - perform affine slice by slice
    else:
        output_data = None
        for z in range(input_data.shape[2]):
            input_data_z, input_affine_z = vreg.utils.extract_slice(input_data, input_affine, z, slice_thickness=slice_thickness)
            output_data_z, output_affine = affine_reslice(input_data_z, input_affine_z, output_affine, output_shape=output_shape, **kwargs)
            if output_data is None:
                output_data = output_data_z
            else:
                output_data += output_data_z
    # If source is a mask array, convert to binary:
    if mask:
        output_data[output_data > 0.5] = 1
        output_data[output_data <= 0.5] = 0
    # If source is a label array, convert to integers:
    elif label:
        output_data = np.around(output_data)

    return output_data, output_affine


def passive_inslice_translation_slice_by_slice(input_affine, parameters, slice_thickness=None):
    output_affine = []
    for z, pz in enumerate(parameters):
        input_affine_z = vreg.utils.affine_slice(input_affine, z, slice_thickness=slice_thickness)
        transformed_input_affine = passive_inslice_translation(input_affine_z, pz)
        output_affine.append(transformed_input_affine)
    return output_affine

def passive_translation_slice_by_slice(input_affine, parameters, slice_thickness=None):
    output_affine = []
    for z, pz in enumerate(parameters):
        input_affine_z = vreg.utils.affine_slice(input_affine, z, slice_thickness=slice_thickness)
        transformed_input_affine = passive_translation(input_affine_z, pz)
        output_affine.append(transformed_input_affine)
    return output_affine

def passive_rigid_transform_slice_by_slice(input_affine, parameters, slice_thickness=None):
    output_affine = []
    for z, pz in enumerate(parameters):
        input_affine_z = vreg.utils.affine_slice(input_affine, z, slice_thickness=slice_thickness)
        transformed_input_affine = passive_rigid_transform(input_affine_z, pz)
        output_affine.append(transformed_input_affine)
    return output_affine

def passive_ortho_translation(input_affine, translation):
    translation = vreg.utils.ortho_translation(input_affine, translation)
    transform = vreg.utils.affine_matrix(translation=translation)
    output_affine = transform.dot(input_affine)
    return output_affine

def passive_inslice_translation(input_affine, parameters):
    translation = vreg.utils.inslice_translation(input_affine, parameters)
    transform = vreg.utils.affine_matrix(translation=translation)
    output_affine = transform.dot(input_affine)
    return output_affine

def passive_translation(input_affine, parameters):
    transform = vreg.utils.affine_matrix(translation=parameters)
    output_affine = transform.dot(input_affine)
    return output_affine

def passive_rigid_transform(input_affine, parameters):
    rigid_transform = vreg.utils.affine_matrix(rotation=parameters[:3], translation=parameters[3:])
    output_affine = rigid_transform.dot(input_affine)
    return output_affine

def passive_ortho_com_rigid(input_affine, input_com, parameters):
    translation = vreg.utils.ortho_translation(input_affine, parameters[3:])
    transform = vreg.utils.affine_matrix(
        rotation=parameters[:3], 
        center=translation+input_com, 
        translation=translation,
    )
    output_affine = transform.dot(input_affine)
    return output_affine