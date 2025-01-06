import numpy as np

import vreg.mod_affine
import vreg.utils


def is_passive(transform):
    return transform in [
        translate_passive_ortho, 
        translate_passive,
        translate_passive_inslice,
        rigid_passive_com_ortho,
    ]


def passive_transform(affine, transform, parameters, static_array=None, static_affine=None):

    if transform==translate_passive_ortho:
        return passive_ortho_translation(affine, parameters)
    elif transform==translate_passive:
        return passive_translation(affine, parameters)
    elif transform==translate_passive_inslice:
        return passive_inslice_translation(affine, parameters)
    elif transform==rigid_passive_com_ortho:
        static_com = vreg.utils.center_of_mass(static_array, static_affine)
        return passive_ortho_com_rigid(affine, static_com, parameters) 
    
    
def active_transform(input_array, input_affine, transform, parameters, output_shape=None, output_affine=None, **kwargs):

    if transform==translate:
        return translate(input_array, input_affine, output_shape, output_affine, parameters, **kwargs)
    elif transform==translate_inslice:
        return translate_inslice(input_array, input_affine, output_shape, output_affine, parameters, **kwargs)
    elif transform==translate_reshape:
        return translate_reshape(input_array, input_affine, parameters, **kwargs)
    elif transform==rotate:
        return rotate(input_array, input_affine, output_shape, output_affine, parameters, **kwargs)
    elif transform==rotate_reshape:
        return rotate_reshape(input_array, input_affine, parameters, **kwargs)
    elif transform==stretch:
        return stretch(input_array, input_affine, output_shape, output_affine, parameters, **kwargs)
    elif transform==stretch_reshape:
        return stretch_reshape(input_array, input_affine, parameters, **kwargs)
    elif transform==rotate_around:
        return rotate_around(input_array, input_affine, output_shape, output_affine, parameters, **kwargs)
    elif transform==rotate_around_com:
        return rotate_around_com(input_array, input_affine, output_shape, output_affine, parameters, **kwargs)
    elif transform==rotate_around_reshape:
        return rotate_around_reshape(input_array, input_affine, parameters, kwargs['center'], **{k:v for k,v in kwargs.items() if k!='center'})
    elif transform==rigid:
        return rigid(input_array, input_affine, output_shape, output_affine, parameters, **kwargs)
    elif transform==rigid_around:
        return rigid_around(input_array, input_affine, output_shape, output_affine, parameters, **kwargs)
    elif transform==rigid_around_com:
        return rigid_around_com(input_array, input_affine, output_shape, output_affine, parameters, **kwargs)
    elif transform==rigid_reshape:
        return rigid_reshape(input_array, input_affine, parameters[:3], parameters[3:], **kwargs)
    elif transform==affine:
        return affine(input_array, input_affine, output_shape, output_affine, parameters, **kwargs)
    elif transform==affine_reshape:
        return affine_reshape(input_array, input_affine, parameters[:3], parameters[3:6], parameters[6:], **kwargs)
     


# Passive

def rigid_passive_ortho(moving_array_shape, moving_affine, static_array, static_affine, parameters):
    moving_affine_rigid = vreg.mod_affine.passive_ortho_rigid(moving_affine, parameters)
    static_array_reslice, _ =  vreg.mod_affine.affine_reslice(static_array, static_affine, moving_affine_rigid, output_shape=moving_array_shape)
    return static_array_reslice

def rigid_passive_com_ortho(moving_array_shape, moving_affine, static_array, static_affine, parameters):
    static_com = vreg.utils.center_of_mass(static_array, static_affine)
    moving_affine_rigid = vreg.mod_affine.passive_ortho_com_rigid(moving_affine, static_com, parameters)
    static_array_reslice, _ =  vreg.mod_affine.affine_reslice(static_array, static_affine, moving_affine_rigid, output_shape=moving_array_shape)
    return static_array_reslice

def translate_passive_inslice(moving_array_shape, moving_affine, static_array, static_affine, translation):
    moving_affine_translated = vreg.mod_affine.passive_inslice_translation(moving_affine, translation)
    static_array_reslice, _ =  vreg.mod_affine.affine_reslice(static_array, static_affine, moving_affine_translated, output_shape=moving_array_shape)
    return static_array_reslice

def translate_passive(moving_array_shape, moving_affine, static_array, static_affine, translation):
    moving_affine_translated = vreg.mod_affine.passive_translation(moving_affine, translation)
    static_array_reslice, _ =  vreg.mod_affine.affine_reslice(static_array, static_affine, moving_affine_translated, output_shape=moving_array_shape)
    return static_array_reslice

def translate_passive_ortho(moving_array_shape, moving_affine, static_array, static_affine, translation):
    moving_affine_translated = vreg.mod_affine.passive_ortho_translation(moving_affine, translation)
    static_array_reslice, _ =  vreg.mod_affine.affine_reslice(static_array, static_affine, moving_affine_translated, output_shape=moving_array_shape)
    return static_array_reslice

# Active

def translate(moving_array, moving_affine, static_shape, static_affine, translation, **kwargs):
    transformation = vreg.utils.affine_matrix(translation=translation)
    return vreg.mod_affine.affine_transform_and_reslice(moving_array, moving_affine, static_shape, static_affine, transformation, **kwargs)

def translate_inslice(input_data, input_affine, output_shape, output_affine, translation, **kwargs):
    transformation = vreg.utils.affine_matrix(translation=vreg.utils.inslice_vector(input_affine, translation))
    return vreg.mod_affine.affine_transform_and_reslice(input_data, input_affine, output_shape, output_affine, transformation, **kwargs)

def translate_reshape(input_data, input_affine, translation, **kwargs):
    transformation = vreg.utils.affine_matrix(translation=translation)
    return vreg.mod_affine.affine_transform(input_data, input_affine, transformation, reshape=True, **kwargs)

def rotate(input_data, input_affine, output_shape, output_affine, rotation, **kwargs):
    transformation = vreg.utils.affine_matrix(rotation=rotation)
    return vreg.mod_affine.affine_transform_and_reslice(input_data, input_affine, output_shape, output_affine, transformation, **kwargs)

def rotate_reshape(input_data, input_affine, rotation, **kwargs):
    transformation = vreg.utils.affine_matrix(rotation=rotation)
    return vreg.mod_affine.affine_transform(input_data, input_affine, transformation, reshape=True, **kwargs)

def stretch(input_data, input_affine, output_shape, output_affine, stretch, **kwargs):
    transformation = vreg.utils.affine_matrix(pixel_spacing=stretch)
    return vreg.mod_affine.affine_transform_and_reslice(input_data, input_affine, output_shape, output_affine, transformation, **kwargs)

def stretch_reshape(input_data, input_affine, stretch, **kwargs):
    transformation = vreg.utils.affine_matrix(pixel_spacing=stretch)
    return vreg.mod_affine.affine_transform(input_data, input_affine, transformation, reshape=True, **kwargs)

def rotate_around(input_data, input_affine, output_shape, output_affine, parameters, **kwargs):
    transformation = vreg.utils.affine_matrix(rotation=parameters[:3], center=parameters[3:])
    return vreg.mod_affine.affine_transform_and_reslice(input_data, input_affine, output_shape, output_affine, transformation, **kwargs)

def rotate_around_com(input_data, input_affine, output_shape, output_affine, parameters, **kwargs):
    input_data = vreg.utils.to_3d(input_data) # need for com - not the right place
    input_com = vreg.utils.center_of_mass(input_data, input_affine) # can be precomputed
    transformation = vreg.utils.affine_matrix(rotation=parameters, center=input_com)
    return vreg.mod_affine.affine_transform_and_reslice(input_data, input_affine, output_shape, output_affine, transformation, **kwargs)

def rotate_around_reshape(input_data, input_affine, rotation, center, **kwargs):
    transformation = vreg.utils.affine_matrix(rotation=rotation, center=center)
    return vreg.mod_affine.affine_transform(input_data, input_affine, transformation, reshape=True, **kwargs)

def rigid(input_data, input_affine, output_shape, output_affine, parameters, **kwargs):
    transformation = vreg.utils.affine_matrix(rotation=parameters[:3], translation=parameters[3:])
    return vreg.mod_affine.affine_transform_and_reslice(input_data, input_affine, output_shape, output_affine, transformation, **kwargs)

def rigid_around(input_data, input_affine, output_shape, output_affine, parameters, **kwargs):
    transformation = vreg.utils.affine_matrix(rotation=parameters[:3], center=parameters[3:6], translation=parameters[6:])
    return vreg.mod_affine.affine_transform_and_reslice(input_data, input_affine, output_shape, output_affine, transformation, **kwargs)

def rigid_around_com(input_data, input_affine, output_shape, output_affine, parameters, **kwargs):
    input_data = vreg.utils.to_3d(input_data) # needed for com - not the right place
    input_com = vreg.utils.center_of_mass(input_data, input_affine) 
    transformation = vreg.utils.affine_matrix(rotation=parameters[:3], center=parameters[3:]+input_com, translation=parameters[3:])
    return vreg.mod_affine.affine_transform_and_reslice(input_data, input_affine, output_shape, output_affine, transformation, **kwargs)

def rigid_reshape(input_data, input_affine, rotation, translation, **kwargs):
    transformation = vreg.utils.affine_matrix(rotation=rotation, translation=translation)
    return vreg.mod_affine.affine_transform(input_data, input_affine, transformation, reshape=True, **kwargs)

def affine(input_data, input_affine, output_shape, output_affine, parameters, **kwargs):
    transformation = vreg.utils.affine_matrix(rotation=parameters[:3], translation=parameters[3:6], pixel_spacing=parameters[6:])
    return vreg.mod_affine.affine_transform_and_reslice(input_data, input_affine, output_shape, output_affine, transformation, **kwargs)

def affine_reshape(input_data, input_affine, rotation, translation, stretch, **kwargs):
    transformation = vreg.utils.affine_matrix(rotation=rotation, translation=translation, pixel_spacing=stretch)
    return vreg.mod_affine.affine_transform(input_data, input_affine, transformation, reshape=True, **kwargs)


def transform_slice_by_slice(input_data, input_affine, output_shape, output_affine, parameters, transformation=translate, slice_thickness=None):
    
    # Note this does not work for center of mass rotation because weight array has different center of mass.
    nz = input_data.shape[2]
    if slice_thickness is not None:
        if not isinstance(slice_thickness, list):
            slice_thickness = [slice_thickness]*nz

    weight = np.zeros(output_shape)
    coregistered = np.zeros(output_shape)
    input_ones_z = np.ones(input_data.shape[:2])
    for z in range(nz):
        input_data_z, input_affine_z = vreg.utils.extract_slice(input_data, input_affine, z, slice_thickness)
        weight_z = transformation(input_ones_z, input_affine_z, output_shape, output_affine, parameters[z])
        coregistered_z = transformation(input_data_z, input_affine_z, output_shape, output_affine, parameters[z])
        weight += weight_z
        coregistered += weight_z*coregistered_z

    # Average each pixel value over all slices that have sampled it
    nozero = np.where(weight > 0)
    coregistered[nozero] = coregistered[nozero]/weight[nozero]
    return coregistered



def affine_reslice_slice_by_slice(input_data, input_affine, output_affine, output_shape=None, slice_thickness=None, mask=False, label=False, **kwargs):
    # generalizes affine_reslice - also works with multislice volumes where slice thickness is less than slice gap

    # If 3D volume - do normal affine_reslice
    if slice_thickness is None:
        output_data, output_affine = vreg.mod_affine.affine_reslice(input_data, input_affine, output_affine, output_shape=output_shape, **kwargs)
    # If slice thickness equals slice spacing:
    # then its a 3D volume - do normal affine_reslice 
    elif slice_thickness == np.linalg.norm(input_affine[:3,2]):
        output_data, output_affine = vreg.mod_affine.affine_reslice(input_data, input_affine, output_affine, output_shape=output_shape, **kwargs)
    # If multislice - perform affine slice by slice
    else:
        output_data = None
        for z in range(input_data.shape[2]):
            input_data_z, input_affine_z = vreg.utils.extract_slice(input_data, input_affine, z, slice_thickness=slice_thickness)
            output_data_z, output_affine = vreg.mod_affine.affine_reslice(input_data_z, input_affine_z, output_affine, output_shape=output_shape, **kwargs)
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
    translation = vreg.utils.volume_vector(translation, input_affine)
    transform = vreg.utils.affine_matrix(translation=translation)
    output_affine = transform.dot(input_affine)
    return output_affine

def passive_inslice_translation(input_affine, parameters):
    translation = vreg.utils.inslice_vector(input_affine, parameters)
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
    translation = vreg.utils.volume_vector(parameters[3:], input_affine)
    transform = vreg.utils.affine_matrix(
        rotation=parameters[:3], 
        center=translation+input_com, 
        translation=translation,
    )
    output_affine = transform.dot(input_affine)
    return output_affine

def passive_ortho_rigid(input_affine, parameters):
    translation = vreg.utils.volume_vector(parameters[3:], input_affine)
    transform = vreg.utils.affine_matrix(
        rotation=parameters[:3], 
        translation=translation,
    )
    output_affine = transform.dot(input_affine)
    return output_affine

