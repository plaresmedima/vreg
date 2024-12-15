import vreg.utils
import vreg.mod_affine


def mask_volume(array, affine, array_mask, affine_mask, margin:float=0):

    # Overlay the mask on the array.
    array_mask, _ = vreg.mod_affine.affine_reslice(array_mask, affine_mask, affine, array.shape)

    # Mask out array pixels outside of region.
    array *= array_mask

    # Extract bounding box around non-zero pixels in masked array.
    array, affine = vreg.utils.bounding_box(array, affine, margin)
    
    return array, affine