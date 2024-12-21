import numpy as np

from vreg import mod_affine, tools, mod_align, transforms, utils

class Volume:
    """
    A spatially aware numpy array
    
    Args:
        values (np.ndarray): 2D or 3D numpy array with values
        affine (np.ndarray, optional): 4x4 array with the affine matrix 
            of the value array. If not provided, the identity is assumed. 
            Defaults to None.    
    """

    def __init__(self, values:np.ndarray, affine=None | np.ndarray):

        if affine is None:
            affine = np.eye(4)

        if not isinstance(values, np.ndarray):
            raise ValueError('values must be a numpy array.')
        
        if not isinstance(affine, np.ndarray):
            raise ValueError('affine must be a numpy array.')
        
        if values.ndim not in [2,3]:
            raise ValueError('values must have 2 or 3 dimensions.')
        
        if affine.shape != (4,4):
            raise ValueError('affine must be a 4x4 array.')
        
        self._values = values
        self._affine = affine
    
    @property
    def values(self):
        return self._values
    
    @property
    def affine(self):
        return self._affine
    
    @property
    def shape(self):
        return self._values.shape
    
    def copy(self, **kwargs):
        """Return a copy

        Returns:
            Volume: copy
        """
        return Volume(
            self.values.copy(**kwargs), 
            self.affine.copy(**kwargs),
        )
    
    def slice_like(self, v):
        """Slice the volume to the geometry of another volume

        Args:
            v (Volume): reference volume with desired orientation and shape.

        Returns:
            Volume: resliced volume
        """
        values, affine = mod_affine.affine_reslice(
            self.values, self.affine, 
            v.affine, output_shape=v.shape)
        return Volume(values, affine)


    def add(self, v, *args, **kwargs):
        """Add another volume

        Args:
            v (Volume): volume to add. If this is in a different geometry, it 
              will be resliced first
            args, kwargs: arguments and keyword arguments of `numpy.copy`.

        Returns:
            Volume: sum of the two volumes
        """
        v = v.slice_like(self)
        values = np.add(self.values, v.values, *args, **kwargs)
        return Volume(values, self.affine)
    
    def bounding_box(self, mask=None, margin=0.0):
        """Return the bounding box

        Args:
            mask (Volume, optional): If mask is None, the bounding box is 
              drawn around the non-zero values of the Volume. If mask is 
               provided, it is drawn around the non-zero values of mask 
               instead. Defaults to None.
            margin (float, optional): How big a margin (in mm) around the 
              object. Defaults to 0.

        Returns:
            Volume: the bounding box
        """
        if mask is None:
            values, affine = tools.mask_volume(
                self.values, self.affine, self.values, self.affine, margin)
        else:
            values, affine = tools.mask_volume(
                self.values, self.affine, mask.values, mask.affine, margin)
        return Volume(values, affine)
    

    # def translate(self, translation, ref=None, inplace=False):
    #     if ref is None:
    #         ref = self.affine
    #     translation = utils.ortho_translation(ref, translation)
    #     transform = utils.affine_matrix(translation=translation)
    #     affine_translated = transform.dot(self.affine)
    #     if inplace:
    #         self.affine = affine_translated
    #         return self
    #     else:
    #         return Volume(self.values.copy(), affine_translated)

    # def translate_to(
    #         self, static, metric=None, params=False, inplace=False, 
    #         optimize='LS', options={}):
    #     if metric is None:
    #         metric=mod_align.mutual_information
    #     transformation=transforms.translate_passive_ortho
    #     translation=mod_align.align(
    #         moving=self.values, moving_affine=self.affine,
    #         static=static.values, static_affine=static.affine,
    #         transformation=transformation, metric=metric,
    #         optimize=optimize, options=options,
    #     ) 
    #     aligned = self.translate(translation, inplace=inplace)
    #     if params:
    #         return aligned, translation
    #     else:
    #         return aligned
        
    def coreg_to(
            self, static, metric=None, transform=None, params=None, 
            return_params=False, optimize='LS', options={}, 
            resolutions=[1], static_mask=None, moving_mask=None):
        
        if metric is None:
            metric=mod_align.mutual_information
        if transform is None:
            transform=transforms.translate_passive_ortho
        if static_mask is not None:
            static_mask=static_mask.values
            static_mask_affine=static_mask.affine
        else:
            static_mask_affine=None
        if moving_mask is not None:
            moving_mask=moving_mask.values
            moving_mask_affine=moving_mask.affine
        else:
            moving_mask_affine=None

        params=mod_align.align(
            moving=self.values, moving_affine=self.affine,
            static=static.values, static_affine=static.affine,
            transformation=transform, metric=metric, optimize=optimize, 
            options=options, parameters=params, resolutions=resolutions,
            static_mask=static_mask, static_mask_affine=static_mask_affine, 
            moving_mask=moving_mask, moving_mask_affine=moving_mask_affine
        ) 
        aligned = self.transform(transform, params, ref=static)

        if return_params:
            return aligned, params
        else:
            return aligned
        
    
    def transform(self, transform, params, inplace=False, ref=None):
        if transforms.is_passive(transform):
            return self._transform_affine(transform, params, inplace, ref)
        else:
            return self._transform_values(transform, params, inplace, ref)
        
    
    def _transform_affine(self, transform, params, inplace=False, ref=None):
        if ref is not None:
            static_array = ref.values
            static_affine = ref.affine
        else:
            static_array = None
            static_affine = None
        affine = transforms.passive_transform(
            self.affine, transform, params, 
            static_array=static_array, static_affine=static_affine)
        if inplace:
            self._affine = affine
            return self
        else:
            return Volume(self.values.copy(), affine)


    def _transform_values(self, transform, params, inplace=False, ref=None):
        if ref is not None:
            output_shape = ref.values.shape
            output_affine = ref.affine
        else:
            output_shape = None
            output_affine = None
        values = transforms.active_transform(
            self.values, self.affine, transform, params, 
            output_shape=output_shape, output_affine=output_affine)
        if inplace:
            self._values = values
            return self
        else:
            return Volume(values, self.affine.copy())



def zeros(shape, affine=None | np.ndarray, **kwargs):
    """Return a new volume of given shape and affine, filled with zeros.

    Args:
        shape (int or tuple of ints): Shape of the new array, e.g., (2, 3) or 2.
        affine (array, optional): 4x4 array with affine values. If not provided, 
         the identity is assumed. Defaults to None.

    Returns:
        Volume: vreg.Volume with zero values.
    """
    values = np.zeros(shape, **kwargs)
    return Volume(values, affine)

