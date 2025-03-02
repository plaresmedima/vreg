import numpy as np

try:
    import nibabel as nib
except ImportError:
    nib_installed = False
else:
    nib_installed = True

from vreg.vol import Volume3D


IMPORT_ERROR = (
    "Saving in NIfTI format requires the nibabel python package. You can "
    "install it with 'pip install nibabel', or else install vreg with the "
    "rw option as 'pip install vreg[rw]'.")


def write_npz(vol:Volume3D, filepath:str):
    """Write a volume to a single file in numpy's uncompressed .npz file format

    Args:
        vol (Volume3D): the volume to write.
        filepath (str): filepath to the .npz file.
    """
    np.savez(filepath, values=vol.values, affine=vol.affine)


def read_npz(filepath:str):
    """Load a volume created by write_npz()

    Args:
        filepath (str): filepath to the .npz file.

    Returns:
        Volume3D: the volume read from file.
    """
    npz = np.load(filepath)
    if 'values' not in npz.files:
        raise ValueError("The .npz file has not been created by write_npz.")
    if 'affine' not in npz.files:
        raise ValueError("The .npz file has not been created by write_npz.")
    return Volume3D(npz['values'], npz['affine'])


def _affine_to_from_RAH(affine):
    # convert to/from nifti coordinate system
    rot_180 = np.identity(4, dtype=np.float32)
    rot_180[:2,:2] = [[-1,0],[0,-1]]
    return np.matmul(rot_180, affine)


def write_nifti(vol:Volume3D, filepath:str):
    """Write volume to disk in NIfTI format.

    Args:
        vol (Volume3D): the volume to write.
        filepath (str): filepath to the NIfTI file.

    Raises:
        ImportError: Error raised if nibabel is not installed.

    Note:
        This requires a separate installation of the optional nibabel package, 
        either via 'pip install nibabel' or by installing vreg with the rw 
        option 'pip install vreg[rw]'.
    """
    if not nib_installed:
        raise ImportError(IMPORT_ERROR)
    affine = _affine_to_from_RAH(vol.affine)
    nifti = nib.Nifti1Image(vol.values, affine)
    nib.save(nifti, filepath)


def read_nifti(filepath:str):
    """Read volume from a NIfTI file on disk.

    Args:
        filepath (str): filepath to the NIfTI file.

    Raises:
        ImportError: Error raised if nibabel is not installed.

    Returns:
        Volume3D: the volume read from file.

    Note:
        This requires a separate installation of the optional nibabel package, 
        either via 'pip install nibabel' or by installing vreg with the rw 
        option 'pip install vreg[rw]'.
    """
    if not nib_installed:
        raise ImportError(IMPORT_ERROR)
    img = nib.load(filepath)
    values = img.get_fdata()
    affine = _affine_to_from_RAH(img.affine)
    return Volume3D(values, affine)
