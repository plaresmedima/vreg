import sys
import pickle

# filepaths need to be identified with importlib_resources
# rather than __file__ as the latter does not work at runtime
# when the package is installed via pip install

if sys.version_info < (3, 9):
    # importlib.resources either doesn't exist or lacks the files()
    # function, so use the PyPI version:
    import importlib_resources
else:
    # importlib.resources has files(), so use that:
    import importlib.resources as importlib_resources


def fetch(dataset: str) -> dict:
    """Fetch a dataset included in vreg

    Args:
        dataset (str): name of the dataset. See below for options.

    Returns:
        dict: Data as a dictionary.

    Notes:

        The following datasets are currently available:

        **iBEAt**

            **Background**: data are provided by the imaging work package of the 
            `BEAt-DKD project <https://www.beat-dkd.eu>`_ .

            **Data format**: The fetch function returns a list of dictionaries, 
            one per subject visit. Each dictionary contains the following items:

            - **item1**: description.
            - **item2**: description.

            Please reference the following paper when using these data:

            Gooding et al.

    """

    f = importlib_resources.files('vreg.datafiles')
    datafile = str(f.joinpath(dataset + '.pkl'))

    with open(datafile, 'rb') as fp:
        data_dict = pickle.load(fp)

    return data_dict