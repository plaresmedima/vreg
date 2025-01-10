import os
import sys
import pickle
import requests

from vreg import vol

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


def fetch(dataset: str) -> vol.Volume3D:
    """Fetch a dataset included in vreg

    Args:
        dataset (str): name of the dataset. See below for options.

    Returns:
        Volume3D: Data as a vreg.Volume3D.

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

    # If this is the first time the data are accessed, download them.
    if not os.path.exists(datafile):

        # Dataset location
        version_doi = "14630319" # This will change if a new version is created on zenodo
        file_url = "https://zenodo.org/records/" + version_doi + "/files/" + dataset + ".pkl"

        # Make the request and check for connection error
        try:
            file_response = requests.get(file_url) 
        except requests.exceptions.ConnectionError as err:
            raise requests.exceptions.ConnectionError(
                "\n\n"
                "A connection error occurred trying to download the test data \n"
                "from Zenodo. This usually happens if you are offline. The \n"
                "first time a dataset is fetched via vreg.fetch you need to \n"
                "be online so the data can be downloaded. After the first \n"
                "time they are saved locally so afterwards you can fetch \n"
                "them even if you are offline. \n\n"
                "The detailed error message is here: " + str(err)) 
        
        # Check for other errors
        file_response.raise_for_status()

        # Save the file locally 
        with open(datafile, 'wb') as f:
            f.write(file_response.content)

    with open(datafile, 'rb') as f:
        v = pickle.load(f)

    return v


def clear_cache():
    """
    Clear the folder where the data downloaded via fetch are saved.

    Note if you clear the cache the data will need to be downloaded again 
    if you need them.
    """

    f = importlib_resources.files('vreg.datafiles')
    for item in f.iterdir(): 
        if item.is_file(): 
            item.unlink() # Delete the file