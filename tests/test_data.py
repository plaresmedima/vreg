import numpy as np

import vreg
import vreg.plot as plt


def test_fetch():
    mask = vreg.fetch('left_kidney')
    assert 1504 == np.sum(mask.values[:,:,2])
    vreg.fetch(clear_cache=True)
    vreg.fetch(download_all=True)
    vreg.fetch(download_all=True)


if __name__ == '__main__':

    #test_fetch()
    vreg.fetch(clear_cache=True)