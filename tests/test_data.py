import vreg
import vreg.plot as plt


def test_fetch():
    img = vreg.fetch('Dixon_out_phase')
    img = img.extract_slice(100)
    mask = vreg.fetch('left_kidney')
    plt.overlay_2d(img, mask)

def test_clear_cache():
    vreg.clear_cache()


if __name__ == '__main__':

    test_fetch()
    #test_clear_cache()