import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def make_object_mask(diameter, dim, centre=[15, 15]):
    obj_mask = np.ones(dim).astype(int)

    r = diameter/0.2/2 # pixel aperture radius

    for i in range(obj_mask.shape[1]):
        for j in range(obj_mask.shape[2]):
            if (i - centre[1])**2 + (j - centre[0])**2 > r**2:
                    obj_mask[:, i, j] = 0

    return obj_mask


def extract_1d_spec(hdulist, plot=False, redshift=None, centre=[15, 15], diameter=1.):

    cube = hdulist[0].data
    err_cube = hdulist[1].data

    header = hdulist[0].header

    # Find the wavelength sampling of the cube
    max_wav = header["CRVAL3"] + header["CDELT3"]*2047
    wavs = 10000.*np.arange(header["CRVAL3"], max_wav, header["CDELT3"])

    if plot:
        plt.figure()
        plt.imshow(np.nanmedian(cube, axis=0))
        plt.show()

    # Make object mask for extraction
    print(centre)
    obj_mask = make_object_mask(diameter, dim=cube.shape, centre=centre)

    cube *= obj_mask.astype(bool)
    cube[cube == 0.] *= np.nan

    err_cube *= obj_mask.astype(bool)
    err_cube[err_cube == 0.] *= np.nan

    if plot:
        plt.figure()
        plt.imshow(np.nanmedian(cube, axis=0))
        plt.show()

    spec_1d = np.nansum(np.nansum(cube, axis=-1), axis=-1)
    errs_1d = np.sqrt(np.nansum(np.nansum(err_cube**2, axis=-1), axis=-1))

    spectrum = np.c_[wavs, spec_1d, errs_1d]

    return spectrum
