import numpy as np
import matplotlib.pyplot as plt


def bin(spectrum, binn):
    """ Bins up a two or three column spectrum by a given factor. """

    binn = int(binn)
    nbins = int(len(spectrum)/binn)
    binspec = np.zeros((nbins, spectrum.shape[1]))
    for i in range(binspec.shape[0]):
        binspec[i, 0] = np.mean(spectrum[i*binn:(i+1)*binn, 0])
        binspec[i, 1] = np.mean(spectrum[i*binn:(i+1)*binn, 1])
        if spectrum.shape[1] == 3:
            sq_sum = np.sum(spectrum[i*binn:(i+1)*binn, 2]**2)
            binspec[i, 2] = (1./float(binn))*np.sqrt(sq_sum)

    return binspec


def bin_inv_var(spectrum, binn):
    """ Bins up a two or three column spectrum by a given factor. """

    binn = int(binn)
    nbins = int(len(spectrum)/binn)
    binspec = np.zeros((nbins, spectrum.shape[1]))
    for i in range(binspec.shape[0]):
        spec_slice = spectrum[i*binn:(i+1)*binn, :]
        binspec[i, 0] = np.mean(spec_slice[:, 0])
        binspec[i, 2] = np.sqrt(1./np.sum(1./spec_slice[:, 2]**2))
        binspec[i, 1] = np.sum(spec_slice[:, 1]/spec_slice[:, 2]**2)*binspec[i, 2]**2

    return binspec


def bin_inv_var_reject_sky(spectrum, binn, rejection_threshold=2):
    """ Bins up a two or three column spectrum by a given factor. """

    sky_mask = (spectrum[:, 2] < rejection_threshold*np.median(spectrum[:, 2]))

    """
    fig = plt.figure(figsize=(15, 5))
    ax = plt.subplot()

    ax.plot(spectrum[:, 0], spectrum[:, 2], color="green")
    ax.scatter(spectrum[np.invert(sky_mask), 0],
               spectrum[np.invert(sky_mask), 2], color="black", marker="x")

    ax.axhline(2*np.median(spectrum[:, 2]), color="black", zorder=10)

    plt.show()
    """

    binn = int(binn)
    nbins = int(len(spectrum)/binn)
    binspec = np.zeros((nbins, spectrum.shape[1]))

    for i in range(binspec.shape[0]):
        spec_slice = spectrum[i*binn:(i+1)*binn, :]
        mask_slice = sky_mask[i*binn:(i+1)*binn]

        binspec[i, 0] = np.mean(spec_slice[:, 0])

        if np.sum(mask_slice) < 2:
            binspec[i, 1] = 0.
            binspec[i, 2] = 9.9*10**99
            continue

        binspec[i, 2] = np.sqrt(1./np.sum(1./spec_slice[mask_slice, 2]**2))
        binspec[i, 1] = np.sum(spec_slice[mask_slice, 1]/spec_slice[mask_slice, 2]**2)*binspec[i, 2]**2

    return binspec


def reject_sky(spectrum, rejection_threshold=2):
    """ Mask sky pixels. """

    sky_mask = (spectrum[:, 2] > rejection_threshold*np.median(spectrum[:, 2]))

    """
    fig = plt.figure(figsize=(15, 5))
    ax = plt.subplot()

    ax.plot(spectrum[:, 0], spectrum[:, 2], color="green")
    ax.scatter(spectrum[sky_mask, 0],
               spectrum[sky_mask, 2], color="black", marker="x")

    ax.axhline(2*np.median(spectrum[:, 2]), color="black", zorder=10)

    plt.show()
    """
    spectrum[sky_mask, 1] = np.nan
    spectrum[sky_mask, 2] = 9.9*10**99

    return spectrum
