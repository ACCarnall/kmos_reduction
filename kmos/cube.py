import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import bagpipes as pipes

from matplotlib.colors import Normalize
from astropy.io import fits

from .utils import bin, bin_inv_var, bin_inv_var_reject_sky

features = ["$\\mathrm{H\\alpha}$", "$\\mathrm{H\\beta}$", "$\\mathrm{H\\delta}$",
            "$\\mathrm{Fe}$\\,\\textsc{i}", "$\\mathrm{Fe}$\\,\\textsc{i}",
            "$\\mathrm{Fe}$\\,\\textsc{i}", "$\\mathrm{Fe}$\\,\\textsc{i}",
            "$\\mathrm{Mg}$\\,\\textsc{i}", "$\\mathrm{Mg}$\\,\\textsc{uv}",
            "$\\mathrm{Ca}$\\,\\textsc{h,k}", "$\\mathrm{Na}$\\,\\textsc{i}",
            "$\\mathrm{TiO}$", "$\\mathrm{[O}\\,\\textsc{iii}\\mathrm{]}$"
            ]


feature_wavs = [6564.5, 4861.3, 4101.7,
                4531., 5015., 5270., 5335.,
                5177., 2800., 3925.,
                5896., 6233., 5007.]


class cube(object):

    def __init__(self, object_name, pointing, path):

        self.object_name = object_name
        self.pointing = pointing
        self.path = path

        image_hdu = fits.open(path + "/" + pointing + "_COMBINED_IMAGE_"
                              + object_name + ".fits")

        cube_hdu = fits.open(path + "/" + pointing + "_COMBINED_CUBE_"
                              + object_name + ".fits")

        self.image = image_hdu[1].data
        self.cube = cube_hdu[1].data
        self.err_cube = cube_hdu[2].data

        cube_header = cube_hdu[1].header
        max_wav = cube_header["CRVAL3"] + cube_header["CDELT3"]*2047
        self.wavs = 10000.*np.arange(cube_header["CRVAL3"], max_wav,
                                     cube_header["CDELT3"])

    def extract_1d_spec(self, centroid, diameter):
        r = diameter/0.2/2 # pixel aperture radius
        mask = np.ones(self.cube.shape).astype(int)

        for i in range(mask.shape[1]):
            for j in range(mask.shape[2]):
                if (i - centroid[0])**2 + (j - centroid[1])**2 > r**2:
                    mask[:, i, j] = 0

        self.spec1d = np.c_[self.wavs, np.nansum(self.cube*mask, axis=(2, 1)),
                            np.sqrt(np.nansum(mask*self.err_cube**2, axis=(2, 1)))]

        self.cube = self.cube*mask

    def plot_image(self, show=True, crop=0, centroid=None, collapsed_cube=False):
        fig = plt.figure()
        ax = plt.subplot()
        self.add_image(ax, crop=crop, centroid=centroid,
                       collapsed_cube=collapsed_cube)

        if show:
            plt.show()

        else:
            return fig, ax

    def add_image(self, ax, crop=0, centroid=None, collapsed_cube=False):
        norm = Normalize(vmin=-10**-19, vmax=2.*10**-19)

        if collapsed_cube:
            image_plot = np.nanmedian(self.cube, axis=0).T

        else:
            image_plot = self.image.T

        if crop:
            image_plot = image_plot[crop:-crop, crop:-crop]

        ax.imshow(image_plot, norm=norm, cmap="binary_r")

        if centroid is not None:
            ax.scatter(centroid[0], centroid[1], marker="+", color="red")

    def plot_1d_spec(self, bin_pixels=1, xlim=[10200., 13500.], redshift=None,
                     bin_method="standard", show=True, spec_plot=None):

        fig = plt.figure(figsize=(15, 5))
        ax = plt.subplot()

        if not spec_plot:
            spec_plot = self.spec1d

        wav_mask = (self.wavs > xlim[0]) & (self.wavs < xlim[1])
        spec_plot = spec_plot[wav_mask, :]

        if bin_pixels > 1 and bin_method == "standard":
            spec_plot = bin(spec_plot, bin_pixels)

        if bin_pixels > 1 and bin_method == "inv_var":
            spec_plot = bin_inv_var(spec_plot, bin_pixels)

        if bin_pixels > 1 and bin_method == "inv_var_reject_sky":
            spec_plot = bin_inv_var_reject_sky(spec_plot, bin_pixels)

        yscale = pipes.plotting.add_spectrum(spec_plot, ax,
                                             ymax=4*np.nanmedian(spec_plot[:, 1]))

        #ax.plot(spec_plot[:, 0], spec_plot[:, 2]*10**-yscale, color="green")

        if redshift:
            self._add_lines(redshift, ax, spec_plot, yscale=yscale, xlim=xlim)

        if show:
            plt.show()

        else:
            return fig, ax

    def plot_spaxel(self, x, y, bin_pixels=1, xlim=[10200., 13500.],
                    redshift=None, bin_method="standard", show=True):

        spec_plot = np.c_[self.wavs, self.cube[:, x, y],
                          self.err_cube[:, x, y]]

        self.plot_1d_spec(bin_pixels=bin_pixels, xlim=xlim, redshift=redshift,
                          bin_method=bin_method, show=show,
                          spec_plot=spec_plot)

    def _add_lines(self, z, ax, spectrum, yscale=-18, xlim=[10200., 13500.]):
        for i in range(len(features)):

            z_wav = feature_wavs[i]*(1.+z)
            if (z_wav < xlim[0]) or (z_wav > xlim[1]):
                continue

            ind = np.argmin(np.abs(feature_wavs[i] - spectrum[:, 0]/(1.+z)))

            if np.all(np.isnan(spectrum[ind-10:ind+11, 1])):
                continue

            y = np.nanmean(spectrum[ind-10:ind+11, 1]*10**-yscale) + 0.5

            ax.plot([feature_wavs[i]*(1.+z), feature_wavs[i]*(1.+z)],
                    [y, y + 0.25], color="black", lw=1.5)

            ax.annotate(features[i], (feature_wavs[i]*(1.+z), y + 0.45),
                        fontsize=11, rotation=90, verticalalignment="left",
                        horizontalalignment="center")
