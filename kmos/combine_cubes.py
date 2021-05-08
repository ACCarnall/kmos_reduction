import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import glob

from bagpipes.plotting import add_spectrum

from matplotlib.colors import Normalize
from astropy.io import fits
from astropy.wcs import WCS

from .utils import bin, bin_inv_var, bin_inv_var_reject_sky


class combine_cubes(object):

    def __init__(self, cube):

        self.cube = cube
        self.cube._load_single_cubes()
        self.n_cubes = len(self.cube.single_cubes)

        centroid_path = "centroids/" + self.cube.object_name + ".txt"
        if os.path.exists(centroid_path):
            self.centroids = np.loadtxt(centroid_path)

    def get_object_centroids(self, plot=False, radec=None, wcs_offsets=None):
        wcs_centroids = np.zeros((self.n_cubes, 2))

        centroids = np.zeros((self.n_cubes, 4))
        centroids[:, 0] = np.arange(self.n_cubes)
        centroids[:, -1] += 1

        # Get centroids from brightest pixel if no wcs offsets supplied
        if wcs_offsets is None:

            path = "centroids/" + self.cube.object_name + ".txt"

            if os.path.exists(path):
                centroids = np.loadtxt(path)
                print("Centroids file already exists, values loaded.")
                write = False

            else:
                for i in range(self.n_cubes):
                    cube = self.cube.single_cubes[i]
                    image = np.nanmedian(cube, axis=0)
                    image_crop = image[1:-1, 1:-1]

                    cent = np.unravel_index(image_crop.argmax(), image_crop.shape)
                    cent = np.array(cent)
                    cent += 1

                    centroids[i, 1:-1] = cent

                np.savetxt(path, centroids, fmt="%d")

        # Calculate centroid positions predicted by attached cube wcs
        if radec is not None:
            for i in range(self.n_cubes):
                wcs = self.cube.single_wcs[i]
                wcs_coords = wcs.all_world2pix(radec[0], radec[1], 1.2*10**-6, 0)
                wcs_coords = np.round(np.array(wcs_coords), 0)
                wcs_centroids[i, :] = wcs_coords[:-1][::-1]

        # If wcs_offsets supplied get centroids by adding to wcs_centroids
        if wcs_offsets is not None:
            centroids[:, 1:-1] = wcs_centroids + wcs_offsets

        # Plot collapsed cubes for manual object centroiding/testing
        if plot:
            for i in range(self.n_cubes):
                if centroids[i, -1] == 0:
                    continue

                print(i)
                cube = self.cube.single_cubes[i]
                image = np.nanmedian(cube, axis=0)
                image[:, [0, -1]] *= np.nan
                image[[0, -1], :] *= np.nan

                plt.figure()
                plt.imshow(image.T)
                plt.scatter(centroids[i, 1], centroids[i, 2], marker="+", color="green", lw=1, s=100)

                if radec is not None:
                    plt.scatter(wcs_centroids[i, 0], wcs_centroids[i, 1], marker="+", color="black", lw=1, s=100)

                plt.show()

        self.centroids = centroids
        self.wcs_centroids = wcs_centroids

    def recentre_single_cubes(self, radec=None, crop=1, plot=False, frame_mask=None):

        if frame_mask is not None:
            self.centroids[:, -1] = frame_mask

        recentred_cubes = np.zeros((2048, 31, 31, self.n_cubes))*np.nan
        centroids = self.centroids.astype(int)

        for i in range(self.n_cubes):
            if centroids[i, -1] == 0:  # Ignore cubes marked as no good
                print("skipping frame", i)
                continue

            ##### Crop edge pixels from individual exposure cube
            if crop != 0:
                self.cube.single_cubes[i] = self.cube.single_cubes[i][:, crop:-crop, crop:-crop]

            ##### Sky subtract individual exposure cube using its own sky
            obj_mask = self.make_object_mask(1.0, self.cube.single_cubes[i].shape, centre=self.centroids[i, 1:-1]-1.)  # Make object mask for sky sub

            sky_cube = self.cube.single_cubes[i]*np.invert(obj_mask.astype(bool))
            sky_cube[sky_cube == 0.] *= np.nan

            sky_1d = np.nanmedian(np.nanmedian(sky_cube, axis=-1), axis=-1)
            sky_1d = np.expand_dims(np.expand_dims(sky_1d, -1), -1)

            self.cube.single_cubes[i] -= sky_1d
            """
            ##### Flux normalise individual exposure cube using obj aperture
            norm_cube = self.cube.single_cubes[i]*obj_mask.astype(bool)
            norm_cube[norm_cube == 0.] *= np.nan

            norm_1d = np.nansum(np.nansum(norm_cube, axis=-1), axis=-1)

            wav_mask = (self.cube.wavs > 10500.) & (self.cube.wavs < 13200.)
            norm = np.nanmedian(norm_1d[wav_mask])

            self.cube.single_cubes[i] /= norm
            print(norm)
            """
            ##### Correctly position individual cube within array of all cubes
            shape = self.cube.single_cubes[i].shape[1:]

            recentred_cubes[:, 15-centroids[i, 1]+crop:15-centroids[i, 1]+shape[0]+crop,
                               15-centroids[i, 2]+crop:15-centroids[i, 2]+shape[1]+crop, i] = self.cube.single_cubes[i]

        self.recentred_cubes = recentred_cubes

        # Plot recentred individual cubes
        if plot:
            colno = 16
            rowno = len(self.cube.single_cubes)//colno + 1
            fig = plt.figure(figsize=(24*1.5, 1.5*2.*rowno))

            gs = mpl.gridspec.GridSpec(rowno, colno, wspace=0.1, hspace=0.)
            axes = []

            for i in range(len(self.cube.single_cubes)):

                column = int(np.round(i/float(colno) - i%colno/float(colno)))
                axes.append(plt.subplot(gs[column, i%colno]))
                plt.setp(axes[-1].get_xticklabels(), visible=False)
                plt.setp(axes[-1].get_yticklabels(), visible=False)

                # Sky normalise
                image = np.nanmedian(self.recentred_cubes[:, :, :, i], axis=0)
                if centroids[i, -1] == 0:
                    image -= np.nanmedian(image.flatten()[image.flatten() != 0.])

                axes[-1].imshow(image, cmap="binary_r")
                axes[-1].axhline(15, color="green", lw=1)
                axes[-1].axvline(15, color="green", lw=1)

            return fig, axes

    def stack_recentred_cubes(self, plot=False):

        # Calculate number of exposures in each spaxel
        exposures = np.sum(np.invert(np.isnan(np.nanmean(self.recentred_cubes, axis=0))), axis=-1)

        # Mask spaxels that do not have data from >10 per cent of exposures
        exp_mask = (exposures >= np.sum(self.centroids[:, -1])*0.9)

        if plot:
            plt.figure()
            plt.imshow(exp_mask)
            plt.show()

        # Stack cubes
        new_cube = np.nanmedian(self.recentred_cubes, axis=-1)

        # Calculate error cube by MAD estimator
        err_cube = 1.4826*np.nanmedian(np.abs(self.recentred_cubes - np.expand_dims(new_cube, axis=-1)), axis=-1)
        err_cube = err_cube/np.sqrt(np.sum(np.invert(np.isnan(np.nanmean(self.recentred_cubes, axis=0))), axis=-1))

        if plot:
            plt.figure()
            plt.imshow(np.nanmedian(new_cube, axis=0))
            plt.show()

        # Restrict stacked cube to spaxels with data from data from >90 per cent of exposures
        new_cube = new_cube*exp_mask
        new_cube[new_cube == 0.] *= np.nan

        if plot:
            plt.figure()
            plt.imshow(np.nanmedian(new_cube*exp_mask, axis=0))
            plt.show()

        self.new_cube = new_cube
        self.err_cube = err_cube

        hdus = [fits.PrimaryHDU(new_cube),
                fits.ImageHDU(err_cube),
                fits.ImageHDU(exposures)]

        for l in list(self.cube.cube_header):
            hdus[0].header[l] = self.cube.cube_header[l]

        self.new_cube_hdulist = fits.HDUList(hdus)

    def make_object_mask(self, diameter, dim, centre=[15, 15]):
        obj_mask = np.ones(dim).astype(int)

        r = diameter/0.2/2 # pixel aperture radius

        for i in range(obj_mask.shape[1]):
            for j in range(obj_mask.shape[2]):
                if (i - centre[0])**2 + (j - centre[1])**2 > r**2:
                    obj_mask[:, i, j] = 0

        return obj_mask

    def extract_1d_spec(self, plot=False, redshift=None):
        """
        obj_mask = self.make_object_mask(1.5, dim=self.new_cube.shape)  # Make object mask for sky sub

        sky_cube = self.new_cube*np.invert(obj_mask.astype(bool))
        sky_cube[sky_cube == 0.] *= np.nan

        if plot:
            plt.figure()
            plt.imshow(np.nanmedian(self.new_cube, axis=0))
            plt.show()

            plt.figure()
            plt.imshow(np.nanmedian(sky_cube, axis=0))
            plt.show()

        sky_1d = np.nanmedian(np.nanmedian(sky_cube, axis=-1), axis=-1)

        sky_1d = np.expand_dims(np.expand_dims(sky_1d, -1), -1)
        print(np.nanmedian(sky_1d), np.nanstd(sky_1d))
        self.new_cube -= sky_1d

        if plot:
            plt.figure()
            plt.imshow(np.nanmedian(self.new_cube, axis=0))
            plt.show()
        """
        # Make object mask for extraction
        obj_mask = self.make_object_mask(1.0, dim=self.new_cube.shape)

        self.new_cube *= obj_mask.astype(bool)
        self.new_cube[self.new_cube == 0.] *= np.nan

        self.err_cube *= obj_mask.astype(bool)
        self.err_cube[self.err_cube == 0.] *= np.nan

        if plot:
            plt.figure()
            plt.imshow(np.nanmedian(self.new_cube, axis=0))
            plt.show()

        spec_1d = np.nansum(np.nansum(self.new_cube, axis=-1), axis=-1)
        errs_1d = np.sqrt(np.nansum(np.nansum(self.err_cube**2, axis=-1), axis=-1))

        self.reduced_spectrum = np.c_[self.cube.wavs, spec_1d, errs_1d]

        if plot:
            fig, ax = self.cube.plot_1d_spec(spec_plot=np.c_[self.cube.wavs, spec_1d, errs_1d],
                                             bin_pixels=4, bin_method="inv_var", redshift=redshift, show=False)
            return fig, ax
