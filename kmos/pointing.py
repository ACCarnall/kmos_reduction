import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os

from matplotlib.colors import Normalize
from astropy.io import fits

from .cube import cube


class pointing(object):

    def __init__(self, pointing_name, data_path, ins_file_path):

        self.pointing_name = pointing_name
        self.data_path = data_path
        self.ins_file_path = ins_file_path

        self._create_catalogue(ins_file_path)

        for i in range(1, 25):
            if not pd.isnull(self.cat.loc[i, "object"]):
                self.cat.loc[i, "cube"] = cube(self.cat.loc[i, "object"],
                                               pointing_name, data_path)

    def _create_catalogue(self, ins_file_path):
        cat = pd.DataFrame(np.arange(1, 25), columns=["arm"],
                           index=np.arange(1, 25))

        for name in ["object", "mag", "band", "priority", "cube"]:
            cat[name] = np.repeat(np.nan, 24)

        f = open(ins_file_path)

        for l in f:
            line = l.split()
            if len(line) > 0 and line[0].endswith(".SCI.NAME"):
                armno = int(line[0].split(".")[1][3:])
                cat.loc[armno, "object"] = line[1][1:-1]

            if len(line) > 0 and line[0].endswith(".SCI.MAG"):
                armno = int(line[0].split(".")[1][3:])
                cat.loc[armno, "mag"] = float(line[1])

            if len(line) > 0 and line[0].endswith(".SCI.BAND"):
                armno = int(line[0].split(".")[1][3:])
                cat.loc[armno, "band"] = line[1][1:-1]

            if len(line) > 0 and line[0].endswith(".SCI.PRIOR"):
                armno = int(line[0].split(".")[1][3:])
                cat.loc[armno, "priority"] = int(line[1])

        self.cat = cat

    def plot_images(self, crop=0, centroids=None, collapsed_cube=False,
                    show=True, single_cube=None, plot_coords=False):

        fig = plt.figure(figsize=(16, 8))
        rowno = 8
        gs = mpl.gridspec.GridSpec(3, rowno, wspace=0.25, hspace=0.)
        axes = []

        for i in range(24):

            obj = self.cat.loc[i+1, "object"]
            if pd.isnull(obj):
                continue

            colno = int(np.round(i/float(rowno) - i%rowno/float(rowno)))
            axes.append(plt.subplot(gs[colno, i%rowno]))
            plt.setp(axes[-1].get_xticklabels(), visible=False)
            plt.setp(axes[-1].get_yticklabels(), visible=False)

            if centroids is not None:
                cent = centroids.loc[obj, ["x", "y"]].values

            else:
                cent = None

            cube = self.cat.loc[i+1, "cube"]

            if plot_coords:
                coords = centroids.loc[obj, ["RA", "DEC"]]

            else:
                coords = None

            if single_cube is None:
                image = None

            else:
                print(cube.single_cubes[single_cube].shape)
                image = np.median(cube.single_cubes[single_cube], axis=0)
                print(image)
                image = np.nanmedian(cube.single_cubes[single_cube], axis=0)

            cube.add_image(axes[-1], image=image, crop=crop, coords=cent)

            if obj.endswith("SELECT"):
                axes[-1].set_title(obj[:-6], fontsize=9)

            else:
                axes[-1].set_title(obj, fontsize=9)

            axes[-1].set_xlabel("$" + self.cat.loc[i+1, "band"] + " = "
                                + str(self.cat.loc[i+1, "mag"]) + "$",
                                fontsize=10)

        if show:
            plt.show()

        else:
            return fig, axes

    def extract_1d_spectra(self, centroids, diameter):
        for i in range(24):

            obj = self.cat.loc[i+1, "object"]
            if pd.isnull(obj):
                continue

            cent = centroids.loc[obj, ["x", "y"]].values
            self.cat.loc[i+1, "cube"].extract_1d_spec(cent, diameter)

    def load_single_cubes(self):

        for i in range(24):
            obj = self.cat.loc[i+1, "object"]
            if pd.isnull(obj):
                continue

            self.cat.loc[i+1, "cube"]._load_single_cubes()

    def plot_single_cubes(self, centroids=None, plot_coords=False):
        if not os.path.exists("single_cube_plots_" + self.pointing_name):
            os.mkdir("single_cube_plots_" + self.pointing_name)

        n_cubes = len(self.cat.loc[1, "cube"].single_cubes)

        for i in range(n_cubes):
            self.plot_images(single_cube=i+1, centroids=centroids, show=False,
                             plot_coords=plot_coords)

            plt.savefig("single_cube_plots_" + self.pointing_name + "/cube_"
                        + str(i+1).rjust(3, "0") + "_"
                        + self.cat.loc[1, "cube"].cube_names[i]
                        + ".pdf", bbox_inches="tight")

            plt.close()
