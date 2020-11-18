from matplotlib import colors, patches
from skimage.feature import peak_local_max
import pandas as pd
from skimage import io
from skimage.measure import label, regionprops, regionprops_table
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# function for post-processing (size filter)
from skimage.morphology import (  
    ball,
    dilation,
    erosion,
    remove_small_objects,
    watershed,
)




def plot_flash_zdr(flash_dataset, segmented_files, times, zdr_df, cmap=None, cbar_label = None, alpha=None,vmin=None,vmax=None):
    """
    This function generates overlay plots of gridded flash products and
    ZDR columns (surrounded by their respective bounded boxes)
    
    Input
    ----------
    
    flash_dataset : xarray DataArray
                Gridded flash product to be plotted as pcolormesh plot
    segmented_files : list
                List of all files processed and segmented for ZDR column identification 
    times : datetime objects
        datetime for files of interest (retrieved from original radar file list)
    zdr_df : pandas dataframe
        dataframe containing info about bbox, area, and label of each ZDR column object at each time
    cmap : str
        colormap for gridded flash product
    cbar_label : str
            text label for pcolormesh colorbar 
    alpha : float
        float number between 0 and 1 to set the transparency of pcolormesh plot
    vmin : float
        lowest value of pcolormesh data to be mapped to first rgb combination in cmap color list
    vmax : float
        highest value of pcolormesh data to be mapped to last rgb combination in cmap color list
        
    Returns
    ----------
    
    matplotlib plots
    """
    flash_dataset = flash_dataset
    final_zdr_df = zdr_df
    dt_tmpstmps = times
    cmap = cmap
    
    for i in range(len(flash_dataset.ntimes)):
        fig, ax = plt.subplots(1)
        tidx = dt_tmpstmps[i]
        obj_bounds = pd.DataFrame()

        cell_seg_reader = io.imread(segmented_files[i])
        cell_seg = cell_seg_reader
        seg = label(cell_seg)
        seg = np.fliplr(seg)

        ax.pcolormesh(seg[0, :, :], cmap="Greys", alpha=0.7)
        fig = ax.pcolormesh(
            flash_dataset[i, :, :],
            cmap=cmap,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
        )

        try:
            obj_bounds = pd.DataFrame(final_zdr_df.loc[tidx])

            if len(obj_bounds.columns) == 1:
                obj = obj_bounds.transpose()
                rect = patches.Rectangle(
                    (obj["bbox-2"][0], obj["bbox-1"][0]),
                    obj["bbox-5"][0] - obj["bbox-2"][0],
                    obj["bbox-4"][0] - obj["bbox-1"][0],
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                )
                ax.annotate(
                    obj["label"][0],
                    (obj["bbox-5"][0], obj["bbox-4"][0]),
                    color="k",
                    weight="bold",
                    fontsize=10,
                    ha="center",
                    va="center",
                )
                # Add the patch to the Axes
                ax.add_patch(rect)
                cbar = plt.colorbar(mappable=fig)
                cbar.set_label(cbar_label,
                    fontsize=20,
                    labelpad=16,
                )
                ratio = 1
                xleft, xright = ax.get_xlim()
                ybottom, ytop = ax.get_ylim()
                # the abs method is used to make sure that all numbers are positive
                # because x and y axis of an axes maybe inversed.
                ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)
                ax.set_xticklabels(
                    ["-75", "-50", "-25", "0", "25"]
                )  # since #grid cells = 241 correspond to half the distance dx=dy=500m
                ax.set_yticklabels(["0", "25", "50", "75", "100"])
                ax.set_xlabel("Distance from KTLX <- West (km) East ->")
                ax.set_ylabel("Distance from KTLX <- South (km) North ->")
                fsavename = segmented_files[i].split("/")[-1].split(".")[0]
                raw_title = datetime.strptime(
                    segmented_files[i].split("/")[-1].split(".")[0], "%H%M%S"
                )
                final_title = datetime.strftime(raw_title, "%H%M:%S") + " UTC"

                plt.title(f"Time = {final_title}")
                plt.show()

            if len(obj_bounds.columns) > 1:
                for j in range(len(obj_bounds)):
                    obj = pd.DataFrame(obj_bounds.iloc[j]).transpose()
                    rect = patches.Rectangle(
                        (obj["bbox-2"][0], obj["bbox-1"][0]),
                        obj["bbox-5"][0] - obj["bbox-2"][0],
                        obj["bbox-4"][0] - obj["bbox-1"][0],
                        linewidth=1,
                        edgecolor="r",
                        facecolor="none",
                    )
                    ax.annotate(
                        obj["label"][0],
                        (obj["bbox-5"][0], obj["bbox-4"][0]),
                        color="k",
                        weight="bold",
                        fontsize=10,
                        ha="center",
                        va="center",
                    )
                    # Add the patch to the Axes
                    ax.add_patch(rect)
                #                 ax.colorbar()
                cbar = plt.colorbar(mappable=fig)
                cbar.set_label(cbar_label,
                    fontsize=20,
                    labelpad=16,
                )
                ratio = 1
                xleft, xright = ax.get_xlim()
                ybottom, ytop = ax.get_ylim()
                # the abs method is used to make sure that all numbers are positive
                # because x and y axis of an axes maybe inversed.
                ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)
                ax.set_xticklabels(
                    ["-75", "-50", "-25", "0", "25"]
                )  # since #grid cells = 241 correspond to half the distance dx=dy=500m
                ax.set_yticklabels(["0", "25", "50", "75", "100"])
                ax.set_xlabel("Distance from KTLX <- West (km) East ->")
                ax.set_ylabel("Distance from KTLX <- South (km) North ->")
                fsavename = segmented_files[i].split("/")[-1].split(".")[0]
                raw_title = datetime.strptime(
                    segmented_files[i].split("/")[-1].split(".")[0], "%H%M%S"
                )
                final_title = datetime.strftime(raw_title, "%H%M:%S") + " UTC"

                plt.title(f"Time = {final_title}")
                plt.show()

        except KeyError as error:  # happens when  obj_pounds is empty (zero entries) i.e. no ZDR objects to identify in our data
            continue