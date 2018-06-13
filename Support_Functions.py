import numpy as np
from scipy import (interpolate)
from sklearn import (decomposition, preprocessing as pre, cluster, neighbors)
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import (pyplot as plt, animation, colors,
                        ticker, path, patches, patheffects)
import string
import os
from os.path import join as pjoin

import keras
from keras.models import Sequential, Input, Model
from keras.layers import (Dense, Conv1D, Convolution2D, GRU, LSTM, Recurrent, Bidirectional, TimeDistributed,
                          Dropout, Flatten, RepeatVector, Reshape, MaxPooling1D, UpSampling1D, BatchNormalization)
from keras.layers.core import Lambda
from keras.optimizers import Adam
from keras.regularizers import l1

import glob
import sys
import re

from scipy.signal import savgol_filter as sg

import moviepy as mpy

#import moviepy.video.io.ImageSequenceClip
#try:
#    output = subprocess.check_output(['ffmpeg', '-version'])
#    version = output.split(b'\n')[0].split()[2]
#    print('Found: ffmpeg v{}'.format(version.decode('utf-8')))
#    ffmpeg_installed = True
#except:
#    ffmpeg_installed = False

Path = path.Path
PathPatch = patches.PathPatch

# Defines a set of custom colormaps used
cmap_2 = colors.ListedColormap(['#003057',
                                      '#FFBD17'])
cmap_3 = colors.ListedColormap(['#003057', '#1b9e77', '#d95f02'])
cmap_4 = colors.ListedColormap(['#f781bf', '#e41a1c',
                                 '#4daf4a', '#003057'])
cmap_5 = colors.ListedColormap(['#003057', '#1b9e77', '#d95f02',
                                 '#7570b3', '#e7298a'])
cmap_6 = colors.ListedColormap(['#003057', '#1b9e77', '#d95f02',
                                 '#7570b3', '#e7298a', '#66a61e'])
custom_cmap = ['#e41a1c', '#003057', '#4daf4a',
                 '#a65628', '#984ea3', '#ff7f00',
                 '#ffff33', '#f781bf', '#377eb8']
cmap_9 = colors.ListedColormap(['#e41a1c',  '#f781bf', '#003057',
                 '#a65628', '#984ea3','#377eb8',
                 '#f46d43','#cab2d6', '#4daf4a'])

def interpolate_missing_points(data, fit_type='spline'):
    """
    Interpolates bad pixels in piezoelectric hystereis loops.\n
    The interpolation of missing points alows for machine learning operations

    Parameters
    ----------
    data : numpy array
        array of loops
    fit_type : string
        selection of type of function for interpolation

    Returns
    -------
    data_cleaned : numpy array
        arary of loops
    """

    # reshapes the data such that it can run with different data sizes
    if data.ndim == 2:
        data = data.reshape(np.sqrt(data.shape[0]).astype(int),
                            np.sqrt(data.shape[0]).astype(int), -1)
        data = np.expand_dims(data, axis=3)
    elif data.ndim == 3:
        data = np.expand_dims(data, axis=3)

    # creates a vector of the size of the data
    point_values = np.linspace(0, 1, data.shape[2])

    # Loops around the x index
    for i in range(data.shape[0]):

        # Loops around the y index
        for j in range(data.shape[1]):

            # Loops around the number of cycles
            for k in range(data.shape[3]):

                if any(~np.isfinite(data[i, j, :, k])):

                    # selects the index where values are nan
                    ind = np.where(np.isnan(data[i, j, :, k]))

                    # if the first value is 0 copies the second value
                    if 0 in np.asarray(ind):
                        data[i, j, 0, k] = data[i, j, 1, k]

                    # selects the values that are not nan
                    true_ind = np.where(~np.isnan(data[i, j, :, k]))

                    # for a spline fit
                    if fit_type == 'spline':
                        spline = interpolate.InterpolatedUnivariateSpline(point_values[true_ind],
                                                                          data[i, j, true_ind, k].squeeze())
                        data[i, j, ind, k] = spline(point_values[ind])

                    # for a linear fit
                    elif fit_type == 'linear':

                        data[i, j, :, k] = np.interp(point_values,
                                                     point_values[true_ind],
                                                     data[i, j, true_ind, k].squeeze())

    return data.squeeze()

def sg_filter_data(data, num_to_remove=3, window_length=7, polyorder=3, fit_type='spline'):
    """
    Applies a Savitzky-Golay filter to the data which is used to remove outlier or noisy points from the data

    Parameters
    ----------
    data : numpy, array
        array of loops
    num_to_remove : numpy, int
        sets the number of points to remove
    window_length : numpy, int
        sets the size of the window for the sg filter
    polyorder : numpy, int
        sets the order of the sg filter
    fit_type : string
        selection of type of function for interpolation

    Returns
    -------
    cleaned_data : numpy array
        array of loops
    """
    # reshapes the data such that it can run with different data sizes
    if data.ndim == 2:
        data = data.reshape(np.sqrt(data.shape[0]).astype(int),
                            np.sqrt(data.shape[0]).astype(int), -1)
        data = np.expand_dims(data, axis=3)
    elif data.ndim == 3:
        data = np.expand_dims(data, axis=3)

    cleaned_data = np.copy(data)

    # creates a vector of the size of the data
    point_values = np.linspace(0, 1, data.shape[2])

    # Loops around the x index
    for i in range(data.shape[0]):

        # Loops around the y index
        for j in range(data.shape[1]):

            # Loops around the number of cycles
            for k in range(data.shape[3]):

                sg_ = sg(data[i, j, :, k],
                         window_length=window_length, polyorder=polyorder)
                diff = np.abs(data[i, j, :, k] - sg_)
                sort_ind = np.argsort(diff)
                remove = sort_ind[-1 * num_to_remove::].astype(int)
                cleaned_data[i, j, remove, k] = np.nan

    cleaned_data = clean_and_interpolate(cleaned_data, fit_type)

    return cleaned_data

#def interpolate_missing_points(data):
#    """
#    Interpolates bad pixels in piezoelectric hystereis loops.\n
#    The interpolation of missing points alows for machine learning operations
#
#    Parameters
#    ----------
#    data : numpy array
#        arary of loops
#
#    Returns
#    -------
#    data_cleaned : numpy array
#        arary of loops
#    """
#
#    # reshapes the data such that it can run with different data sizes
#    if data.ndim == 2:
#        data = data.reshape(np.sqrt(data.shape[0]),
#                                      np.sqrt(data.shape[0]), -1)
#        data = np.expand_dims(data, axis=0)
#    elif data.ndim == 3:
#        data = np.expand_dims(data, axis=0)
#
#    # Loops around the x index
#    for i in range(data.shape[0]):
#
#        # Loops around the y index
#        for j in range(data.shape[1]):
#
#            # Loops around the number of cycles
#            for k in range(data.shape[3]):
#
#                if any(~np.isfinite(data[i, j, :, k])):
#
#                    true_ind = np.where(~np.isnan(data[i, j, :, k]))
#                    point_values = np.linspace(0, 1, data.shape[2])
#                    spline = interpolate.InterpolatedUnivariateSpline(point_values[true_ind],
#                                                                      data[i, j, true_ind, k].squeeze())
#                    ind = np.where(np.isnan(data[i, j, :, k]))
#                    val = spline(point_values[ind])
#                    data[i, j, ind, k] = val
#
#    return data.squeeze()

def conduct_PCA(loops, n_components=15, verbose=True):
    """
    Computes the PCA and forms a low-rank representation of a series of response curves
    This code can be applied to all forms of response curves.
    loops = [number of samples, response spectra for each sample]

    Parameters
    ----------
    loops : numpy array
        1 or 2d numpy array - [number of samples, response spectra for each sample]
    n_components : int, optional
        int - sets the number of componets to save
    verbose : bool, optional
        output operational comments

    Returns
    -------
    PCA : object
        results from the PCA
    PCA_reconstructed : numpy array
        low-rank representation of the raw data reconstructed based on PCA denoising
    """

    # resizes the array for hyperspectral data
    if loops.ndim == 3:
        original_size = loops.shape[0]
        loops = loops.reshape(-1, loops.shape[2])
        verbose_print(verbose, 'shape of data resized to [{0}x {1}]'.format(
            loops.shape[0], loops.shape[1]))
    elif loops.ndim == 2:
        pass
    else:
        raise ValueError("data is of an incorrect size")

    if np.isnan(loops).any():
        raise ValueError(
            'data has infinite values consider using a imputer \n see interpolate_missing_points function')

    # Sets the number of components to save
    pca = decomposition.PCA(n_components=n_components)

    # Computes the PCA of the piezoelectric hysteresis loops
    PCA = pca.fit(loops)

    # does the inverse transform - creates a low rank representation of the data
    # this process denoises the data
    PCA_reconstructed = pca.inverse_transform(pca.transform(loops))

    # resized the array for hyperspectral data
    try:
        PCA_reconstructed = PCA_reconstructed.reshape(original_size, original_size, -1)
    except:
        pass

    return PCA, PCA_reconstructed

def _maps(pca, loops, add_colorbars=True, verbose=False, letter_labels=False,
                  add_scalebar=False, filename='./PCA_maps', print_EPS=False,
                  print_PNG=False, dpi=300, num_of_plots=True):
    """
    Adds a colorbar to a imageplot

    Parameters
    ----------
    pca : model
        previously computed pca
    loops : numpy, array
        data to plot
    add_colorbars : bool, optional
        adds colorbars to images
    verbose : bool, optional
        sets the verbosity level
    letter_labels : bool, optional
        adds letter labels for use in publications
    add_scalebar : bool, optional
        sets whether a scalebar is added to the maps
    filename : str, optional
        sets the path and filename for the exported images
    print_EPS : bool, optional
        to export as EPS
    print_PNG : bool, optional
        to export as PNG
    dpi : int, optional
        resolution of exported image
    num_of_plots : int, optional
            number of principle componets to show
    """
    if num_of_plots == True:
        num_of_plots = pca.n_components_

    # creates the figures and axes in a pretty way
    fig, ax = layout_graphs_of_arb_number(num_of_plots)

    # resizes the array for hyperspectral data
    if loops.ndim == 3:
        original_size = loops.shape[0]
        loops = loops.reshape(-1, loops.shape[2])
        verbose_print(verbose, 'shape of data resized to [{0} x {1}]'.format(
            loops.shape[0], loops.shape[1]))
    elif loops.ndim == 2:
        original_size = np.sqrt(loops.shape[0]).astype(int)
    else:
        raise ValueError("data is of an incorrect size")

    PCA_maps = pca_weights_as_embeddings(pca, loops, num_of_components=num_of_plots)

    for i in range(num_of_plots):
        im = ax[i].imshow(PCA_maps[:, i].reshape(original_size, original_size))
        ax[i].set_yticklabels('')
        ax[i].set_xticklabels('')
        #

        if add_colorbars:
            add_colorbar(ax[i], im)

        # labels figures
        if letter_labels:
            labelfigs(ax[i], i)
        labelfigs(ax[i], i, string_add='PC {0}'.format(i + 1), loc='bm')

        if add_scalebar is not False:
            add_scalebar_to_figure(ax[i], add_scalebar[0], add_scalebar[1])

    plt.tight_layout(pad=0, h_pad=0)

    savefig(filename, dpi=300, print_EPS=print_EPS, print_PNG=print_PNG)

def plot_pca_results(pca, loops, voltage, signal_info, add_colorbars=True, verbose=False, letter_labels=False,
                  add_scalebar=False, filename='./PCA_maps', print_EPS=False,
                  print_PNG=False, dpi=300, num_of_plots=True):

    min_ = np.min(pca.components_.reshape(-1))
    max_ = np.max(pca.components_.reshape(-1))

    if num_of_plots == True:
        num_of_plots = pca.n_components_

    # stores the number of plots in a row
    mod = num_of_plots//(np.sqrt(num_of_plots)//1).astype(int)
    # creates the figures and axes in a pretty way
    fig, ax = layout_graphs_of_arb_number(num_of_plots*2, mod = mod)

    # resizes the array for hyperspectral data
    if loops.ndim == 3:
        original_size = loops.shape[0]
        loops = loops.reshape(-1, loops.shape[2])
        verbose_print(verbose, 'shape of data resized to [{0} x {1}]'.format(
            loops.shape[0], loops.shape[1]))
    elif loops.ndim == 2:
        original_size = np.sqrt(loops.shape[0]).astype(int)
    else:
        raise ValueError("data is of an incorrect size")

    # computes the PCA maps
    PCA_maps = pca_weights_as_embeddings(pca, loops, num_of_components=num_of_plots)

    # Formats figures
    for i, ax in enumerate(ax):

        # Checks if axes is an image or a plot
        if (i // mod % 2 == 0):
            pc_number = i - mod * (i // (mod*2))
            plot_imagemap(ax, PCA_maps[:, pc_number], color_bar = add_colorbars)
            # labels figures
            labelfigs(ax, i, string_add='PC {0:d}'.format(pc_number+1), loc='bm')
            add_scalebar_to_figure(ax, 2000, 500)

        else:

            # Plots the PCA egienvector and formats the axes
            ax.plot(voltage, pca.components_[
                    i - mod - ((i // mod) // 2) * mod], 'k')

            # Formats and labels the axes
            ax.set_xlabel('Voltage')
            ax.set_ylabel(signal_info['y_label'])
            ax.set_yticklabels('')
            ax.set_ylim([min_,max_])

            if signal_info['y_lim'] is not None:
                ax.set_ylim(signal_info['y_lim'])

        # labels figures
        if letter_labels:
            if (i // mod % 2 == 0):
                labelfigs(ax, pc_number-1)

        set_axis_aspect(ax)

    plt.tight_layout(pad=0, h_pad=0)

    savefig(filename, dpi=300, print_EPS=print_EPS, print_PNG=print_PNG)


def layout_graphs_of_arb_number(graph, mod=None):
    """
    Sets the layout of graphs in matplotlib in a pretty way based on the number of plots

    Parameters
    ----------
    graphs : int
        number of axes to make

    Returns
    -------
    fig : matplotlib figure
        handel to figure being created.
    axes : numpy array (axes)
        numpy array of axes that are created.
    """

    if mod == None:
        # Selects the number of columns to have in the graph
        if graph < 3:
            mod = 2
        elif graph < 5:
            mod = 3
        elif graph < 10:
            mod = 4
        elif graph < 17:
            mod = 5
        elif graph < 26:
            mod = 6
        elif graph < 37:
            mod = 7

    # builds the figure based on the number of graphs and selected number of columns
    fig, axes = plt.subplots(graph // mod + (graph % mod > 0), mod,
                             figsize=(3 * mod, 3 * (graph // mod + (graph % mod > 0))))

    # deletes extra unneeded axes
    axes = axes.reshape(-1)
    for i in range(axes.shape[0]):
        if i + 1 > graph:
            fig.delaxes(axes[i])

    return (fig, axes)

def pca_weights_as_embeddings(pca, loops, num_of_components=0, verbose=True):
    """
    Computes the eigenvalue maps computed from PCA

    Parameters
    ----------
    pca : object
        computed PCA
    loops: numpy array
        raw piezoresponse data
    num_of _components: int
        number of PCA components to compute

    Returns
    -------
    fig : matplotlib figure
        handel to figure being created.
    axes : numpy array (axes)
        numpy array of axes that are created.
    """
    if loops.ndim == 3:
        loops = loops.reshape(-1, loops.shape[2])
        verbose_print(verbose, 'shape of data resized to [{0} x {1}]'.format(
            loops.shape[0], loops.shape[1]))

    if num_of_components == 0:
        num_of_components = pca.n_components_

    PCA_embedding = pca.transform(loops)[:, 0:num_of_components]

    return (PCA_embedding)

def verbose_print(verbose, *args):
    if verbose:
        print(*args)

def add_colorbar(axes, plot, location='right', size=10, pad=0.05, format='%.1e', ticks = True):
    """
    Adds a colorbar to a imageplot

    Parameters
    ----------
    axes : matplotlib axes
        axes which to add the plot to
    axes : matplotlib plot
        Plot being references for the scalebar
    location : str, optional
        position to place the colorbar
    size : int, optional
        percent size of colorbar realitive to the plot
    pad : float, optional
        gap between colorbar and plot
    format : str, optional
        string format for the labels on colorbar
    """

    # Adds the scalebar
    divider = make_axes_locatable(axes)
    cax = divider.append_axes(location, size='{0}%'.format(size), pad=pad)
    cbar = plt.colorbar(plot, cax=cax, format=format)

    if not ticks:
        cbar.set_ticks([])

def labelfigs(axes, number, style='wb', loc='br', string_add='', size=14, text_pos='center'):
    """
    Adds labels to figures

    Parameters
    ----------
    axes : matplotlib axes
        axes which to add the plot to
    number : int
        letter number
    style : str, optional
        sets the color of the letters
    loc : str, optional
        sets the location of the label
    string_add : str, optional
        custom string as the label
    size : int, optional
        sets the fontsize for the label
    text_pos : str, optional
        set the justification of the label
    """

    # Sets up various color options
    formating_key = {'wb': dict(color='w',
                                linewidth=1.5),
                     'b': dict(color='k',
                               linewidth=0),
                     'w': dict(color='w',
                               linewidth=0)}

    # Stores the selected option
    formatting = formating_key[style]

    # finds the position for the label
    x_min, x_max = axes.get_xlim()
    y_min, y_max = axes.get_ylim()
    x_value = .08 * (x_max - x_min) + x_min

    if loc == 'br':
        y_value = y_max - .1 * (y_max - y_min)
        x_value = .08 * (x_max - x_min) + x_min
    elif loc == 'tr':
        y_value = y_max - .9 * (y_max - y_min)
        x_value = .08 * (x_max - x_min) + x_min
    elif loc == 'bl':
        y_value = y_max - .1 * (y_max - y_min)
        x_value = x_max - .08 * (x_max - x_min)
    elif loc == 'tl':
        y_value = y_max - .9 * (y_max - y_min)
        x_value = x_max - .08 * (x_max - x_min)
    elif loc == 'tm':
        y_value = y_max - .9 * (y_max - y_min)
        x_value = x_min + (x_max - x_min) / 2
    elif loc == 'bm':
        y_value = y_max - .1 * (y_max - y_min)
        x_value = x_min + (x_max - x_min) / 2
    else:
        raise ValueError(
            'Unknown string format imported please look at code for acceptable positions')

    if string_add == '':

        # Turns to image number into a label
        if number < 26:
            axes.text(x_value, y_value, string.ascii_lowercase[number],
                      size=14, weight='bold', ha=text_pos,
                      va='center', color=formatting['color'],
                      path_effects=[patheffects.withStroke(linewidth=formatting['linewidth'],
                                                           foreground="k")])

        # allows for double letter index
        else:
            axes.text(x_value, y_value, string.ascii_lowercase[0] + string.ascii_lowercase[number - 26],
                      size=14, weight='bold', ha=text_pos,
                      va='center', color=formatting['color'],
                      path_effects=[patheffects.withStroke(linewidth=formatting['linewidth'],
                                                           foreground="k")])
    else:

        axes.text(x_value, y_value, string_add,
                  size=14, weight='bold', ha=text_pos,
                  va='center', color=formatting['color'],
                  path_effects=[patheffects.withStroke(linewidth=formatting['linewidth'],
                                                       foreground="k")])


def add_scalebar_to_figure(axes, image_size, scale_size, units='nm', loc='br'):
    """
    Adds scalebar to figures

    Parameters
    ----------
    axes : matplotlib axes
        axes which to add the plot to
    image_size : int
        size of the image in nm
    scale_size : str, optional
        size of the scalebar in units of nm
    units : str, optional
        sets the units for the label
    loc : str, optional
        sets the location of the label
    """

    x_lim, y_lim = axes.get_xlim(), axes.get_ylim()
    x_size, y_size = np.abs(np.floor(x_lim[1] - x_lim[0])), np.abs(np.floor(y_lim[1] - y_lim[0]))

    fract = scale_size / image_size

    x_point = np.linspace(x_lim[0], x_lim[1], np.floor(image_size))
    y_point = np.linspace(y_lim[0], y_lim[1], np.floor(image_size))

    if loc == 'br':
        x_start = x_point[np.int(.9 * image_size // 1)]
        x_end = x_point[np.int((.9 - fract) * image_size // 1)]
        y_start = y_point[np.int(.1 * image_size // 1)]
        y_end = y_point[np.int((.1 + .025) * image_size // 1)]
        y_label_height = y_point[np.int((.1 + .075) * image_size // 1)]
    elif loc == 'tr':
        x_start = x_point[np.int(.9 * image_size // 1)]
        x_end = x_point[np.int((.9 - fract) * image_size // 1)]
        y_start = y_point[np.int(.9 * image_size // 1)]
        y_end = y_point[np.int((.9 - .025) * image_size // 1)]
        y_label_height = y_point[np.int((.9 - .075) * image_size // 1)]

    path_maker(axes, [x_start, x_end, y_start, y_end], 'w', 'k', '-', 1)

    axes.text((x_start + x_end) / 2,
              y_label_height,
              '{0} {1}'.format(scale_size, units),
              size=14, weight='bold', ha='center',
              va='center', color='w',
              path_effects=[patheffects.withStroke(linewidth=1.5,
                                                   foreground="k")])


def path_maker(axes, locations, facecolor, edgecolor, linestyle, lineweight):
    """
    Adds path to figure

    Parameters
    ----------
    axes : matplotlib axes
        axes which to add the plot to
    locations : numpy array
        location to position the path
    facecolor : str, optional
        facecolor of the path
    edgecolor : str, optional
        edgecolor of the path
    linestyle : str, optional
        sets the style of the line, using conventional matplotlib styles
    lineweight : float, optional
        thickness of the line
    """
    vertices = []
    codes = []
    codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
    vertices = [(locations[0], locations[2]),
                (locations[1], locations[2]),
                (locations[1], locations[3]),
                (locations[0], locations[3]),
                (0, 0)]
    vertices = np.array(vertices, float)
    path = Path(vertices, codes)
    pathpatch = PathPatch(path, facecolor=facecolor, edgecolor=edgecolor,
                          ls=linestyle, lw=lineweight)
    axes.add_patch(pathpatch)


def savefig(filename, dpi=300, print_EPS=False, print_PNG=False):
    """
    Adds path to figure

    Parameters
    ----------
    filename : str
        path to save file
    dpi : int, optional
        resolution to save image
    print_EPS : bool, optional
        selects if export the EPS
    print_PNG : bool, optional
        selects if print the PNG
    """
    # Saves figures at EPS
    if print_EPS:
        plt.savefig(filename + '.eps', format='eps',
                    dpi=dpi, bbox_inches='tight')

    # Saves figures as PNG
    if print_PNG:
        plt.savefig(filename + '.png', format='png',
                    dpi=dpi, bbox_inches='tight')

def plot_pca_vectors(voltage, pca, num_of_plots=True, set_ylim=True, letter_labels=False,
                    filename='./PCA_vectors', print_EPS=False,
                    print_PNG=False, dpi=300):
    """
    Plots the PCA eigenvectors

    Parameters
    ----------
    voltage : numpy array
        voltage vector for the hysteresis loop
    pca : model
        previously computed pca
    num_of_plots : int, optional
        number of principle componets to show
    set_ylim : int, optional
        optional manual overide of y scaler
    letter_labels : bool, optional
        adds letter labels for use in publications
    filename : str, optional
        sets the path and filename for the exported images
    print_EPS : bool, optional
        to export as EPS
    print_PNG : bool, optional
        to export as PNG
    dpi : int, optional
        resolution of exported image
    """
    if num_of_plots:
        num_of_plots = pca.n_components_

    # creates the figures and axes in a pretty way
    fig, ax = layout_graphs_of_arb_number(num_of_plots)

    for i in range(num_of_plots):
        ax[i].plot(voltage, pca.components_[i], 'k')
        ax[i].set_xlabel('Voltage')
        ax[i].set_ylabel('Amplitude (Arb. U.)')
        ax[i].set_yticklabels('')
        #ax[i].set_title('PC {0}'.format(i+1))
        if not set_ylim:
            ax[i].set_ylim(set_ylim[0], set_ylim[1])

        # labels figures
        if letter_labels:
            labelfigs(ax[i], i)
        labelfigs(ax[i], i, string_add='PC {0}'.format(i + 1), loc='bm')

    plt.tight_layout(pad=0, h_pad=0)

    savefig(filename, dpi=300, print_EPS=print_EPS, print_PNG=print_PNG)

def plot_embedding_maps(data, add_colorbars=True, verbose=False, letter_labels=False,
                        add_scalebar=False, filename='./embedding_maps', print_EPS=False,
                        print_PNG=False, dpi=300, num_of_plots=True, ranges=None):
    """
    Adds a colorbar to a imageplot

    Parameters
    ----------
    data : raw data to plot of embeddings
        data of embeddings
    add_colorbars : bool, optional
        adds colorbars to images
    verbose : bool, optional
        sets the verbosity level
    letter_labels : bool, optional
        adds letter labels for use in publications
    add_scalebar : bool, optional
        sets whether a scalebar is added to the maps
    filename : str, optional
        sets the path and filename for the exported images
    print_EPS : bool, optional
        to export as EPS
    print_PNG : bool, optional
        to export as PNG
    dpi : int, optional
        resolution of exported image
    num_of_plots : int, optional
            number of principle componets to show
    ranges : float, optional
            sets the clim of the images
    """
    if num_of_plots:
        num_of_plots = data.shape[data.ndim - 1]

    # creates the figures and axes in a pretty way
    fig, ax = layout_graphs_of_arb_number(num_of_plots)
    # resizes the array for hyperspectral data

    if data.ndim == 3:
        original_size = data.shape[0].astype(int)
        data = data.reshape(-1, data.shape[2])
        verbose_print(verbose, 'shape of data resized to [{0} x {1}]'.format(
            data.shape[0], data.shape[1]))
    elif data.ndim == 2:
        original_size = np.sqrt(data.shape[0]).astype(int)
    else:
        raise ValueError("data is of an incorrect size")

    for i in range(num_of_plots):
        im = ax[i].imshow(data[:, i].reshape(original_size, original_size))
        ax[i].set_yticklabels('')
        ax[i].set_xticklabels('')

        if ranges is None:
            pass
        else:
            im.set_clim(0,ranges[i])

        if add_colorbars:
            add_colorbar(ax[i], im)

        # labels figures
        if letter_labels:
            labelfigs(ax[i], i)
        labelfigs(ax[i], i, string_add='emb. {0}'.format(i + 1), loc='bm')

        if add_scalebar is not False:
            add_scalebar_to_figure(ax[i], add_scalebar[0], add_scalebar[1])

    plt.tight_layout(pad=0, h_pad=0)

    savefig(filename, dpi=300, print_EPS=print_EPS, print_PNG=print_PNG)

    return(fig)

def custom_plt_format():

    """
    Defines custom plotting style
    """

    # Loads the custom style
    plt.style.use('./custom.mplstyle')

def plot_raw_BE_data(x, y, cycle, data, signals, folder_name, cmaps = 'inferno'):

    """
    Plots raw BE data
    TODO: fix the size to make it generalizable

    Parameters
    ----------
    data : raw data to plot
        Band Excitation Piezoresponse Data
    signals : list
        description of what to plot
    folder_name : string
        folder where to save
    cmaps : string, optional
        colormap to use for plot
    """
    # Makes folder
    folder = Make_folder('Raw_Loops_mixed')

    # Sets the colormap
    mymap = plt.get_cmap(cmaps)

    # Defines the axes positions
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    #fig.suptitle('Representitive Raw Loops', fontsize=16, y=1,ha = 'center')

    # Plots each of the graphs
    for i, (signal, values) in enumerate(signals.items()):

        # Formats the figures
        axes[i].set_xlabel('Voltage (V)')
        axes[i].set_aspect('auto')
        axes[i].set_xticks(np.linspace(-15, 15, 7))
        axes[i].set_ylabel('{0} {1}'.format(signal, values['units']))
        # axes[i].set_title('{0}'.format(signal))
        axes[i].yaxis.set_major_formatter(
            ticker.FormatStrFormatter('{}'.format(values['format_str'])))

        # Sets standard y-scales
        if np.isfinite(values['y_lim']).any():
            axes[i].set_ylim(values['y_lim'])
            axes[i].set_yticks(values['y_tick'])

        # Constructs the name and removes infinite values
        field = 'Out{0}{1}_mixed'.format(values['symbol'], cycle)

        # Stores values for graphing
        if signal in ['Amplitude', 'Resonance', 'Quality Factor']:

            finite_values = np.isfinite(data[field][x, y, :])
            signal_plot = data[field][x, y, finite_values]
            voltage_plot = data['VoltageDC_mixed'][finite_values, 1]

        elif signal == 'Phase':

            # Normalizes the phase data
            finite_values = np.isfinite(data[field][x, y, :])
            phi_data = data[field][x, y, finite_values]
            signal_plot = phi_data - \
                np.min(phi_data) - (np.max(phi_data) - np.min(phi_data)) / 2

        elif signal == 'Piezoresponse':

            # Computes and shapes the matrix for the piezorespons
            voltage_plot = np.concatenate([data['Voltagedata_mixed'][np.int(96 / 4 * 3)::],
                                           data['Voltagedata_mixed'][0:np.int(96 / 4 * 3)]])
            piezoresponse_data = data['Loopdata_mixed'][x, y, :]
            signal_plot = piezoresponse_data - np.mean(piezoresponse_data)
            signal_plot = np.concatenate([signal_plot[np.int(96 / 4 * 3)::],
                                          signal_plot[0:np.int(96 / 4 * 3)]])

        # plots the graph
        im = axes[i].plot(voltage_plot, signal_plot, '-ok', markersize=3)

        # Removes Y-label for Arb. U.
        if signal in ['Amplitude', 'Piezoresponse']:
            axes[i].set_yticklabels('')

        # Adds markers to plot
        for index in [0, 14, 24, 59, 72]:
            axes[i].plot(voltage_plot[index], signal_plot[index],
                         'ok', markersize=8,
                         color=mymap((data['VoltageDC_mixed'][index, 1] + 16) / 32))

        # labels figures
        labelfigs(axes[i], i)

        plt.tight_layout()

        # Prints the images
        filename = 'Raw_Loops_{0:d}_{1:d}'.format(x, y)
        savefig(pjoin(folder, filename))

def Make_folder(folder_name, **kwargs):
    """
    Function that makes new folders

    Parameters
    ----------

    folder_name : string
        folder where to save

    """

    # Makes folder
    folder = pjoin('./', folder_name)
    os.makedirs(folder, exist_ok=True)

    return (folder)

def cluster_loops(data, int_cluster, num_c_cluster, num_a_cluster, seed=[]):
    """
    Clusters the loops

    Parameters
    ----------
    data : float
        data for clustering
    int_cluster : int
        first level divisive clustering
    c_cluster : int
        c level divissive clustering
    a_cluster : int
        a level divissive clustering
    seed : int, optional
        fixes the seed for replicating results

    """

    if seed!=[]:
        # Defines the random seed for consistant clustering
        np.random.seed(seed)

    # Scales the data for clustering
    scaled_data = pre.StandardScaler().fit_transform(data)

    # Kmeans clustering of the data into c and a domains
    cluster_ca = cluster.KMeans(
        n_clusters=2).fit_predict(scaled_data)

    # K-means clustering of a domains
    a_map = np.zeros(data.shape[0])
    a_cluster = cluster.KMeans(n_clusters=num_a_cluster).fit_predict(
        scaled_data[cluster_ca == 1])
    a_map[cluster_ca == 1] = a_cluster + 1

    # Kmeans clustering of c domains
    c_map = np.zeros(data.shape[0])
    c_cluster = cluster.KMeans(n_clusters=num_c_cluster).fit_predict(
        scaled_data[cluster_ca == 0])
    c_map[cluster_ca == 0] = c_cluster + num_a_cluster + 1

    # Enumerates the k-means clustering map for ploting
    combined_map = a_map + c_map

    return(combined_map, cluster_ca, c_map, a_map)


def plot_hierachical_cluster_maps(cluster_ca, c_map, a_map, names):

    # Defines the figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    # Loops around all the clusteres found
    for i, name in enumerate(names):

        (title, cluster_map) = name

        # sets the order of the plots
        if cluster_map == 'cluster_ca':
            i = 0
        elif cluster_map == 'a_map':
            i = 2
        elif cluster_map == 'c_map':
            i = 1

        size_image = np.sqrt(c_map.shape[0]).astype(int)-1

        num_colors = len(np.unique(eval(cluster_map)))
        scales = [np.max(eval(cluster_map))-(num_colors -.5), np.max(eval(cluster_map))+.5]

        # Formats the axes
        axes[i].set_xticklabels([])
        axes[i].set_yticklabels([])
        axes[i].set_xticks(np.linspace(0, size_image, 5))
        axes[i].set_yticks(np.linspace(0, size_image, 5))
        axes[i].set_title(title)
        axes[i].set_facecolor((.55, .55, .55))

        labelfigs(axes[i], i, loc='tr')
        add_scalebar_to_figure(axes[i], 2000, 500, loc='tr')

        # Plots the axes
        im = axes[i].imshow(eval(cluster_map).reshape(size_image+1, size_image+1),
                            cmap=eval(f'cmap_{num_colors}'), clim=scales)

        # Formats the colorbar
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes('right', size='10%', pad=0.05)
        cbar = plt.colorbar(im, cax=cax, format='%d')
        cbar.set_ticks([])

def get_ith_layer_output(model, X, i, mode='test'):
    ''' see https://keras.io/getting-started/faq/#keras-faq-frequently-asked-keras-questions'''
    get_ith_layer = keras.backend.function(
        [model.layers[0].input, keras.backend.learning_phase()], [model.layers[i].output])
    layer_output = get_ith_layer([X, 0 if mode=='test' else 1])[0]
    return layer_output


def export_images_of_embeddings(data, folder, start_ind=-1, scalebar_values = [2000,500], repeat=False):

    """
    Loads the model and exports embedding maps

    Parameters
    ----------
    data : float
        Piezoresponse loops
    folder : string
        location where the data is saved
    start_ind : int, optional
        sets the index to start
    scalebar_values : numpy array, optional
        image size and scalebar size in nm
    repeat : bool, optional
        fixes the seed for replicating results

    """

    data -= np.mean(data.reshape(-1))
    data /= np.std(data)
    data = np.atleast_3d(data)

    files = glob.glob(folder + '*.hdf5')
    images = glob.glob(folder + '*.png')

    Values = {}

    for i, file in enumerate(files):
        Values[i] = re.split(r'(weights.|-|.hdf5)',file)

    for i, value in enumerate(Values):

        print(i)

        if i > start_ind:

            if folder + f'Epoch_{Values[i][2]}_Loss_{Values[i][4]}.png' not in images or repeat == True:

                model = keras.models.load_model(files[i])

                # preallocates the vector
                out = np.zeros((3600, 16))

                out = get_ith_layer_output(model, np.atleast_3d(data), 3, mode='test').squeeze()

                fig = plot_embedding_maps(out, add_scalebar=scalebar_values, print_PNG=False)

                plt.suptitle(f'Epoch {Values[i][2]} Loss {Values[i][4]}', fontsize=16)

                savefig(folder + f'Epoch_{Values[i][2]}_Loss_{Values[i][4]}',print_PNG=True)

                plt.close(fig)

def make_movies_from_images(folder, name):

    """
    Makes movie from image sequency

    Parameters
    ----------
    folder : string
        location where the data is saved
    name  : string,
        name of file where to save

    """


    files = glob.glob(folder + '*.png')

    Values = {}
    inds = np.zeros((len(files)))

    for i, file in enumerate(files):
        Values[i] = re.split(r'(Epoch_|_Loss_|.png)',file)

        inds[i] = Values[i][2]

    ind_sort = np.argsort(inds)
    files = np.asarray(files)
    files = files[ind_sort]

    files = np.ndarray.tolist(files)

    clip = mpy.video.io.ImageSequenceClip.ImageSequenceClip(files, fps=4)
    clip.write_videofile(name + '.mp4', fps=4)

def plot_clustered_hysteresis(voltage,
                                Piezoresponse,
                                combined_cluster_map,
                                loop_ylim = [-1.5e-4, 1.5e-4],
                                loop_xlim = np.linspace(-15, 15, 7)):

    """
    Plots the cluster maps and the average hysteresis loops

    Parameters
    ----------
    voltage : numpy array, float
        voltage vector
    Piezoresponse  : numpy array, float
        Piezoresponse data from the loops
    combined_cluster_map  : numpy array, int
        computed cluster map of the data
    loop_ylim: numpy array, float
        sets the y limit for the hysteresis loops
    loops_xlim: numpy array, float
        sets the x limit for the hysteresis loops

    """

    # organization of the raw data

    num_pix = np.sqrt(combined_cluster_map.shape[0]).astype(int)
    num_clusters = len(np.unique(combined_cluster_map))

    # Preallocates some matrix
    clustered_maps= np.zeros((num_clusters, num_pix, num_pix))
    clustered_ave_piezoreponse = np.zeros((num_clusters, Piezoresponse.shape[1]))

    cmap_=eval(f'cmap_{num_clusters}')

    # Loops around the clusters found
    for i in range(num_clusters):

        # Stores the binary maps
        binary = (combined_cluster_map == i + 1)
        clustered_maps[i, :, :] = binary.reshape(num_pix, num_pix)

        # Stores the average piezoelectric loops
        clustered_ave_piezoreponse[i] = np.mean(
            Piezoresponse[binary], axis=0)

    fig, ax = layout_graphs_of_arb_number(num_clusters + 1)

    for i in range(num_clusters + 1):

        if i == 0:

            scales = [np.max(combined_cluster_map)-(num_clusters -.5), np.max(combined_cluster_map)+.5]

            # Formats the axes
            ax[i].set_xticklabels([])
            ax[i].set_yticklabels([])
            ax[i].set_xticks(np.linspace(0, num_pix, 5))
            ax[i].set_yticks(np.linspace(0, num_pix, 5))
            ax[i].set_facecolor((.55, .55, .55))

            labelfigs(ax[i], i, loc='tr')
            add_scalebar_to_figure(ax[i], 2000, 500, loc='tr')

            # Plots the axes
            im = ax[i].imshow(combined_cluster_map.reshape(num_pix, num_pix),
                                cmap=cmap_, clim=scales)

            add_colorbar(ax[i], im, ticks=False)

        else:

            # Plots the graphs
            hys_loop = ax[i].plot(voltage, clustered_ave_piezoreponse[i-1],cmap_.colors[i-1])

            # formats the axes
            ax[i].yaxis.tick_left()
            ax[i].set_xticks(loop_xlim)
            ax[i].yaxis.set_label_position('left')
            ax[i].set_ylabel('Piezoresponse (Arb.U.)')
            ax[i].set_xlabel('Voltage (V)')
            ax[i].yaxis.get_major_formatter().set_powerlimits((0, 1))
            ax[i].set_ylim(loop_ylim)
            ax[i].set_yticklabels([])

            pos = ax[i].get_position()

            # Posititions the binary image
            axes_in = plt.axes([pos.x0+.06,
                                pos.y0+.025,
                                .1, .1])

            # Plots and formats the binary image
            axes_in.imshow(clustered_maps[i-1, :, :],
                           cmap=cmap_2)
            axes_in.tick_params(axis='both', labelbottom='off', labelleft='off')

            labelfigs(ax[i], i, loc='br')

            set_axis_aspect(ax[i])


def get_ith_layer_output(model, X, i, mode='test'):

    """
    Computes the activations of a specific layer
    see https://keras.io/getting-started/faq/#keras-faq-frequently-asked-keras-questions'


    Parameters
    ----------
    model : keras model, object
        pre-trained keras model
    X  : numpy array, float
        Input data
    i  : numpy, int
        index of the layer to extract
    mode : string, optional
        test or train, changes the model behavior to scale the network properly when using
        dropout or batchnorm
    """
    get_ith_layer = keras.backend.function(
        [model.layers[0].input, keras.backend.learning_phase()], [model.layers[i].output])
    layer_output = get_ith_layer([X, 0 if mode=='test' else 1])[0]
    return layer_output


def get_activations(model, X=[], i=[], mode='test'):

    """
    support function to get the activations of a specific layer
    this function can take either a model and compute the activations or can load previously
    generated activations saved as an numpy array

    Parameters
    ----------
    model : keras model, object
        pre-trained keras model
    X  : numpy array, float
        Input data
    i  : numpy, int
        index of the layer to extract
    mode : string, optional
        test or train, changes the model behavior to scale the network properly when using
        dropout or batchnorm
    """

    if isinstance(model, str):
        activation = np.load(model)
        print(f'model {model} loaded from saved file')
    else:
        activation = get_ith_layer_output(model, np.atleast_3d(X), i, model)

    return activation

def rnn_auto(layer, size, num_encode_layers, num_decode_layers, embedding, n_step, lr = 3e-5, drop_frac=0.,bidirectional=True, l1_norm = 1e-4,**kwargs):
    """
    Function which builds the reccurrent neural network autoencoder

    Parameters
    ----------
    layer : string; options: 'lstm','gru'
        selects the layer type
    size  : numpy, int
        sets the size of encoding and decoding layers in the network
    num_encode_layers  : numpy, int
        sets the number of encoding layers in the network
    num_decode_layers : numpy, int
        sets the number of decoding layers in the network
    embedding : numpy, int
        sets the size of the embedding layer
    n_steps : numpy, int
        length of the input time series
    lr : numpy, float
        sets the learning rate for the model
    drop_frac : numpy, float
        sets the dropout fraction
    bidirectional : numpy, bool
        selects if the model is linear or bidirectional
    l1_norm : numpy. float
        sets the lambda value of the l1 normalization. The larger the value the greater the
        sparsity. None can be passed to exclude the use or l1 normailzation.

    Returns
    -------
    model : Keras, object
        Keras tensorflow model
    """

    # defines the model
    model = Sequential()

    # selects if the model is bidirectional
    if bidirectional:
        wrapper = Bidirectional
        # builds the first layer
        model.add(Bidirectional(layer(size, return_sequences=(num_encode_layers > 1),  dropout=drop_frac),
                            input_shape=(n_step, 1)))
    else:
        wrapper = lambda x: x
        # builds the first layer
        model.add(wrapper(layer(size, return_sequences=(num_encode_layers > 1),  dropout=drop_frac,
                input_shape=(n_step, 1))))

    # builds the encoding layers
    for i in range(1, num_encode_layers):
        model.add(wrapper(layer(size, return_sequences=(i < num_encode_layers - 1), dropout=drop_frac)))

    # builds the embedding layer
    if l1_norm == None:
        # embedding layer without l1 regulariization
        model.add(Dense(embedding, activation='relu', name='encoding'))
    else:
        # embedding layer with l1 regularization
        model.add(Dense(embedding, activation='relu', name='encoding',activity_regularizer=l1(l1_norm)))

    # builds the repeat vector
    model.add(RepeatVector(n_step))

    # builds the decoding layer
    for i in range(num_decode_layers):
        model.add(wrapper(layer(size, return_sequences=True, dropout=drop_frac)))

    # builds the time distributed layer to reconstruct the original input
    model.add(TimeDistributed(Dense(1, activation='linear')))

    # complies the model
    model.compile(Adam(lr), loss='mse')

    # returns the model
    return model

def train_model(model, data_train, data_test, path, epochs, batch_size):
    """
    Function which trains the neural network

    Parameters
    ----------
    model : Keras, object
        model to train
    data_train  : numpy, float
        data to train the network
    data_test  : numpy, float
        data to test the network
    path : string
        sets the folder to save the data
    epochs : numpy, int
        train the network for this number of epochs
    batch_size : numpy, int
        sets the size of the batch. Batch size should be as large as possible. The batch
        size is limited by the GPU memory.
    """

    make_folder(path)
    keras.models.save_model(model,path + '/start')

    tbCallBack = keras.callbacks.TensorBoard(
                        log_dir= path, histogram_freq=0,
                        write_graph=True, write_images=True)

    #builds the filename
    filepath = path + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5'
    # sets the control of checkpoints
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True,
                                             save_weights_only=True, mode='min', period=1)

    # fits the model
    model.fit(np.atleast_3d(data), np.atleast_3d(data), epochs=250000,
          batch_size=1200, validation_data=(np.atleast_3d(data), np.atleast_3d(data)),
          callbacks=[tbCallBack, checkpoint])


def clean_and_interpolate(data, fit_type='spline'):
    """
    Function which removes bad datapoints

    Parameters
    ----------
    data : numpy, float
        data to clean
    """

    data[~np.isfinite(data)] = np.nan
    data = interpolate_missing_points(data, fit_type)
    data = data.reshape(-1, data.shape[2])
    return data

def plot_imagemap(axis, data, clim = None, color_bar = False, custom_colormap = None):

        """
        Plots an imagemap

        Parameters
        ----------
        axis : matplotlib, object
            axis which is plotted
        data  : numpy, float
            data to plot
        clim  : numpy, float, optional
            sets the climit for the image
        color_bar  : bool, optional
            selects to plot the colorbar bar for the image
        """
        if data.ndim == 1:
            data = data.reshape(np.sqrt(data.shape[0]).astype(int), np.sqrt(data.shape[0]).astype(int))

        if custom_colormap is None:
            cmap = plt.get_cmap('viridis')
        else:
            cmap = custom_colormap

        if clim is None:
            im = axis.imshow(data,  cmap=cmap)
        else:
            im = axis.imshow(data, clim = clim, cmap=cmap)

        axis.set_yticklabels('')
        axis.set_xticklabels('')

        if color_bar:
            # Adds the colorbar
            divider = make_axes_locatable(axis)
            cax = divider.append_axes('right', size='10%', pad=0.05)
            cbar = plt.colorbar(im, cax=cax, format='%.1e')




def normalize_data(data):

    data_norm = np.copy(data)
    data_norm -= np.mean(data_norm.reshape(-1))
    data_norm /= np.std(data_norm)

    return data_norm

def get_run_id(layer_type, size, num_encode_layers,
             num_decode_layers, embedding,
             lr, drop_frac,
             bidirectional, l1_norm,
             batch_norm, **kwargs):

    run = (f"{layer_type}_size{size:03d}_enc{num_encode_layers}_emb{embedding}_dec{num_decode_layers}_lr{lr:1.0e}"
           f"_drop{int(100 * drop_frac)}").replace('e-', 'm')
    if Bidirectional:
        run = 'Bidirect_' + run
    if layer_type == 'conv':
        run += f'_k{kernel_size}'
    if np.any(batch_norm):

        if batch_norm[0]:
            ind = 'T'
        else:
            ind = 'F'

        if batch_norm[1]:
            ind1 = 'T'
        else:
            ind1 = 'F'

        run += f'_batchnorm_{ind}{ind1}'
    return run

def add_dropout(model, value):
    if value > 0:
        return model.add(Dropout(value))
    else:
        pass

def rnn_auto(layer_type, size, num_encode_layers,
             num_decode_layers, embedding,
             n_step, lr = 3e-5, drop_frac=0.,
             bidirectional=True, l1_norm = 1e-4,
             batch_norm=[False, False],**kwargs):
    """
    Function which builds the reccurrent neural network autoencoder

    Parameters
    ----------
    layer : string; options: 'lstm','gru'
        selects the layer type
    size  : numpy, int
        sets the size of encoding and decoding layers in the network
    num_encode_layers  : numpy, int
        sets the number of encoding layers in the network
    num_decode_layers : numpy, int
        sets the number of decoding layers in the network
    embedding : numpy, int
        sets the size of the embedding layer
    n_steps : numpy, int
        length of the input time series
    lr : numpy, float
        sets the learning rate for the model
    drop_frac : numpy, float
        sets the dropout fraction
    bidirectional : numpy, bool
        selects if the model is linear or bidirectional
    l1_norm : numpy. float
        sets the lambda value of the l1 normalization. The larger the value the greater the
        sparsity. None can be passed to exclude the use or l1 normailzation.

    Returns
    -------
    model : Keras, object
        Keras tensorflow model
    """

    # Selects the type of RNN neurons to use
    if layer_type == 'lstm':
        layer = LSTM
    elif layer_type == 'gru':
        layer = GRU

    # defines the model
    model = Sequential()

    # selects if the model is bidirectional
    if bidirectional:
        wrapper = Bidirectional
        # builds the first layer

        # builds the first layer
        model.add(wrapper(layer(size, return_sequences=(num_encode_layers > 1)),
                        input_shape=(n_step, 1)))
        add_dropout(model, drop_frac)
    else:
        wrapper = lambda x: x
        # builds the first layer
        model.add(wrapper(layer(size, return_sequences=(num_encode_layers > 1),
                        input_shape=(n_step, 1))))
        add_dropout(model, drop_frac)


    # builds the encoding layers
    for i in range(1, num_encode_layers):
        model.add(wrapper(layer(size, return_sequences=(i < num_encode_layers - 1))))
        add_dropout(model, drop_frac)

    # adds batch normalization prior to embedding layer
    if batch_norm[0]:
        model.add(BatchNormalization())

    # builds the embedding layer
    if l1_norm == None:
        # embedding layer without l1 regulariization
        model.add(Dense(embedding, activation='relu', name='encoding'))
    else:
        # embedding layer with l1 regularization
        model.add(Dense(embedding, activation='relu', name='encoding',activity_regularizer=l1(l1_norm)))

    # adds batch normalization after embedding layer
    if batch_norm[1]:
        model.add(BatchNormalization())

    # builds the repeat vector
    model.add(RepeatVector(n_step))

    # builds the decoding layer
    for i in range(num_decode_layers):
        model.add(wrapper(layer(size, return_sequences=True)))
        add_dropout(model, drop_frac)

    # builds the time distributed layer to reconstruct the original input
    model.add(TimeDistributed(Dense(1, activation='linear')))

    # complies the model
    model.compile(Adam(lr), loss='mse')

    run_id = get_run_id(layer_type, size, num_encode_layers,
             num_decode_layers, embedding,
             lr, drop_frac, bidirectional, l1_norm,
             batch_norm)

    # returns the model
    return model, run_id

def check_folder_exist(folder_name):
    value = 1
    folder_name += f'_{value:03d}'

    if os.path.exists("./" + folder_name):
        while os.path.exists("./" + folder_name):
            value += 1
            folder_name = folder_name[:-4]
            folder_name += f'_{value:03d}'
    return folder_name

def prints_all_BE_images(data, signal_clim, scalebar=None):

    """
    Function which prints all of the BE images

    Parameters
    ----------
    data : numpy, float
        raw data to plot
    signal_clim  : dictonary
        Instructions for extracting the data and plotting the data
    scalebar  : numpy, int
        sets if a scalebar is added to the image. First index is the size of the image, second is
        the size of the scalebar.
    """

    # Graphs and prints all figures
    for (signal_name, signal), colorscale in signal_clim.items():

        # Makes the folder
        folder = pjoin('./', '{}_mixed'.format(signal_name))
        os.makedirs(folder, exist_ok=True)

        # Cycles around each loop
        for cycle in (1, 2):

            # Bulids data name
            field = 'Out{0}{1}_mixed'.format(signal, cycle)

            # Displays loop status
            print('{0} {1}'.format(signal_name, cycle))

            # Loops around each voltage step
            for i in range(data[field].shape[2]):

                # Defines the figure and axes
                fig, ax1 = plt.subplots(figsize=(3, 3))

                # Plots the data
                im = ax1.imshow((data[field][:, :, i]), cmap='viridis')

                # Formats the figure
                ax1.set_xticklabels([])
                ax1.set_yticklabels([])
                ax1.set_xticks(np.linspace(0, data[field].shape[0], 5))
                ax1.set_yticks(np.linspace(0, data[field].shape[0], 5))
                im.set_clim(colorscale)
                ax1.set_facecolor((.55, .55, .55))

                # adds the scalebar to the images
                if scalebar is not None:
                    add_scalebar_to_figure(ax1, scalebar[0], scalebar[1])

                # Generates the filename
                filename = '{0}{1}_{2:03d}'.format(signal, cycle, i)

                # Saves the figure
                fig.savefig(pjoin(folder, filename + '.png'), format='png',
                            dpi=300)

                # Closes the figure
                plt.close(fig)


def make_imageseries_for_movie(data, signal_clim,  scalebar=None, cycles=(1,2), folder_name = ''):

    """
    Function which prints all of the BE images for movie

    Parameters
    ----------
    data : numpy, float
        raw data to plot
    signal_clim  : dictonary
        Instructions for extracting the data and plotting the data
    scalebar  : numpy, int
        sets if a scalebar is added to the image. First index is the size of the image, second is
        the size of the scalebar.
    cycles  : tuple
        cycles to included in the movie
    folder_name  : string
        sets the name of the folder.
    """

    # Makes Folders to store images
    folder = pjoin('.', 'Movie_Images'+folder_name)
    os.makedirs(folder, exist_ok=True)

    # Cycles around each loop
    for cycle in cycles:

        # Loops around each voltage step
        for i in range(96):

            # Defines the axes positions
            fig = plt.figure(figsize=(8, 12))
            ax1 = plt.subplot2grid((3, 2), (0, 0))
            ax2 = plt.subplot2grid((3, 2), (0, 1))
            ax3 = plt.subplot2grid((3, 2), (1, 0))
            ax4 = plt.subplot2grid((3, 2), (1, 1))
            ax5 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
            axes = (ax1, ax2, ax3, ax4)

            # Sets the axes labels and scales
            for ax in axes:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks(np.linspace(0, 1024, 5))
                ax.set_yticks(np.linspace(0, 1024, 5))
                ax.set_facecolor((.55, .55, .55))

            # Plots the response maps
            for j, (signals, colorscale) in enumerate(signal_clim.items()):

                (signal_name, signal, formspec) = signals

                # Defines the data location
                field = 'Out{0}{1}_mixed'.format(signal, cycle)

                # Plots and formats the images
                im = axes[j].imshow(data[field][:, :, i])
                axes[j].set_title(signal_name)
                im.set_clim(colorscale)
                # adds the scalebar to the images
                if scalebar is not None:
                    add_scalebar_to_figure(ax1, scalebar[0], scalebar[1])

                # adds a colorbar
                add_colorbar(axes[j], im, ticks=False)

                if signal_name == 'Quality Factor':
                    add_scalebar_to_figure(axes[j], 2000, 500)

            # Plots the voltage graph
            im5 = ax5.plot(np.reshape(
                data['VoltageDC_mixed'], 96 * 2, 1), 'ok')
            ax5.plot(i + (cycle - 1) * 96,
                     data['VoltageDC_mixed'][i, 0],
                     'rs', markersize=12)
            ax5.set_xlabel('Time Steps')
            ax5.set_ylabel('Voltage (V)')

            # Generates the filename
            filename = 'M{0}_{1:03d}'.format(cycle, i)

            # Saves the figure
            fig.savefig(pjoin(folder, filename + '.png'), format='png',
                        dpi=300)

            # Closes the figure
            plt.close(fig)

def make_movie(movie_name, folder, file_format, fps, output_format = 'mp4', reverse = False):

    """
    Function which makes movies from an imageseries

    Parameters
    ----------
    movie_name : string
        name of the movie
    folder  : string
        folder where the image series is located
    file_format  : string
        sets the format of the files to import
    fps  : numpy, int
        frames per second
    output_format  : string
        sets the format for the output file
        supported types .mp4 and gif
        animated gif create large files
    """

    # searches the folder and finds the files
    file_list = glob.glob('./' + folder + '/*.' + file_format)

    # Sorts the files by number makes 2 lists to go forward and back
    list.sort(file_list)
    file_list_rev = glob.glob('./' + folder + '/*.' + file_format)
    list.sort(file_list_rev,reverse=True)

    if reverse:
        new_list = file_list + file_list_rev
    else:
        new_list = file_list


    if output_format == 'gif':
        # makes an animated gif from the images
        clip = mpy.editor.ImageSequenceClip(new_list, fps=fps)
        clip.write_gif('{}.gif'.format(movie_name), fps=fps)
    else:
        clip = mpy.video.io.ImageSequenceClip.ImageSequenceClip(new_list, fps=fps)
        clip.write_videofile('{}.mp4'.format(movie_name), fps=fps)


def explore_raw_BE_data(data, x, y, cycle, signal_clim, folder_name,
                                            print_PNG=False, print_EPS=False):

    # Makes folder
    folder = Make_folder(folder_name)

    # Sets the colormap
    mymap = plt.get_cmap('inferno')

    # Defines the axes positions
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))

    v_length = data['Voltagedata_mixed'].shape[0]

    # Plots each of the graphs
    for i, (signal, values) in enumerate(signal_clim.items()):

        # Formats the figures
        axes[i].set_xlabel('Voltage (V)')
        axes[i].set_aspect('auto')
        axes[i].set_xticks(np.linspace(-15, 15, 7))
        axes[i].set_ylabel('{0} {1}'.format(signal, values['units']))
        # axes[i].set_title('{0}'.format(signal))
        axes[i].yaxis.set_major_formatter(
            ticker.FormatStrFormatter('{}'.format(values['format_str'])))

        # Sets standard y-scales
        if np.isfinite(values['y_lim']).any():
            axes[i].set_ylim(values['y_lim'])
            axes[i].set_yticks(values['y_tick'])

        # Constructs the name and removes infinite values
        field = 'Out{0}{1}_mixed'.format(values['symbol'], cycle)

        # Stores values for graphing
        if signal in ['Amplitude', 'Resonance', 'Quality Factor']:

            finite_values = np.isfinite(data[field][x, y, :])
            signal_plot = data[field][x, y, finite_values]
            voltage_plot = data['VoltageDC_mixed'][finite_values, 1]

        elif signal == 'Phase':

            # Normalizes the phase data
            finite_values = np.isfinite(data[field][x, y, :])
            phi_data = data[field][x, y, finite_values]
            signal_plot = phi_data - \
                np.min(phi_data) - (np.max(phi_data) - np.min(phi_data)) / 2

        elif signal == 'Piezoresponse':

            # Computes and shapes the matrix for the piezorespons
            voltage_plot = np.concatenate([data['Voltagedata_mixed'][np.int(v_length / 4 * 3)::],
                                           data['Voltagedata_mixed'][0:np.int(v_length / 4 * 3)]])
            piezoresponse_data = data['Loopdata_mixed'][x, y, :]
            signal_plot = piezoresponse_data - np.mean(piezoresponse_data)
            signal_plot = np.concatenate([signal_plot[np.int(v_length / 4 * 3)::],
                                          signal_plot[0:np.int(v_length / 4 * 3)]])

        # plots the graph
        im = axes[i].plot(voltage_plot, signal_plot, '-ok', markersize=3)

        # Removes Y-label for Arb. U.
        if signal in ['Amplitude', 'Piezoresponse']:
            axes[i].set_yticklabels('')

        # Adds markers to plot
        for index in [0, 14, 24, 59, 72]:
            axes[i].plot(voltage_plot[index], signal_plot[index],
                         'ok', markersize=8,
                         color=mymap((data['VoltageDC_mixed'][index, 1] + 16) / 32))

        # labels figures
        labelfigs(axes[i], i)

        plt.tight_layout()

        # Prints the images
        filename = 'Raw_Loops_{0:d}_{1:d}'.format(x, y)
        savefig(pjoin(folder, filename), print_PNG = print_PNG, print_EPS=print_EPS)

def plot_fitting_results(data, signal_clim, print_PNG=False, print_EPS=False):
    # Defines the figure and axes
    fig, axes = plt.subplots(5, 6, figsize=(18, 15))
    axes = axes.reshape(30)

    # Plots each of the graphs
    for i, (signal, values) in enumerate(signal_clim.items()):

        # Sets the axes
        axes[i].set_xticklabels([])
        axes[i].set_yticklabels([])
        axes[i].set_xticks(np.linspace(0, 59, 5))
        axes[i].set_yticks(np.linspace(0, 59, 5))
        # axes[i].set_title('{0}'.format(values['label']))
        axes[i].set_facecolor((.55, .55, .55))

        # labels figures
        labelfigs(axes[i], i, loc='tr')
        labelfigs(axes[i], i, loc='tl', string_add=values['label'],
                  size=9, text_pos='right')

        field = '{}'.format(values['data_loc'])

        # Plots the graphs either abs of values or normal
        if i in {13, 20, 21, 22, 23}:
            im = axes[i].imshow(np.abs(data[field]),
                                cmap='viridis', clim=values['c_lim'])
        else:
            im = axes[i].imshow(data[field], cmap='viridis', clim=values['c_lim'])

        # Sets the colorbar
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes('right', size='10%', pad=0.05)
        cbar = plt.colorbar(im, cax=cax, format=values['format_str'])

        if signal not in ['Raw Voltage Centroid', 'Fitted Voltage Centroid', 'Loop Width',
                          'Left Coercive field', 'Right Coercive field', 'Negative Nucleation Bias',
                          'Positive Nucleation Bias', 'Optimum Rotation Angle', 'Normalized Amplitude Centroid',
                          'Normalized Voltage Centroid']:
            cbar.ax.set_yticklabels('')
            cbar.ax.set_yticks('')

        add_scalebar_to_figure(axes[i], 2000, 500)

    # Deletes unused figures
    fig.delaxes(axes[28])
    fig.delaxes(axes[29])

    # Saves Figure
    plt.tight_layout(pad=0, h_pad=-20)
    savefig('LoopFit_Results_mixed', print_PNG = print_PNG, print_EPS=print_EPS)

def set_axis_aspect(ax,ratio = 1):

        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)

def plot_NMF_maps(voltage, W, H, add_colorbars=True, verbose=False, letter_labels=False,
                  add_scalebar=False, filename='./PCA_maps', print_EPS=False,
                  print_PNG=False, dpi=300, custom_order = None):

    # extracts the number of maps
    num_of_plots = H.shape[1]

    image_size = np.sqrt(H.shape[0]).astype(int)

    # creates the figures and axes in a pretty way
    fig, ax = layout_graphs_of_arb_number(num_of_plots*2, mod = num_of_plots)

    min_ = np.min(W[:,:].reshape(-1))
    max_ = np.max(W[:,:].reshape(-1))

    if custom_order is not None:
        order=custom_order

    for i, ax in enumerate(ax):


        # Checks if axes is an image or a plot
        if (i // num_of_plots % 2 == 0):
            # converts axis number to index number
            k = i - ((i // num_of_plots) // 2) * num_of_plots

            plot_imagemap(ax, H[:, order[i]].reshape(image_size,image_size),
                             color_bar = add_colorbars)

            if add_scalebar is not False:
                add_scalebar_to_figure(ax, 2000, 500) # todo fix this for dependancies

            # labels figures
            if letter_labels:
                if (i // num_of_plots % 2 == 0):
                    labelfigs(ax, k)
        else:
            # converts axis number to index number
            k = i - num_of_plots - ((i // num_of_plots) // 2) * num_of_plots

            ax.plot(voltage, W[:,order[k]],'k')

            ax.set_xlabel('Voltage')
            ax.set_ylim([min_,max_])
            ax.set_ylabel('Piezoresponse (Arb. U.)')
            ax.set_yticklabels('')

            set_axis_aspect(ax)


    plt.tight_layout(pad=0, h_pad=-10)

    savefig(filename, dpi=300, print_EPS=print_EPS, print_PNG=print_PNG)


def find_dense_layer(model):

    model_summary = model.get_config()

    for i in range(len(model_summary)):
        if 'Dense' in model_summary[i]['class_name']:
            break

    return i


def remove_max_min_outliers(data, range_):

    if np.ndim(data) == 3:
        for i, j in np.ndindex(data.shape[:-1]):
            # finds the values less than the minimum
            low = data[i,j] < np.min(range_)
            # finds the values greater than the maximum
            high =data[i,j] > np.max(range_)

            # finds the index of the outlier values
            outliers = np.where(low + high)
            # sets the outliers equal to nan
            data[i,j,outliers] = np.nan
    else:
        for i in np.ndindex(data[:-1].shape):

            # finds the values less than the minimum
            low = data[i] < np.min(range_)
            # finds the values greater than the maximum
            high =data[i] > np.max(range_)

            # finds the index of the outlier values
            outliers = np.where(low + high)
            # sets the outliers equal to nan
            data[i,outliers] = np.nan

    return data
