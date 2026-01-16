"""
To plot HEALPix maps easily.
Assuming **RING** order**
"""

import healpy as hp
import numpy as np
from matplotlib import pyplot as plt


def create_subplots(n=1, ncols=1, figsize=(12, 8), **kwargs):
    """
    Creates a grid of subplots with n active axes and some empty space.

    Args:
        n (int): The number of subplots to create.
        ncols (int): The number of columns in the subplot grid. Defaults to 1.
        figsize (tuple, optional): The figure size. Defaults to (12, 8).

    Returns:
        tuple: A tuple containing the matplotlib figure and a flattened
               list of the subplot axes.
    """
    # Calculate the number of rows needed
    if n > ncols:
        nrows = np.ceil(n / ncols).astype(int)
    else:
        nrows = 1

    # Create the figure and subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)

    if n == 1:
        return fig, axes

    # Flatten the axes array for easier iteration, regardless of shape
    axes = axes.flatten()

    # Turn off the visibility of unused axes
    for i in range(n, len(axes)):
        axes[i].axis("off")

    return fig, axes[:n]


def plot_hp_grid(
    data,
    title=None,
    hpfunc=hp.mollview,
    hpfunc_kw=None,
    ncol=1,
    figsize=(12, 8),
    subplots_kw=None,
    return_pmap=False,
    graticule=True,
    graticule_kw=None,
):
    """
    Plot a grid of Healpy maps with flexible row/column arrangements.

    Parameters
    ----------
    data : list or array
        A list/array of maps to plot.
    title : list or array, optional
        Titles for each subplot. Must have the same length as data if provided.
    hpfunc : function, optional
        Healpy plotting function to use (default: hp.mollview).
    hpfunc_kw : dict or list of dicts, optional
        Extra keyword arguments for the Healpy plotting function. If a list is
        passed, each dictionary is passed to the corresponding subplot.
    ncol : int
        Number of columns in the grid. Default: 1.
    figsize : tuple, optional
        Figure size in inches (default: (12, 8)).
    subplots_kw : dict, optional
        Extra keyword arguments for plt.subplots().
    return_pmap : bool, optional
        Whether to return the projected map. Default: False.
    graticule : bool, optional
        Whether to plot graticule lines. Default: True.
    graticule_kw : dict, optional
        Extra keyword arguments for hp.graticule().
    """
    if isinstance(data, np.ma.MaskedArray) or isinstance(data, np.ndarray):
        data = [data]
    n = len(data)
    if subplots_kw is None:
        subplots_kw = {}

    if hpfunc_kw is None:
        hpfunc_kw_list = [{}] * n
    elif isinstance(hpfunc_kw, dict):
        hpfunc_kw_list = [hpfunc_kw] * n
    elif isinstance(hpfunc_kw, list):
        if len(hpfunc_kw) != n:
            raise ValueError("Length of hpfunc_kw must match length of data")
        hpfunc_kw_list = hpfunc_kw
    else:
        raise TypeError("hpfunc_kw must be a dict or a list of dicts")

    if title is None:
        title = [None] * n
    elif isinstance(title, str) and n > 1:
        title = [title + str(i) for i in range(n)]
    elif len(title) != n:
        raise ValueError("Length of title must match length of data")

    kw_graticule = dict(dpar=10, dmer=10)
    if graticule_kw is not None:
        kw_graticule.update(graticule_kw)

    # Create figure and axes
    fig, axs = create_subplots(n, ncols=ncol, figsize=figsize, **subplots_kw)

    if n == 1:
        plt.sca(axs)  # type: ignore # set current axis for healpy plotting
        sub_title = title[0]
        if return_pmap:
            pm = hpfunc(
                data[0],
                hold=True,
                title=sub_title,
                return_projected_map=True,
                **hpfunc_kw_list[0],
            )
            if graticule:
                hp.graticule(**kw_graticule)
            return axs, pm
        else:
            hpfunc(data[0], hold=True, title=sub_title, **hpfunc_kw_list[0])
            if graticule:
                hp.graticule(**kw_graticule)
            return axs

    else:
        # Loop through data and plot
        pm_list = []
        for idx, ax in enumerate(axs):
            plt.sca(ax)  # set current axis for healpy plotting
            sub_title = title[idx]
            if return_pmap:
                pm = hpfunc(
                    data[idx],
                    hold=True,
                    title=sub_title,
                    return_projected_map=True,
                    **hpfunc_kw_list[idx],
                )
                pm_list.append(pm)
                if graticule:
                    hp.graticule(**kw_graticule)
            else:
                hpfunc(data[idx], hold=True, title=sub_title, **hpfunc_kw_list[idx])
                if graticule:
                    hp.graticule(**kw_graticule)
        if return_pmap:
            return axs, pm_list
        else:
            return axs
