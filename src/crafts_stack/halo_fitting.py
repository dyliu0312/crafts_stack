"""
Utility functions for fitting 2D halo models to data.
"""

import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from astropy.modeling import fitting
from astropy.utils.exceptions import AstropyUserWarning
from mytools.halo_new import get_coord, info_fitness

from crafts_stack.halo_models import build_model


def gen_test_data(
    model_name: str = "nfw",
    background_name: str = "const",
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    kw_h1: Optional[Dict[str, float]] = None,
    kw_h2: Optional[Dict[str, float]] = None,
    kw_bg: Optional[Dict[str, float]] = None,
    noise_std: Optional[float] = None,
    seed: float = 42,
) -> np.ndarray:
    """
    Generate test data based on a specified model.

    Parameters
    ----------
    model_name : str, optional
        Name of the model to use for generating the test data. Default is "nfw".
    background_name : str, optional
        Name of the background model to use, by default "const"
    x, y : np.ndarray, optional
        Coordinates of the data points. If not provided, they will be generated using `get_coord()`.
    kw_h1, kw_h2, kw_bg : dict, optional
        Keyword arguments for the model. Default is None to use defaults based on model_name.
    noise_std : float, optional
        Standard deviation of the Gaussian noise to be added to the data. If None, no noise is added.
    seed : int, optional
        Seed for the random number generator. Default is 42.

    Returns
    -------
    data : np.ndarray
        The generated test data.
    """
    if x is None and y is None:
        x, y = get_coord()
    elif x is None or y is None:
        y = x if y is None else y
        x = y if x is None else x

    model = build_model(model_name, background_name, kw_h1, kw_h2, kw_bg)  # pyright: ignore[reportArgumentType]
    data = model(x, y)

    np.random.seed(seed)
    if noise_std is not None:
        data += np.random.normal(0, noise_std, data.shape)

    return data


def halofit(
    *args: np.ndarray,
    model: Optional[Any] = None,
    model_name: Literal["nfw", "gaussian", "lorentz"] = "nfw",
    background_name: Literal["const", "poly"] = "const",
    mask: Optional[np.ndarray] = None,
    print_model: bool = True,
) -> Tuple[List[np.ndarray], Any]:
    """
    Fit 2D data using a provided or auto-generated model.

    Parameters
    ----------
    args : tuple
        Either (data), ((x, y), data), or (x, y, data).
    model : astropy.modeling.models, optional
        Model to use for fitting. If not provided, a default model will be generated based on the model_name.
    model_name : str, optional
        Name of the model to use for fitting. Default is "nfw".
    background_name : str, optional
        Name of the background model to use, by default "const"
    mask : np.ndarray, optional
        Mask to apply to the data before fitting.
    print_model : bool, optional
        Whether to print the fitted model. Default is True.

    Returns
    -------
    Tuple[List[np.ndarray], Any]
    """
    # Parse input arguments
    if len(args) == 1:
        data = args[0]
        x, y = get_coord()
    elif len(args) == 2:
        (x, y), data = args
    elif len(args) == 3:
        x, y, data = args
    else:
        raise ValueError(
            "Invalid number of arguments. Use (data), ((x, y), data), or (x, y, data)."
        )

    # Build model if not provided
    if model is None:
        model = build_model(model_name, background_name=background_name)

    # Apply mask if available
    if mask is not None:
        x_fit, y_fit, data_fit = x[mask], y[mask], data[mask]
    else:
        x_fit, y_fit, data_fit = x, y, data

    # Perform fitting
    fitter = fitting.TRFLSQFitter()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=AstropyUserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        fit_model = fitter(model, x_fit, y_fit, data_fit)

    if print_model:
        if model_name:
            print(f"\nFitted double {model_name.upper()} model:")
        print(fit_model)

    # Compute fitted data and residuals
    fit_data = fit_model(x, y)
    res = info_fitness(data, fit_data)

    return [data, fit_data, res], fitter.fit_info  # pyright: ignore[reportReturnType]
