"""
Utility functions for fitting 2D halo models to data.
"""

import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from astropy.modeling import Fittable2DModel, FittableModel, fitting
from astropy.utils.exceptions import AstropyUserWarning
from mytools.utils import get_coord, info_fitness

from crafts_stack.halo_models import build_model, get_default_model


def gen_test_data(
    halo_model: Literal["nfw", "gaussian", "lorentz"] = "nfw",
    bg_model: Literal["const", "poly", None] = "const",
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    param_h1: Optional[Dict[str, float]] = None,
    param_h2: Optional[Dict[str, float]] = None,
    param_bg: Optional[Dict[str, float]] = None,
    noise_std: Optional[float] = None,
    seed: float = 42,
) -> np.ndarray:
    """
    Generate test data based on a specified model.

    Parameters
    ----------
    halo_model : str, optional
        Name of the model to use for generating the test data. Default is "nfw".
    bg_model : str, optional
        Name of the background model to use, by default "const"
    x, y : np.ndarray, optional
        Coordinates of the data points. If not provided, they will be generated using `get_coord()`.
    param_h1, param_h2, param_bg : dict, optional
        Keyword arguments for specifying the parameters of model. Default is None to use defaults based on halo_model.
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

    model = build_model(halo_model, bg_model, param_h1, param_h2, param_bg)  # pyright: ignore[reportArgumentType]
    data = model(x, y)

    np.random.seed(seed)
    if noise_std is not None:
        data += np.random.normal(0, noise_std, data.shape)

    return data


def halofit(
    *args: np.ndarray,
    model: Optional[Fittable2DModel] = None,
    mask: Optional[np.ndarray] = None,
    print_model: bool = True,
    **kwargs,
) -> Tuple[List[np.ndarray], Fittable2DModel, Any]:
    """
    Fit 2D data using a provided or auto-generated model.

    Parameters
    ----------
    args : tuple
        Either (data), ((x, y), data), or (x, y, data).
    model : astropy.modeling.models, optional
        Model to use for fitting. \
            If not provided, a default model will be generated using `get_default_model`.
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
        print("Using default model:")
        model = get_default_model()

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
        fit_model = fitter(model, x_fit, y_fit, data_fit, **kwargs)

    if print_model:
        print("\nFitted model:")
        print(fit_model)

    # Compute fitted data and residuals
    fit_data = fit_model(x, y)
    res = data - fit_data
    info_fitness(data_fit, fit_model(x_fit, y_fit), len(model.parameters))

    return [data, fit_data, res], fit_model, fitter.fit_info


def get_fitting_error_bootstrap(
    fit_model: FittableModel,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    fitter: Optional[Any] = None,
    n_bootstrap: int = 100,
    seed: Optional[int] = 42,
    use_residuals: bool = True,
    progress: bool = True,
    return_detailed: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
    """
    Calculate a 2D error map using bootstrap resampling with detailed diagnostics.

    Parameters
    ----------
    fit_model : FittableModel
        The model to fit. Will be copied for each bootstrap iteration.
    x, y, z : np.ndarray
        The coordinates and data values to fit.
    fitter : object, optional
        The fitter to use. If not provided, LevMarLSQFitter() will be used.
    n_bootstrap : int, optional
        Number of bootstrap samples, by default 100.
    seed : int, optional
        Random seed for reproducibility, by default 42.
    use_residuals : bool, optional
        Whether to use residual resampling (True) or case resampling (False).
    progress : bool, optional
        Whether to show progress, by default True.
    return_detailed : bool, optional
        Whether to return detailed diagnostics, by default False.

    Returns
    -------
    np.ndarray or tuple
        The 2D error map, and optionally detailed diagnostics.
    """
    # Input validation
    # if not isinstance(fit_model, FittableModel):
    # raise TypeError("fit_model must be a FittableModel instance")

    if x.shape != y.shape or x.shape != z.shape:
        raise ValueError(
            f"x, y, and z must have same shape. Got x: {x.shape}, y: {y.shape}, z: {z.shape}"
        )

    if n_bootstrap <= 0:
        raise ValueError(f"n_bootstrap must be positive, got {n_bootstrap}")

    # Initialize fitter if not provided
    if fitter is None:
        fitter = fitting.TRFLSQFitter()

    # Get best-fit model to original data
    if progress:
        print("Fitting model to original data...")

    # model_best = fitter(fit_model.copy(), x, y, z)
    # z_fit = model_best(x, y)
    z_fit = fit_model(x, y)
    residuals = z - z_fit

    # Set up random number generator
    rng = np.random.default_rng(seed)

    # Pre-allocate arrays
    model_realizations = np.empty((n_bootstrap, *x.shape))
    bootstrap_parameters = []

    # Track successful fits
    successful_fits = 0

    if progress:
        print(f"Running {n_bootstrap} bootstrap iterations...")

    for i in range(n_bootstrap):
        try:
            if use_residuals:
                # Residual resampling
                bootstrap_residuals = rng.choice(
                    residuals.flatten(), size=residuals.size, replace=True
                ).reshape(residuals.shape)

                z_bootstrap = z_fit + bootstrap_residuals
            else:
                # Case resampling
                indices = rng.choice(z.size, size=z.size, replace=True)
                z_bootstrap = z.flat[indices].reshape(z.shape)

            # Fit model to bootstrap sample
            model_bootstrap = fitter(fit_model.copy(), x, y, z_bootstrap)
            model_realizations[successful_fits] = model_bootstrap(x, y)
            bootstrap_parameters.append(model_bootstrap.parameters.copy())
            successful_fits += 1

            if progress and (i + 1) % max(1, n_bootstrap // 10) == 0:
                print(
                    f"  Completed {i + 1}/{n_bootstrap} iterations ({successful_fits} successful)"
                )

        except Exception as e:
            if progress:
                print(f"  Bootstrap iteration {i + 1} failed: {e}")
            # Skip failed fits

    # Use only successful fits
    if successful_fits == 0:
        raise RuntimeError(
            "All bootstrap iterations failed. Check your model and data."
        )

    if successful_fits < n_bootstrap:
        warnings.warn(
            f"Only {successful_fits} out of {n_bootstrap} bootstrap iterations succeeded."
        )
        model_realizations = model_realizations[:successful_fits]
        bootstrap_parameters = bootstrap_parameters[:successful_fits]

    # Calculate error map
    error_map = np.std(model_realizations, axis=0, ddof=1)

    # Calculate parameter statistics
    bootstrap_parameters = np.array(bootstrap_parameters)
    param_means = np.mean(bootstrap_parameters, axis=0)
    param_stds = np.std(bootstrap_parameters, axis=0, ddof=1)

    if progress:
        print("\nBootstrap error map statistics:")
        print(f"  Min: {error_map.min():.6f}")
        print(f"  Max: {error_map.max():.6f}")
        print(f"  Median: {np.median(error_map):.6f}")
        print(f"  Mean: {error_map.mean():.6f}")

        print("\nParameter uncertainties from bootstrap:")
        for i, name in enumerate(fit_model.param_names):
            print(
                f"  {name}: {param_stds[i]:.6f} (relative: {param_stds[i] / np.abs(param_means[i]):.3%})"
            )

    if return_detailed:
        diagnostics = {
            "model_realizations": model_realizations,
            "bootstrap_parameters": bootstrap_parameters,
            "param_means": param_means,
            "param_stds": param_stds,
            # "residual_std": residual_std,
            # "reduced_chi_squared": reduced_chi_squared,
            "successful_fits": successful_fits,
            # "model_best": model_best,
        }
        return error_map, diagnostics

    return error_map


def get_fitting_error_mc(
    fit_model: FittableModel,
    x: np.ndarray,
    y: np.ndarray,
    cov: Optional[np.ndarray] = None,
    fit_info: Optional[Dict[str, Any]] = None,
    n_samples: int = 100,
    seed: Optional[int] = 42,
    scale_cov: float = 1.0,
    return_detailed: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
    """
    Monte Carlo error estimation with scaling option and diagnostics.

    Parameters
    ----------
    scale_cov : float, optional
        Factor to scale the covariance matrix, by default 1.0.
        Use this if you suspect the covariance matrix is over/under-estimated.
    """
    # Get covariance matrix
    if cov is None:
        if fit_info and "param_cov" in fit_info and fit_info["param_cov"] is not None:
            cov = fit_info["param_cov"]
        else:
            warnings.warn(
                "Covariance matrix not provided and not found in fit_info. Returning zeros."
            )
            if return_detailed:
                return np.zeros(x.shape, dtype=float), {}
            return np.zeros(x.shape, dtype=float)

    # Scale covariance if requested
    if scale_cov != 1.0:
        cov = cov * scale_cov  # pyright: ignore[reportOptionalOperand]
        print(f"Scaled covariance matrix by factor {scale_cov}")

    # Identify free parameters
    free_mask = np.array(
        [
            not (fit_model.fixed[name] or fit_model.tied[name])
            for name in fit_model.param_names
        ]
    )

    p_all_best = np.array(fit_model.parameters)
    p_free_best = p_all_best[free_mask]

    if cov.shape[0] != len(p_free_best):  # pyright: ignore[reportOptionalMemberAccess]
        raise ValueError(
            f"Covariance matrix shape {cov.shape} doesn't match free parameters {len(p_free_best)}"  # pyright: ignore[reportOptionalMemberAccess]
        )

    print("MC parameter uncertainties from covariance matrix:")
    param_stds_mc = np.sqrt(np.diag(cov))  # pyright: ignore[reportCallIssue, reportArgumentType]
    for i, (name, is_free) in enumerate(zip(fit_model.param_names, free_mask)):
        if is_free:
            idx = np.sum(free_mask[:i])  # Index in free parameters
            print(
                f"  {name}: {param_stds_mc[idx]:.6f} (relative: {param_stds_mc[idx] / np.abs(p_all_best[i]):.3%})"
            )

    # Generate parameter samples
    rng = np.random.default_rng(seed)
    cov_sym = (cov + cov.T) / 2.0  # pyright: ignore[reportOptionalMemberAccess]

    try:
        free_param_samples = rng.multivariate_normal(
            p_free_best, cov_sym, size=n_samples
        )
    except np.linalg.LinAlgError:
        warnings.warn("Using diagonal covariance approximation")
        param_std = np.sqrt(np.abs(np.diag(cov_sym)))
        free_param_samples = rng.normal(
            p_free_best, param_std, size=(n_samples, len(p_free_best))
        )

    # Evaluate model for each parameter sample
    model_realizations = []
    mc_parameters = []
    original_params = fit_model.parameters.copy()

    for free_params_sample in free_param_samples:
        p_sample_all = p_all_best.copy()
        p_sample_all[free_mask] = free_params_sample
        fit_model.parameters = p_sample_all
        model_realizations.append(fit_model(x, y))
        mc_parameters.append(p_sample_all.copy())

    fit_model.parameters = original_params  # Restore original parameters

    error_map = np.std(model_realizations, axis=0, ddof=1)

    print("MC error map statistics:")
    print(f"  Min: {error_map.min():.6f}")
    print(f"  Max: {error_map.max():.6f}")
    print(f"  Median: {np.median(error_map):.6f}")
    print(f"  Mean: {error_map.mean():.6f}")

    if return_detailed:
        diagnostics = {
            "model_realizations": np.array(model_realizations),
            "mc_parameters": np.array(mc_parameters),
            "free_param_samples": free_param_samples,
            "cov_matrix": cov,
        }
        return error_map, diagnostics

    return error_map
