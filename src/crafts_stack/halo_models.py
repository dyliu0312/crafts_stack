"""
Analytical models for the halo fitting.

This module contains the definitions of the analytical models, implemented as classes that inherit from `astropy.modeling.Fittable2DModel`.

Currently included models are:
- `NFW2D`: Elliptical 2D NFW profile.
- `Lorentz2D`: Elliptical 2D Lorentzian profile.
- `Gaussian2D`: 2D Gaussian profile.

"""

import numpy as np
from astropy.modeling import Fittable2DModel, Parameter, models


class NFW2D(Fittable2DModel):
    r"""
    Elliptical 2D NFW profile.

    $$\rho(r) = \frac{\rho_0}{\frac{r}{R_s} \left( 1 + \frac{r}{R_s} \right)^2}$$
    """

    amplitude = Parameter(default=1.0, min=0)  # ρ₀ - characteristic density
    x_mean = Parameter(default=0.0)  # x center position
    y_mean = Parameter(default=0.0)  # y center position
    r_s = Parameter(default=1.0, min=0.1)  # scale radius
    ellipticity = Parameter(default=0.0, min=0, max=0.9)  # 1 - b/a
    theta = Parameter(default=0.0)  # position angle in radians

    @staticmethod
    def evaluate(x, y, amplitude, x_mean, y_mean, r_s, ellipticity, theta):
        """
        Evaluate the NFW density profile on 2D xy grid
        """
        q = 1.0 - ellipticity
        # Add a small epsilon to avoid division by zero for q=0 (ellipticity=1)
        q = np.maximum(q, 1e-8)

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        x_c = x - x_mean
        y_c = y - y_mean

        x_rot = x_c * cos_theta + y_c * sin_theta
        y_rot = -x_c * sin_theta + y_c * cos_theta

        r = np.sqrt(x_rot**2 + (y_rot / q) ** 2)

        # Dimensionless radius, ensure it\'s not exactly zero
        x_val = np.maximum(r / r_s, 1e-8)

        # 3D NFW density formula
        density = amplitude / (x_val * (1 + x_val) ** 2)

        return density


class Lorentz2D(Fittable2DModel):
    r"""
    Elliptical 2D Lorentzian profile.
    """

    amplitude = Parameter(default=1.0, min=0)
    x_mean = Parameter(default=0.0)
    y_mean = Parameter(default=0.0)
    fwhm = Parameter(default=1.0, min=0.1)  # Full width at half maximum
    ellipticity = Parameter(default=0.0, min=0, max=0.9)
    theta = Parameter(default=0.0)

    @staticmethod
    def evaluate(x, y, amplitude, x_mean, y_mean, fwhm, ellipticity, theta):
        """
        Evaluate the Lorentzian profile on 2D xy grid
        """
        q = 1.0 - ellipticity
        q = np.maximum(q, 1e-8)

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        x_c = x - x_mean
        y_c = y - y_mean

        x_rot = x_c * cos_theta + y_c * sin_theta
        y_rot = -x_c * sin_theta + y_c * cos_theta

        r_sq = x_rot**2 + (y_rot / q) ** 2

        # gamma is HWHM
        gamma = fwhm / 2.0

        density = amplitude / (1 + r_sq / gamma**2)

        return density


# Dictionary of available models
AVAILABLE_MODELS = {
    "nfw": NFW2D,
    "gaussian": models.Gaussian2D,
    "lorentz": Lorentz2D,
}


def get_model(model_name: str):
    """
    Get a halo model class by its name.
    """
    model_name = model_name.lower()
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Model '{model_name}' not available. "
            f"Choose from {list(AVAILABLE_MODELS.keys())}"
        )
    return AVAILABLE_MODELS[model_name]
