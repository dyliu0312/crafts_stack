import os

import numpy as np
from matplotlib import pyplot as plt


class Measurement:
    """
    A class representing a numerical value with an associated uncertainty (error).
    Supports basic arithmetic operations: addition, subtraction, multiplication, and division.
    Supports slicing (e.g., m[:1]) to obtain subsets.
    """

    def __init__(self, value, error):
        """
        Initializes a Measurement object.

        Parameters:
        - value (float/np.ndarray): The measured value or calculated result.
        - error (float/np.ndarray): The uncertainty or error of the measurement.
        """
        # Enforce conversion to numpy arrays for consistent handling
        self.value = np.array(value)
        self.error = np.array(error)

        if self.value.shape != self.error.shape:
            raise ValueError("Value and error arrays must have the same shape.")

    def __add__(self, other):
        """
        Overloads the addition operator (+).
        Error propagation formula for Z = A + B:
        Delta_Z = sqrt(Delta_A^2 + Delta_B^2)
        """
        if isinstance(other, (int, float)):
            # Convert constant to a Measurement object compatible with self's shape
            other_value = np.full_like(self.value, other)
            other_error = np.zeros_like(self.error)
            other = Measurement(other_value, other_error)

        new_value = self.value + other.value
        new_error = np.sqrt(self.error**2 + other.error**2)
        return Measurement(new_value, new_error)

    def __sub__(self, other):
        """
        Overloads the subtraction operator (-).
        Error propagation formula for Z = A - B:
        Delta_Z = sqrt(Delta_A^2 + Delta_B^2)
        """
        if isinstance(other, (int, float)):
            other_value = np.full_like(self.value, other)
            other_error = np.zeros_like(self.error)
            other = Measurement(other_value, other_error)

        new_value = self.value - other.value
        new_error = np.sqrt(self.error**2 + other.error**2)
        return Measurement(new_value, new_error)

    def __mul__(self, other):
        """
        Overloads the multiplication operator (*).
        Error propagation formula for Z = A * B:
        Delta_Z = sqrt((B*Delta_A)^2 + (A*Delta_B)^2)
        """
        if isinstance(other, (int, float)):
            other_value = np.full_like(self.value, other)
            other_error = np.zeros_like(self.error)
            other = Measurement(other_value, other_error)

        new_value = self.value * other.value
        new_error = np.sqrt(
            (other.value * self.error) ** 2 + (self.value * other.error) ** 2
        )
        return Measurement(new_value, new_error)

    def __truediv__(self, other):
        """
        Overloads the division operator (/).
        Error propagation formula for Z = A / B:
        Delta_Z = sqrt((Delta_A/B)^2 + (A*Delta_B/B^2)^2)
        """
        if isinstance(other, (int, float)):
            if other == 0:
                raise ValueError("Cannot divide by zero.")
            other_value = np.full_like(self.value, other)
            other_error = np.zeros_like(self.error)
            other = Measurement(other_value, other_error)

        if np.any(other.value == 0):
            raise ValueError(
                "Cannot divide by a Measurement object with a value of zero."
            )

        new_value = self.value / other.value
        # The denominator other.value has been checked for zero.
        new_error = np.sqrt(
            (self.error / other.value) ** 2
            + (self.value * other.error / other.value**2) ** 2
        )
        return Measurement(new_value, new_error)

    def __getitem__(self, key):
        """
        Supports getting subsets via indexing or slicing.
        e.g., m[0], m[:5], m[[1, 3, 5]]

        Parameters:
        - key: An integer index, slice object, or boolean array.

        Returns:
        - A new Measurement instance containing the subset.
        """
        # Use numpy array's slicing capability
        new_value = self.value[key]
        new_error = self.error[key]

        # Return a new Measurement instance
        return Measurement(new_value, new_error)

    def __repr__(self):
        """
        Defines the string representation of the object for printing and debugging.
        """
        # Handle single value
        if self.value.ndim == 0 or self.value.size == 1:
            return f"{self.value.item():.4f} +/- {self.error.item():.4f}"
        else:
            # Display array representation (with truncation for large arrays)
            values_repr = np.array2string(
                self.value,
                max_line_width=np.inf,  # type: ignore
                edgeitems=2,
            )
            errors_repr = np.array2string(
                self.error,
                max_line_width=np.inf,  # type: ignore
                edgeitems=2,
            )

            return f"Values (size={self.value.size}): {values_repr}\nErrors (size={self.error.size}): {errors_repr}"

    def save(self, filepath):
        """
        Saves the value and error arrays to a compressed .npz file.

        Parameters:
        - filepath (str): The path to the file (e.g., 'data.npz').
        """
        np.savez_compressed(filepath, value=self.value, error=self.error)
        print(f"Measurement data saved to {filepath}")

    @classmethod
    def load_from_file(cls, filepath):
        """
        Loads the value and error arrays from a .npz file and returns a new Measurement object.

        Parameters:
        - filepath (str): The path to the file (e.g., 'data.npz').

        Returns:
        - Measurement: A new Measurement instance loaded from the file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            data = np.load(filepath)
            if "value" not in data or "error" not in data:
                raise KeyError("File must contain 'value' and 'error' keys.")

            value = data["value"]
            error = data["error"]
            data.close()  # Close the file handle

            # Use cls() to instantiate the class method
            return cls(value, error)
        except Exception as e:
            print(f"Error loading file {filepath}: {e}")
            raise

    def get_snr(self, noise_measurement):
        """
        Calculates Signal-to-Noise Ratio (SNR) based on:
        SNR = self.value / noise_measurement.error

        The error of the resulting SNR is propagated assuming the noise_measurement.error
        is exact (i.e., its uncertainty is zero), resulting in:
        Delta_SNR = self.error / noise_measurement.error

        Parameters:
        - noise_measurement (Measurement): A second Measurement object whose error
                                           is used as the denominator (Noise).

        Returns:
        - Measurement: A new Measurement object representing the calculated SNR
                       and its propagated uncertainty.
        """
        if not isinstance(noise_measurement, Measurement):
            raise TypeError("noise_measurement must be a Measurement object.")

        noise = noise_measurement.error

        if np.any(noise == 0):
            raise ValueError(
                "Noise (noise_measurement.error) cannot be zero for SNR calculation."
            )

        # 1. New Value: Z = A / B
        new_value = self.value / noise

        # 2. New Error: Delta_Z = |Delta_A / B| (assuming Delta_B = 0)
        new_error = np.abs(self.error / noise)

        return Measurement(new_value, new_error)

    # Plotting methods
    def plot(
        self,
        ax=None,
        x=None,
        label=None,
        capsize=5,
        fmt="-o",
        grid=True,
        tick_in=True,
        **kwargs,
    ):
        """
        Plots the measurement values with error bars.

        Parameters:
        - ax: The matplotlib axis to plot on. If None, the current axis is used.
        - x: The x-axis data to plot. If None, default is an array of indices.
        - label: The label for the plot legend.
        - capsize: The size of the error bar caps.
        - fmt: The format string for the plot (e.g., '-o' for line with circles).
        - **kwargs: Additional keyword arguments passed to the errorbar function.
        """
        if x is None:
            x = np.arange(len(self.value))

        if ax is None:
            ax = plt.gca()

        ax.errorbar(
            x,
            self.value,
            yerr=self.error,
            capsize=capsize,
            label=label,
            fmt=fmt,
            **kwargs,
        )
        if label is not None:
            ax.legend()
        if grid:
            ax.grid(True, which="major", axis="both")
        if tick_in:
            ax.tick_params(axis="both", direction="in")

    def fill_plot(
        self,
        ax=None,
        x=None,
        label=None,
        alpha=0.3,
        color=None,
        linestyle="--",
        grid=True,
        tick_in=True,
        **kwargs,
    ):
        """
        Plots the measurement as a line with a filled region representing the error.

        Parameters:
        - ax: The matplotlib axis to plot on. If None, the current axis is used.
        - x: The x-axis data to plot. If None, default is an array of indices.
             If self contains a single value and x is an array, it plots a horizontal line/band.
        - label: The label for the plot legend (applied to the line).
        - alpha: Transparency level for the filled error region.
        - color: Color for both the line and the filled region.
        - linestyle: Linestyle for the center value (e.g., '-', '--', 'None').
        - grid: Whether to enable grid lines.
        - tick_in: Whether to set tick direction to 'in'.
        - **kwargs: Additional keyword arguments passed to the plot function for the center line.
        """
        if ax is None:
            ax = plt.gca()

        is_single_measurement = self.value.size == 1

        # 1. Handle X-data based on size
        if x is None:
            # If x is not provided, use indices
            x_plot = np.arange(len(self.value))
        else:
            # Convert x to array and handle single value case
            x = np.asarray(x)
            if is_single_measurement and x.ndim > 0 and x.size > 1:
                # Case: Single Measurement, but array of X-coordinates provided (Draw horizontal line/band)
                x_plot = x
            elif x.size != self.value.size:
                raise ValueError(
                    f"X data size ({x.size}) must match Measurement size ({self.value.size})."
                )
            else:
                x_plot = x

        # 2. Prepare Y-data (Value and Bounds)
        if is_single_measurement and x_plot.size > 1:
            # Replicate single value/error to match x_plot size
            y_value = np.full_like(x_plot, self.value.item(), dtype=float)
            y_error = np.full_like(x_plot, self.error.item(), dtype=float)
            print(f"Single measurement replicated to match x_plot size: {y_value.size}")
        else:
            y_value = self.value
            y_error = self.error

        lower_bound = y_value - y_error
        upper_bound = y_value + y_error

        # 3. Draw the filled error region (fill_between)
        fill_area = ax.fill_between(
            x_plot,
            lower_bound,
            upper_bound,
            alpha=alpha,
            color=color,
        )

        # Determine the color to use for the line (either specified or from the fill cycle)
        line_color = color if color is not None else fill_area.get_facecolor()[0]

        # 4. Draw the center value line
        if linestyle != "None":
            ax.plot(
                x_plot,
                y_value,
                linestyle=linestyle,
                color=line_color,
                label=label,
                **kwargs,
            )

        if label is not None:
            ax.legend()
        if grid:
            ax.grid(True, which="major", axis="both")
        if tick_in:
            ax.tick_params(axis="both", direction="in")
