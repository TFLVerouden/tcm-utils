import os
from pathlib import Path
from daltonlens import simulate
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Set of functions to check whether Tommie can distinguish the lines in your
# plot. Based on the DaltonLens-Python library, tested on the mpl_colorcycle.png
# file from the matplotlib repository (containing the default colorcycle), using
# the MacBook screen. With the settings as specified in examples/run_cvd_demo,
# the output image is identical to the original for Tommie. Conclusion: when
# skipping green, purple, brown and pink, there are still six colours that he
# can distinguish.

# Note that the colour rendering of the screen is an important factor too;
# beamers are notoriously bad at this. To be completely sure, you can generate a
# monochrome version of the image (using the luminosity method, which is a
# weighted average of the RGB values; W_r = 0.2126, W_g = 0.7152, W_b = 0.0722).
# If you can see the contrast between things in this image, you can be sure
# that anyone with any type of colorblindness can also distinguish these things.

# DaltonLens-Python: https://github.com/DaltonLens/DaltonLens-Python/tree/master
# Explanation: https://daltonlens.org/colorblindness-simulator
# Monochrome weights: https://en.wikipedia.org/wiki/Grayscale
# BW-conversion: https://digital-photography-school.com/black-and-white-conversions-an-introduction-to-luminosity/
# Colorcycle: https://matplotlib.org/stable/gallery/color/color_cycle_default.html#
# More info: https://www.tableau.com/blog/examining-data-viz-rules-dont-use-red-green-together


def simulate_cvd(image_array, deficiency="PROTAN", severity=0.6):
    """
    Simulates colorblindness or converts to monochrome on an RGB image array.

    :param image_array: Input RGB image as a NumPy array. The array should have
        shape (height, width, 3) and values in range [0, 255].
    :param deficiency: Type of colorblindness (default: PROTAN).
        Use "MONOCHROME" for grayscale.
    :param severity: Severity of the colorblindness (default: 0.6).
        Ignored for monochrome.

    :return: Simulated RGB image as a NumPy array.
    """
    if deficiency == "MONOCHROME":
        # Convert to grayscale using luminosity method
        grayscale_array = np.dot(image_array[..., :3], [0.299, 0.587, 0.114])

        # Convert back to 3 channels
        return np.stack((grayscale_array,) * 3, axis=-1)
    else:  # Use daltonlens for colorblindness simulation
        # Create a simulator using the Viénot 1999 algorithm
        simulator = simulate.Simulator_Vienot1999()

        # Apply the colorblindness simulation
        return simulator.simulate_cvd(image_array,
                                      deficiency=getattr(simulate.Deficiency,
                                                         deficiency),
                                      severity=severity)


def simulate_cvd_on_file(file_path, deficiency="PROTAN", severity=0.6,
                         suffix="_cvdsim", output_dir=None):
    """
    Simulates colorblindness on an image file and saves the result with a
    suffix. Not tested on filetypes that are not png.

    :param file_path: Path to the input image file.
    :param deficiency: Type of colorblindness (default: PROTAN).
        Use "MONOCHROME" for grayscale.
    :param severity: Severity of the colorblindness (default: 0.6).
        Ignored for monochrome.
    :param suffix: Suffix to append to the output file name (default: "_cvd").
    :param output_dir: Optional directory to save the output file. Defaults to
        the directory of file_path.

    :return: Path of the saved image with the suffix.
    """
    # Load the image
    image_path = Path(file_path).expanduser().resolve()
    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image)

    # Simulate colorblindness
    simulated_array = simulate_cvd(image_array, deficiency, severity)

    # Convert back to an image
    simulated_image = Image.fromarray(np.uint8(simulated_array))

    # Ensure the output directory exists
    target_dir = Path(output_dir).expanduser() if output_dir else image_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    # Save the image with "_cvd" suffix
    base, ext = os.path.splitext(image_path.name)
    target_path = target_dir / f"{base}{suffix}{ext}"
    simulated_image.save(target_path)

    # Return the path of the saved image
    return str(target_path)


def set_cvd_friendly_colors(style="adjusted", do_reset=False, do_print=False):
    """
    Sets the Matplotlib color cycle and colormap for color vision deficiency
    (CVD) friendliness. Can reset to default settings or use the
    'tableau-colorblind10' style.

    :param style: Style to use ('adjusted' or 'tableau-colorblind10').
        Default is 'adjusted'.
    :param do_reset: If True, resets the settings to the default.
    :param do_print: If True, prints the current color cycle and colormap.
    """
    if do_reset:
        # Reset only the parameters changed by this function
        plt.rcParams['axes.prop_cycle'] = matplotlib.rcParamsDefault['axes.prop_cycle']
        plt.rcParams['image.cmap'] = matplotlib.rcParamsDefault['image.cmap']
    else:
        # Set the colormap to Cividis
        plt.rcParams['image.cmap'] = 'cividis'

        if style == "adjusted":
            # Default Matplotlib color cycle
            default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

            # Colors to skip
            skip_colors = {'#2ca02c',  # Green
                           '#9467bd',  # Purple
                           '#8c564b',  # Brown
                           '#e377c2'}  # Pink

            # Filter out the colors to skip
            adjusted_colors = [color for color in default_colors
                               if color not in skip_colors]

            # Add grey to the end of the list
            grey_color = '#7f7f7f'  # Grey
            adjusted_colors.remove(grey_color)
            adjusted_colors.append(grey_color)

            # Update Matplotlib's color cycle
            plt.rcParams['axes.prop_cycle'] = plt.cycler(color=adjusted_colors)
        elif style == "tableau-colorblind10":
            # Set the style to tableau-colorblind10
            plt.style.use('tableau-colorblind10')

    # Print the updated settings if requested
    if do_print:
        current_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        current_cmap = plt.rcParams['image.cmap']
        print("Current color cycle:", current_colors)
        print("Current colormap:", current_cmap)

    # Return color cycle
    return(plt.rcParams['axes.prop_cycle'].by_key()['color'])


def get_color(n):
    """
    Returns the nth color from the current Matplotlib color cycle.
    
    :param n: Index of the color to retrieve (0-indexed).
    :return: Color string (hex format) from the current color cycle.
    """
    current_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    return current_colors[n % len(current_colors)]