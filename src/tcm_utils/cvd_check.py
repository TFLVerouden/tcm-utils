import os
from daltonlens import simulate
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Set of functions to check whether Tommie can distinguish the lines in your
# plot. Based on the DaltonLens-Python library, tested on the mpl_colorcycle.png
# file from the matplotlib repository (containing the default colorcycle), using
# the MacBook screen. With the below settings, the output image is identical
# to the original for Tommie. Conclusion: when skipping green, purple, brown and
# pink, there are still six colours that he can distinguish.

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
                         suffix="_cvdsim"):
    """
    Simulates colorblindness on an image file and saves the result with a
    suffix. Not tested on filetypes that are not png.

    :param file_path: Path to the input image file.
    :param deficiency: Type of colorblindness (default: PROTAN).
        Use "MONOCHROME" for grayscale.
    :param severity: Severity of the colorblindness (default: 0.6).
        Ignored for monochrome.
    :param suffix: Suffix to append to the output file name (default: "_cvd").

    :return: Path of the saved image with the suffix.
    """
    # Load the image
    image = Image.open(file_path).convert("RGB")
    image_array = np.array(image)

    # Simulate colorblindness
    simulated_array = simulate_cvd(image_array, deficiency, severity)

    # Convert back to an image
    simulated_image = Image.fromarray(np.uint8(simulated_array))

    # Ensure the output directory exists
    output_dir = os.path.dirname(file_path)
    os.makedirs(output_dir) if not os.path.exists(output_dir) else None

    # Save the image with "_cvd" suffix
    base, ext = os.path.splitext(file_path)
    simulated_image.save(f"{base}{suffix}{ext}")

    # Return the path of the saved image
    return f"{base}{suffix}{ext}"


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


if __name__ == "__main__":
    # Define a function to plot the color cycle
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import TABLEAU_COLORS


    def plot_color_cycle(output_path, title="Colors in the property cycle"):
        """
        Plots the current Matplotlib color cycle with TABLEAU_COLORS names and saves the figure.

        :param output_path: Path to save the output image.
        :param title: Title of the plot.
        """

        def f(xi, a):
            """A sigmoid-like parametrized curve."""
            return 0.85 * a * (1 / (1 + np.exp(-xi)) + 0.2)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        ax.set_title(title)

        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        x = np.linspace(-4, 4, 200)

        # Match colors with TABLEAU_COLORS names
        tableau_colors = {color: name for name, color in TABLEAU_COLORS.items()}
        matched_colors = [tableau_colors.get(color, "unnamed") for color in
                          colors]

        # Calculate dynamic y-spacing to center the lines
        num_colors = len(colors)
        y_positions = np.linspace(num_colors / 2, -num_colors / 2, num_colors)

        for i, (color, color_name, pos) in enumerate(
                zip(colors, matched_colors, y_positions)):
            ax.plot(x, f(x, pos), color=color)
            ax.text(4.2, pos, f"'C{i}': '{color_name}'", color=color,
                    va="center")
            ax.bar(9, 1, width=2, bottom=pos - 0.5, color=color)

        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    # Path to the mpl_colorcycle.png file
    cvd_check_image = "cvd_check/color_cycle.png"

    # Generate the default color cycle plot
    plot_color_cycle(cvd_check_image,
                     title="Colors in the default property cycle")

    # Simulate colorblindness and generate a monochrome version
    cvd_image = simulate_cvd_on_file(cvd_check_image)
    monochrome_image = simulate_cvd_on_file(cvd_check_image,
                                            deficiency="MONOCHROME",
                                            suffix="_mono")
    print(f"Colorblindness simulation and monochrome image saved.")

    # Adjust the color cycle and generate the adjusted plot
    adjusted_image = "cvd_check/color_cycles/removed_colors.png"
    set_cvd_friendly_colors(do_print=True)
    plot_color_cycle(adjusted_image,
                     title="Colors in the adjusted property cycle")

    # Also make a version using the tableau-colorblind10 colors
    adjusted_image2 = "cvd_check/color_cycles/tableau-colorblind10.png"
    set_cvd_friendly_colors(style="tableau-colorblind10", do_print=True)
    plot_color_cycle(adjusted_image2,
                     title="Colors in the tableau_colorblind10 property cycle")

    # Reset the color cycle to default and print it
    set_cvd_friendly_colors(do_reset=True, do_print=True)