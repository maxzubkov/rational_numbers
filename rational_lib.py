import math
import mpld3
import numpy as np
import matplotlib.pyplot as plt

######################################################################################################

def make_reduce(point):
    """
    Args:
        point (tuple): A tuple (a, b) where a and b are integers.

    Returns:
        tuple: The reduced point (a / gcd(a,b), b / gcd(a,b)), or a message in case of invalid input.
    """
    a, b = point
    
    if b == 0:
        raise ValueError("Denominator cannot be zero.")
    if a == 0:
        return (0, b)
    
    # Determine the signs of a and b
    sign_a = (1 if a > 0 else -1)
    sign_b = (1 if b > 0 else -1)
    
    # Compute the absolute values and reduce
    a, b = abs(a), abs(b)
    gcd = math.gcd(a, b)
    a, b = (a // gcd, b // gcd)
    
    return (sign_a * a, sign_b * b)


def make_proper(point):
    """
    Converts a point (a, b) to a 'proper' reduced form (a', b) with  |a'| < |b|.

    Args:
        point (tuple): A tuple (a, b) where a and b are integers.

    Returns:
        tuple: The proper point (a', b'), or raises an error for invalid input.
    """
    a, b = point

    if b == 0:
        raise ValueError("Denominator cannot be zero.")
    if a == 0:
        return (0, b)
    
    # Determine the signs of a and b
    sign_a = (1 if a > 0 else -1)
    sign_b = (1 if b > 0 else -1)
    
    # Compute absolute values
    a, b = abs(a), abs(b)
    
    # Make `a` proper 
    a %= b
    
    return (sign_a * a, sign_b * b)

def get_proper_points(lattice):
    temp = []
    for point in lattice:
        temp.append(make_proper(point))
    return np.array(temp)

def get_reduced_points(lattice):
    temp = []
    for point in lattice:
        temp.append(make_reduce(point))
    return np.array(temp)

def filter_with_gcd(lattice):
    temp = []
    for point in lattice:
        if gcd_check(point):
            temp.append(point)
    return np.array(temp)

def gcd_check(point):
    """
    Checks if the greatest common divisor (GCD) of the two numbers in the point is 1.

    Args:
        point (tuple): A tuple (a, b) where a and b are integers.

    Returns:
        bool: True if GCD(a, b) is 1; False otherwise.
    """
    a, b = point
    return abs(a) == 1 or abs(b) == 1 or math.gcd(a, b) == 1

def color_rational_point(point):
    temp = make_proper(point)
    a = abs(int(temp[0]))
    b = abs(int(temp[1]))
    gcd = math.gcd(a, b)
    a = a / gcd
    b = b / gcd
    color = a / b
    
    return color

######################################################################################################

def generate_lattice(n, dimensions, step_sizes, marker=None, dtype=np.float64):
    """
    Generate a regular lattice.
    
    Parameters:
    - n: int, number of dimensions
    - dimensions: list of int, size of the lattice along each dimension
    - step_sizes: list of float, step size between lattice points along each dimension
    - marker: array-like or None, point to add to each lattice point (for translation)
    - dtype: data-type, desired data-type for array elements (e.g., np.float32, np.float64, np.float16)
    
    Returns:
    - lattice: NumPy array, coordinates of the lattice points with specified dtype
    """

    if len(dimensions) != n or len(step_sizes) != n:
        raise ValueError(f"Length of dimensions ({len(dimensions)}) and step_sizes ({len(step_sizes)}) must be equal to n ({n})")
    
    if marker is not None and len(marker) != n:
        raise ValueError(f"Length of marker ({len(marker)}) must be equal to the number of dimensions ({n})")
    
    # Generate the lattice using NumPy's meshgrid function
    grids = np.meshgrid(*[np.arange(0, dim+step, step, dtype=dtype) for dim, step in zip(dimensions, step_sizes)], indexing='ij')
    
    # Reshape and stack the arrays to create coordinates for all points in the lattice
    lattice = np.vstack([grid.ravel() for grid in grids]).T.astype(dtype)

    # Add the marker to each point, if specified
    if marker is not None:
        lattice += np.array(marker, dtype=dtype)
    
    return lattice

######################################################################################################

def plot(data, colors=None, graph_info=None):
    """
    Plots a scatter graph with optional colors and graph information.

    Args:
        data (np.ndarray): 2D array with shape (n, 2), where n is the number of points.
        colors (list or np.ndarray, optional): Color values for the points. Defaults to None.
        graph_info (list or tuple, optional): Contains labels [xlabel, ylabel, title]. Defaults to None.
    """
    # Preprocess colors
    if colors is not None and isinstance(colors[0], str):
        colors = colors_to_numeric(colors)
    
    # Setup plot
    plt.close()
    fig, ax = plt.subplots()

    x, y = data.T
    point_size = 10

    # Handle colors
    scatter_params = {'x': x, 'y': y, 's': point_size}
    if colors is not None:
        scatter_params['c'] = colors
        if all(isinstance(c, (int, float)) for c in colors):
            scatter_params.update({'cmap': 'jet', 'vmin': min(colors), 'vmax': max(colors)})
    else:
        scatter_params['c'] = 'black'
    
    sc = ax.scatter(**scatter_params)
    if 'cmap' in scatter_params:
        plt.colorbar(sc)

    # Add graph information
    if graph_info:
        ax.set_xlabel(graph_info[0])
        ax.set_ylabel(graph_info[1])
        ax.set_title(graph_info[2])

    plt.show()

def save_plot(data, colors=None, graph_info=None, save_dir=None):
    """
    Saves a scatter plot as an interactive HTML file.

    Args:
        data (np.ndarray): 2D array with shape (n, 2), where n is the number of points.
        colors (list or np.ndarray, optional): Color values for the points. Defaults to None.
        graph_info (list or tuple, optional): Contains labels [xlabel, ylabel, title]. Defaults to ['x', 'y', 'graph'].
        save_dir (str, optional): Directory to save the HTML file. Defaults to the current directory.
    """
    # Set default graph_info if None
    if graph_info is None:
        graph_info = ['x', 'y', 'graph']

    # Ensure save_dir exists
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Determine save path
    image_name = f"{graph_info[-1]}.html"
    image_path = os.path.join(save_dir, image_name) if save_dir else image_name

    # Create the plot
    fig, ax = plt.subplots()
    x, y = data.T
    scatter_params = {'x': x, 'y': y, 's': 1}
    if colors is not None:
        scatter_params.update({'c': colors, 'cmap': 'jet'})
        sc = ax.scatter(**scatter_params)
        plt.colorbar(sc, ax=ax)
    else:
        ax.scatter(**scatter_params)

    # Add labels and title
    ax.set_xlabel(graph_info[0])
    ax.set_ylabel(graph_info[1])
    ax.set_title(graph_info[-1])

    # Save the plot
    mpld3.save_html(fig, image_path)
    plt.close(fig)  # Close the figure to free memory


def ensure_directory_exists(directory):
    if directory is not None and not os.path.exists(directory):
        os.makedirs(directory)