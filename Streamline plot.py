import numpy as np
import matplotlib.pyplot as plt

def streamline_plot(fitted_data, num_plot_points=100, r=1):
    """
    Create a streamline plot from the fitted data.
    Parameters: fitted_data: The fitted data to be plotted shape should be shape (2 * num_plot_points^2,) organised as (phi_components : theta_components)
    num_plot_points: The number of points to plot in each direction (default is 100)
    r: The radius of the sphere (default is 1)
    """
    #Get theta and phi ordered from 0 to 2*pi and 0 to pi respectively
    theta_sorted, phi_sorted = np.meshgrid(
    np.linspace(0.01, np.pi - 0.01, num_plot_points),
    np.linspace(0, 2 * np.pi, num_plot_points)
    )
    # Extract the fitted grid components for the streamplot
    u_component = fitted_data[:len(theta_sorted.flatten())].real.reshape(theta_sorted.shape)
    v_component = fitted_data[len(theta_sorted.flatten()):].real.reshape(theta_sorted.shape)

    # Create the streamline plot
    res = plt.streamplot(
        theta_sorted, phi_sorted,
        v_component, u_component, density= 1.8
    )
    plt.xlabel('Longitude (radians)')
    plt.ylabel('Latitude (radians)')
    plt.title('2D Stream Plot of Theta and Phi Components')
    plt.grid(True)
    plt.show()

    #extract the lines from the streamplot 
    lines = res.lines.get_paths()

    # Create a new figure for the lines
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(111, projection='3d')
    for line in lines:
        theta_line = line.vertices[:, 0]
        phi_line = line.vertices[:, 1]
        x_line, y_line, z_line = r * np.sin(theta_line) * np.cos(phi_line), r * np.sin(theta_line) * np.sin(phi_line), r * np.cos(theta_line)
        ax1.plot(x_line, y_line, z_line, color='tab:blue')
        
        # Compute the midpoint of the line
        mid_idx = len(x_line) // 2
        u = x_line[mid_idx + 1] - x_line[mid_idx]
        v = y_line[mid_idx + 1] - y_line[mid_idx]
        w = z_line[mid_idx + 1] - z_line[mid_idx]
        
        ax1.quiver(x_line[mid_idx], y_line[mid_idx], z_line[mid_idx], u, v, w, length=1.5, color='k')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    plt.show()