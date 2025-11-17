import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from meteostat import Stations, Hourly
import Vector_spherical_harmonics as vsh
import Wind_data as wd


def split_train_test(data, theta, phi, test_ratio=0.2, seed=0):
    n = len(theta)
    np.random.seed(seed)
    test_idx = np.random.choice(n, int(test_ratio * n), replace=False)
    train_idx = np.setdiff1d(np.arange(n), test_idx)

    theta_train = theta[train_idx]
    phi_train = phi[train_idx]
    data_train = np.concatenate([data[0:n][train_idx], data[n:2*n][train_idx]])

    theta_test = theta[test_idx]
    phi_test = phi[test_idx]
    data_test = np.concatenate([data[0:n][test_idx], data[n:2*n][test_idx]])

    return data_train, theta_train, phi_train, data_test, theta_test, phi_test

def optimize_regularization_parameter(num_folds, num_samples, num_components, max_degree, grad, data, theta, phi):
    fig = plt.figure(figsize=(12, 6))
    fold_error = np.zeros((num_folds, num_samples))
    reg_param = np.logspace(-6, 2, num_samples)
    for i in range(num_folds):
        print(f'Fold {i+1}')
        
        # Split the data into training and testing sets
        data_train, theta_train, phi_train, data_test, theta_test, phi_test = split_train_test(data, theta, phi, test_ratio=0.2, seed=i)

        # Split the design matrix accordingly
        A_train = vsh.VectorDesignMatrix(theta_train, phi_train, max_degree)
        A_test = vsh.VectorDesignMatrix(theta_test, phi_test, max_degree)


        for idx, e in enumerate(reg_param):
            print(f'Lambda: {e}')

            fitted_coefficients = vsh.Solve_LSQ(max_degree, data_train, A_train, e, grad)

            error = np.linalg.norm(data_test - A_test @ fitted_coefficients)
            fold_error[i][idx] = error
            del fitted_coefficients

        del A_train, A_test

        plt.plot(np.log10(reg_param), np.log10(fold_error[i]), label=f'Fold {i+1}')
        plt.xlabel('Regularization parameter $Log_{10}(\lambda)$', fontsize=16)
        plt.ylabel('Test error $Log_{10}||f_{test}-A_{test}v||$', fontsize=16)
        plt.grid(True)
    mean_error = (fold_error.mean(axis=0))
    min_error = np.min(mean_error)
    best_e = reg_param[np.argmin(mean_error)]
    plt.plot(np.log10(reg_param), np.log10(mean_error), label='Mean error', color='black', linewidth=3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'best_lambda_{num_components}.png')
    plt.close()
    print(f'The optimal value of e is {best_e} with a test error of {min_error}')

    return best_e

def plot_wind_data(theta_train, phi_train, data_train, n_components, r):

    x =   r* np.sin(theta_train) * np.cos(phi_train)
    y =   r* np.sin(theta_train) * np.sin(phi_train)
    z =   r* np.cos(theta_train)

    data_plot = vsh.convert_to_cartesian(theta_train, phi_train, data_train)

    # Use the real parts and extract components
    u = data_plot[:, 0].real
    v = data_plot[:, 1].real
    w = data_plot[:, 2].real

    # Compute per-vector norms and scale vectors so arrows are visible on the sphere
    norms = np.sqrt(u**2 + v**2 + w**2)
    max_norm = np.nanmax(norms)
    if max_norm == 0 or np.isnan(max_norm):
        scale = 1.0
    else:
        # choose a target maximum arrow length as a fraction of the sphere radius
        target_max_length = 0.4 * r
        scale = target_max_length / max_norm

    u_s = u * scale
    v_s = v * scale
    w_s = w * scale

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    # length multiplies the supplied vectors; we scaled vectors to desired absolute size, so use length=1
    ax.quiver(x, y, z, u_s, v_s, w_s, length=1.0, arrow_length_ratio=0.2, linewidth=0.6)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    fig.tight_layout()
    fig.savefig(f'wind_data_plot_{n_components}.png')
    plt.close()

def plot_wind_data_scalar(r, theta_train, phi_train, data_train, n_components):

    x =   r* np.sin(theta_train) * np.cos(phi_train)
    y =   r* np.sin(theta_train) * np.sin(phi_train)
    z =   r* np.cos(theta_train)

    #split data_train and normalise to plot scalar fields
    data_train_phi = data_train[0:len(theta_train)]
    data_train_theta = data_train[len(theta_train):]

    data_train_phi_n = (data_train_phi.real - data_train_phi.real.min()) / (data_train_phi.real.max() - data_train_phi.real.min())
    data_train_theta_n = (data_train_theta.real - data_train_theta.real.min()) / (data_train_theta.real.max() - data_train_theta.real.min())
    
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(x, y, z, c = data_train_phi_n, cmap='viridis')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    mappable = plt.cm.ScalarMappable(cmap='viridis')
    mappable.set_array(data_train_phi.real)
    fig.colorbar(mappable, ax=ax1, shrink=0.5, aspect=5)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(x, y, z, c = data_train_theta_n, cmap='viridis')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    mappable = plt.cm.ScalarMappable(cmap='viridis')
    mappable.set_array(data_train_theta.real)
    fig.colorbar(mappable, ax=ax2, shrink=0.5, aspect=5)
    fig.tight_layout()
    fig.savefig(f'scalar_data_components_{n_components}.png')
    plt.close()

def plot_fitted_data(fitted_grid, theta_grid, phi_grid, r, n_components):

    # Compute the fitted values on the grid
    x_grid = r * np.sin(theta_grid) * np.cos(phi_grid)
    y_grid = r * np.sin(theta_grid) * np.sin(phi_grid)
    z_grid = r * np.cos(theta_grid)

    fitted_grid_plot = vsh.convert_to_cartesian(theta_grid.flatten(), phi_grid.flatten(), fitted_grid)

    # Use the real parts and extract components
    u = fitted_grid_plot[:, 0].real
    v = fitted_grid_plot[:, 1].real
    w = fitted_grid_plot[:, 2].real

    # Compute per-vector norms and scale vectors so arrows are visible on the sphere
    norms = np.sqrt(u**2 + v**2 + w**2)
    max_norm = np.nanmax(norms)
    if max_norm == 0 or np.isnan(max_norm):
        scale = 1.0
    else:
        # choose a target maximum arrow length as a fraction of the sphere radius
        target_max_length = 0.4 * r
        scale = target_max_length / max_norm

    u_s = u * scale
    v_s = v * scale
    w_s = w * scale

    # Plot the fitted data
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(
        x_grid.flatten(), y_grid.flatten(), z_grid.flatten(),
        u_s, v_s, w_s,
        length=1.0, arrow_length_ratio=0.2, linewidth=0.6
    )
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    fig.tight_layout()
    fig.savefig(f'fitted_wind_data_plot_{n_components}.png')
    plt.close()

def plot_fitted_data_scalar(theta_grid_scalar, phi_grid_scalar, r, max_degree, coefficients, n_components):

    # Compute the fitted values on the grid
    A_grid_scalar = vsh.VectorDesignMatrix(theta_grid_scalar.flatten(), phi_grid_scalar.flatten(), max_degree)
    fitted_grid_scalar = A_grid_scalar @ coefficients

    x_grid_scalar, y_grid_scalar, z_grid_scalar = r * np.sin(theta_grid_scalar) * np.cos(phi_grid_scalar), r* np.sin(theta_grid_scalar) * np.sin(phi_grid_scalar),r* np.cos(theta_grid_scalar)

# Reshape and normalize the components
    phi_component = fitted_grid_scalar[:len(theta_grid_scalar.flatten())].real.reshape(theta_grid_scalar.shape)
    theta_component = fitted_grid_scalar[len(theta_grid_scalar.flatten()):].real.reshape(theta_grid_scalar.shape)
    phi_component_n = (phi_component - phi_component.min()) / (phi_component.max() - phi_component.min())
    theta_component_n = (theta_component - theta_component.min()) / (theta_component.max() - theta_component.min())

    # Plot the fitted data
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(x_grid_scalar, y_grid_scalar, z_grid_scalar, c =phi_component_n)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    mappable = plt.cm.ScalarMappable(cmap='viridis')
    mappable.set_array(phi_component)
    fig.colorbar(mappable, ax=ax1, shrink=0.5, aspect=5)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(x_grid_scalar, y_grid_scalar, z_grid_scalar, c = theta_component_n, cmap='viridis')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    mappable = plt.cm.ScalarMappable(cmap='viridis')
    mappable.set_array(theta_component)
    fig.colorbar(mappable, ax=ax2, shrink=0.5, aspect=5)
    fig.tight_layout()
    fig.savefig(f'fitted_scalar_data_components_{n_components}.png')
    plt.close()

def streamline_plot(fitted_data, num_components, num_plot_points, r=1):
    """
    Create a streamline plot from the fitted data.
    Parameters: fitted_data: The fitted data to be plotted shape should be shape (2 * num_plot_points^2,) organised as (phi_components : theta_components)
    num_plot_points: The number of points to plot in each direction
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
    fig = plt.figure(figsize=(12, 6))
    res = plt.streamplot(
        theta_sorted, phi_sorted,
        v_component, u_component, density= 1.8
    )
    plt.xlabel('Longitude (radians)')
    plt.ylabel('Latitude (radians)')
    plt.title('2D Stream Plot of Theta and Phi Components')
    plt.grid(True)
    plt.savefig(f'streamplot_2D_{num_components}.png')
    plt.close()
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
    plt.savefig(f'streamplot_3D_{num_components}.png')
    plt.close()


# Generate a grid of points on the sphere for visualization of vector field
num_plot_points = 50
theta_grid, phi_grid = np.meshgrid(
    np.linspace(0.01, np.pi-0.01, num_plot_points),
    np.linspace(0, 2*np.pi, num_plot_points)
)

# Generate a grid of points on the sphere for visualization of scalar fields
num_plot_points_scalar = 150
theta_grid_scalar, phi_grid_scalar = np.meshgrid(
    np.linspace(0.01, np.pi-0.01, num_plot_points_scalar),
    np.linspace(0, 2*np.pi, num_plot_points_scalar)
)

max_degree = 30
grad = 2
r = 6378000  # Earth radius in meters

num_components = [1, 2, 3, 4]

for n_components in num_components:
    #Load the PCA reduced data
    data, theta, phi = np.load(f'wind_data_pca.npz')['data'][:,n_components-1], np.load(f'wind_data_pca.npz')['theta'], np.load(f'wind_data_pca.npz')['phi']

    n = len(theta)
    print('Find best regularization parameter...')
    #optimize regularization parameter
    regularization_parameter = optimize_regularization_parameter(10, 100, n_components, max_degree, grad, data, theta, phi)

    print('Find coefficients...')
    # Split the data into training and testing sets
    data_train, theta_train, phi_train, data_test, theta_test, phi_test = split_train_test(data, theta, phi, test_ratio=0.2, seed=42)

    # Split the design matrix
    A_train = vsh.VectorDesignMatrix(theta_train, phi_train, max_degree)
    A_test = vsh.VectorDesignMatrix(theta_test, phi_test, max_degree)

    # Solve for the coefficients using the optimal regularization parameter
    coefficients = vsh.Solve_LSQ(max_degree, data_train, A_train, regularization_parameter, grad)

    print('Plotting data...')
    # Plot the wind data

    plot_wind_data(theta_train, phi_train, data_train, n_components, r)

    plot_wind_data_scalar(r, theta_train, phi_train, data_train, n_components)

    print('Fitting model and plotting results...')
    #Fit the model and plot results
    A_grid = vsh.VectorDesignMatrix(theta_grid.flatten(), phi_grid.flatten(), max_degree)
    fitted_grid = A_grid @ coefficients

    plot_fitted_data(fitted_grid, theta_grid, phi_grid, r, n_components)

    plot_fitted_data_scalar(theta_grid_scalar, phi_grid_scalar, r, max_degree, coefficients, n_components)  

    print('Creating streamline plot...')
    #Create streamline plot
    streamline_plot(fitted_grid, n_components, num_plot_points, r)

    train_errors = []
    test_errors = []

    test_error = np.mean((data_test - A_test @ coefficients)**2)
    train_error = np.mean((data_train - A_train @ coefficients)**2)

    train_errors.append(train_error)
    test_errors.append(test_error)

fig = plt.figure(figsize=(8, 6))
plt.plot(num_components, train_errors.real, marker='o', label='Train Error')
plt.plot(num_components, test_errors.real, marker='o', label='Test Error')
plt.xlabel('Number of PCA Components', fontsize=16)
plt.ylabel('Mean Squared Error', fontsize=16)
plt.legend()
plt.grid(True)
fig.tight_layout()
fig.savefig('train_test_errors.png')
plt.show()












