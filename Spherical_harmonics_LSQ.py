import numpy as np
import matplotlib.pyplot as plt
import Spherical_harmonic_functions as sph
from matplotlib.ticker import MaxNLocator

#number of data points
num_points = 400
theta = np.random.uniform(0, np.pi, num_points)
phi = np.random.uniform(0, 2*np.pi, num_points)

#level of complexity of simulation function and approximation
function_max_degree = 16
max_degree = 16

#regularisation parameter and norm penalised 
regularization_parameter = 1e-6
grad = 2

#set random seed for reproducibility
np.random.seed(6)
# Generate the true coefficients
true_coefficients =  (np.random.randn((function_max_degree + 1) ** 2) + 1j * np.random.randn((function_max_degree + 1) ** 2)) + 0.4

# Generate the true function and add noise
true_function = sph.generate_sample_data_vectorized(theta, phi, function_max_degree, true_coefficients)
data = true_function + 0.1 * np.random.randn(num_points)

# Generate the design matrix and solve the least squares problem
A = sph.design_matrix_vectorized(theta, phi, max_degree)
coefficients = sph.Solve_LSQ(max_degree, data, A, regularization_parameter, grad)

# Print the fitted coefficients
print("Fitted coefficients:") 

# Compute the error
error = np.linalg.norm(true_function - A @ coefficients)
print(coefficients)
print(f"Error: {error}")

# Generate a grid of points on the sphere for visualization
num_plot_points = 250
theta_grid, phi_grid = np.meshgrid(
    np.linspace(0, np.pi, num_plot_points),
    np.linspace(0, 2*np.pi, num_plot_points)
)

# Compute the fitted values on the grid
A_grid = sph.design_matrix_vectorized(theta_grid.flatten(), phi_grid.flatten(), max_degree)
fitted_grid = np.dot(A_grid, coefficients).reshape(phi_grid.shape)
function_grid = sph.generate_sample_data_vectorized(theta_grid.flatten(), phi_grid.flatten(), function_max_degree, true_coefficients).reshape(theta_grid.shape)

# Normalize the grids and data to [0, 1] for color mapping
function_grid_normalized = (function_grid.real - function_grid.real.min()) / (function_grid.real.max() - function_grid.real.min())
data_normalized = (data.real - data.real.min()) / (data.real.max() - data.real.min())

# Convert spherical coordinates to Cartesian coordinates for plotting
x_grid = np.sin(theta_grid) * np.cos(phi_grid)
y_grid = np.sin(theta_grid) * np.sin(phi_grid)
z_grid = np.cos(theta_grid)

# Plot the original and fitted data on the sphere
fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta), c = data_normalized, cmap='viridis')
ax1.set_xlabel('X', fontsize=16)
ax1.set_ylabel('Y', fontsize=16)
ax1.set_zlabel('Z', fontsize=16)

# Real function
ax3 = fig.add_subplot(122, projection='3d')
surf = ax3.plot_surface(x_grid, y_grid, z_grid, facecolors = plt.cm.viridis(function_grid_normalized), rstride=1, cstride=1, shade=False)
ax3.set_xlabel('X', fontsize=16)
ax3.set_ylabel('Y', fontsize=16)
ax3.set_zlabel('Z', fontsize=16)

# Add color bar which maps values to colors
mappable = plt.cm.ScalarMappable(cmap='viridis')
mappable.set_array(function_grid.real)
fig.colorbar(mappable, ax=ax3, shrink=0.5, aspect=7)
plt.show()

sph.plot_spherical_harmonics(x_grid, y_grid, z_grid, fitted_grid.real)
plt.show()

# Plot the error for varying max_degree values
max_degrees = range(function_max_degree-3, function_max_degree+15)
errors_degree = []
errors_e = []
e_solutions = []
for j in max_degrees:
    A_new = sph.design_matrix_vectorized(theta, phi, j)
    coefficients = sph.Solve_LSQ(j, data, A_new, regularization_parameter, grad)
    error_degree = np.linalg.norm(true_function - A_new @ coefficients)
    errors_degree.append(error_degree)

for i in np.logspace(-12, -4, 50):
        coefficients = sph.Solve_LSQ(max_degree, data, A, i, grad)
        e_solutions.append(np.linalg.norm(sph.construct_L(max_degree, grad) @ coefficients))
        errors_e.append(np.linalg.norm(true_function - A @ coefficients))

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(111)
ax1.plot(max_degrees, np.log10(errors_degree), scalex=int)
ax1.set_xlabel('L value', fontsize=18)
ax1.set_ylabel('$Log_{10}||Sf - f||_2$', fontsize=18)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.grid(True)
plt.show()
fig = plt.figure(figsize=(14, 6))
ax2 = fig.add_subplot(111)
ax2.plot(np.log10(errors_e), np.log10(e_solutions), label='Error vs Solution Norm')

#[ 10e-9, 10e-8, 10e-7, 10e-6, 10e-5, 10e-4, 10e-3, 10e-2]
# Add points for specific regression parameters
specific_params = [10e-10, 10e-9, 10e-8, 10e-7,  10e-6, 10e-5 ]
edge_colors = ['blue', 'green', 'red', 'purple', 'magenta', 'orange']
for idx, param in enumerate(specific_params):
    coefficients = sph.Solve_LSQ(max_degree, data, A, param, grad)
    error_e = np.linalg.norm(true_function - A @ coefficients)
    solution_norm = np.linalg.norm(sph.construct_L(max_degree, grad) @ coefficients)
    ax2.scatter(np.log10(error_e), np.log10(solution_norm), label=f'$\lambda$={param}', edgecolor=edge_colors[idx % len(edge_colors)], facecolor='none', marker='o', s=100, linewidth=1.8)
ax2.set_xlabel('Residual norm $Log_{10}||Sf - f||_2$', fontsize=18)
ax2.set_ylabel('Coefficients norm $Log_{10}||v||_2$', fontsize=18)
ax2.legend(fontsize=16)
ax2.grid(True)

plt.show()