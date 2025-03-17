import numpy as np
import matplotlib.pyplot as plt
import Spherical_harmonic_functions as sph
from matplotlib.ticker import MaxNLocator

#number of data points
num_points = 600
theta = np.random.uniform(0, np.pi, num_points)
phi = np.random.uniform(0, 2*np.pi, num_points)

#level of complexity of simulation function and approximation
function_max_degree = 16
max_degree = 16

#regularisation parameter and norm penalised 
regularization_parameter = 0
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
num_plot_points = 50
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


#Explore the change in error for the number of data points and the complexity of the function being approximated

max_degree_list = [6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58]
# fig, ax = plt.subplots(figsize=(10, 6))
# index = []
# min_errors = []
# min_points = [ 49,  129,  205,  378,  565,  831, 1091, 1429, 1840, 2284, 2744, 3120, 3584, 4096]
# num_points_matrix = np.zeros((len(max_degree_list), 30))
# print(min_points)
# for j, new_max_degree in enumerate(max_degree_list):
#     num_points_list = np.logspace(np.log10(min_points[j] - 1/5 *min_points[j]), np.log10(min_points[j] + 1/5 * min_points[j]), 30).astype(int)
#     errors = np.zeros((len(max_degree_list), len(num_points_list)))
#     num_points_matrix[j, :] = num_points_list
#     for i, points in enumerate(num_points_list):
#         np.random.seed(6)
#         print(f"Max Degree: {new_max_degree}, Points: {points}")
#         new_theta = np.random.uniform(0, np.pi, points)
#         new_phi = np.random.uniform(0, 2*np.pi, points)
#         new_true_coefficients =  (np.random.randn((new_max_degree + 1) ** 2) + 1j * np.random.randn((new_max_degree + 1) ** 2)) + 0.4
#         new_true_function = sph.generate_sample_data_vectorized(new_theta, new_phi, new_max_degree, new_true_coefficients)
#         new_data = new_true_function + 0.1 * np.random.randn(points)
#         new_A = sph.design_matrix_vectorized(new_theta, new_phi, new_max_degree)
#         coefficients = sph.Solve_LSQ(new_max_degree, new_data, new_A, regularization_parameter, grad)
#         error = np.linalg.norm(new_data - new_A @ coefficients)
#         errors[j, i] = error
#     min_error = np.min(errors[j, :])
#     min_errors.append(min_error)
#     index.append(np.argmin(errors[j, :]))
#     ax.loglog(num_points_list, errors[j, :], label=f'Max Degree: {max_degree}')
    
# ax.set_xlabel('Number of Data Points', fontsize=14)
# ax.set_ylabel('Minimum error', fontsize=14)
# ax.legend(fontsize=12)
# ax.grid(True)

# plt.show()


# print(num_points_matrix[range(0, len(max_degree_list)), index])
# points = num_points_matrix[range(0, len(max_degree_list)), index]

fig, ax = plt.subplots(figsize=(10, 6))
points = [49, 122, 229, 378, 565, 831, 1106, 1429, 1841, 2253, 2745, 3210, 3845, 4456]

ax.plot(np.log10(points), max_degree_list)
ax.set_xlabel('Number of Data Points', fontsize=18)
ax.set_ylabel('Max Degree', fontsize=18)
ax.grid(True)

plt.show()

fig, ax = plt.subplots(figsize=(10, 6))


ax.plot(points, max_degree_list)
ax.set_xlabel('Number of Data Points', fontsize=18)
ax.set_ylabel('Max Degree', fontsize=18)
ax.grid(True)

plt.show()

fig, ax = plt.subplots(figsize=(10, 6))

ax.loglog(points, max_degree_list)
ax.set_xlabel('Number of Data Points', fontsize=18)
ax.set_ylabel('Max Degree', fontsize=18)
ax.grid(True)

plt.show()

errors_e = []
e_solutions = []

for i in np.logspace(-10, -6, 50):
        coefficients = sph.Solve_LSQ(max_degree, data, A, i, grad)
        e_solutions.append(np.linalg.norm(sph.construct_L(max_degree, grad) @ coefficients))
        errors_e.append(np.linalg.norm(true_function - A @ coefficients))


fig = plt.figure(figsize=(14, 6))
ax2 = fig.add_subplot(111)
ax2.plot(np.log10(errors_e), np.log10(e_solutions), label='Error vs Solution Norm')

#[ 10e-9, 10e-8, 10e-7, 10e-6, 10e-5, 10e-4, 10e-3, 10e-2]
# Add points for specific regression parameters
specific_params = [1e-8, 1e-7, 1e-6]
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