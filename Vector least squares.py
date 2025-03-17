import Vector_spherical_harmonics as vsh
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

#number of data points
num_points = 400
theta = np.random.uniform(0.01, np.pi-0.01, num_points)
phi = np.random.uniform(0, 2*np.pi, num_points)
r=1

#level of complexity of simulation function and approximation
function_max_degree = 10
max_degree = 10


#regularisation parameter and norm penalised 
regularization_parameter = 1e-08
grad = 2

#set random seed for reproducibility
np.random.seed(5)
# Generate the true coefficients
true_coefficients =  (np.random.randn((function_max_degree + 1) ** 2) + 1j * np.random.randn((function_max_degree + 1) ** 2)) 

# Generate the true function and add noise
true_function = vsh.VectorSampleData(theta, phi, function_max_degree, true_coefficients)
data = true_function + 0 * np.random.randn(2*num_points)
import matplotlib.pyplot as plt

# Generate a grid of points on the sphere for visualization of vector field
num_plot_points = 30
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

function_grid = vsh.VectorSampleData(theta_grid.flatten(), phi_grid.flatten(), function_max_degree, true_coefficients)
fig = plt.figure(figsize=(12, 6))

function_grid_plot = vsh.convert_to_cartesian(theta_grid.flatten(), phi_grid.flatten(), function_grid)
data_plot = vsh.convert_to_cartesian(theta, phi, data)


# Convert spherical to Cartesian coordinates for plotting
x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)

# Plot the noisy data
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title('Noisy Data')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_xlim([-1, 1])
ax1.quiver(x, y, z, data_plot[:,0].real, data_plot[:,1].real, data_plot[:,2].real, length=0.1, normalize=True)

# Convert grid spherical to Cartesian coordinates for plotting
x_grid = np.sin(theta_grid) * np.cos(phi_grid)
y_grid = np.sin(theta_grid) * np.sin(phi_grid)
z_grid = np.cos(theta_grid)

# Plot the true function
ax2 = fig.add_subplot(122, projection='3d')
ax2.quiver(x_grid.flatten(), y_grid.flatten(), z_grid.flatten(), function_grid_plot[:,0].real, function_grid_plot[:,1].real, function_grid_plot[:,2].real, length=0.1, normalize=True)
ax2.set_title('True Function')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')

plt.show()
#print(vsh.Tangent_check(num_points, x, y, z, vsh.convert_to_cartesian(data)))
#print(vsh.Tangent_check(num_plot_points**2, x_grid.flatten(), y_grid.flatten(), z_grid.flatten(), vsh.convert_to_cartesian(function_grid)))

A = vsh.VectorDesignMatrix(theta, phi, max_degree)
coefficients = vsh.Solve_LSQ(max_degree, data, A, regularization_parameter, grad)

A_grid = vsh.VectorDesignMatrix(theta_grid.flatten(), phi_grid.flatten(), max_degree)
function_grid_approx = A_grid @ coefficients

A_grid_scalar = vsh.VectorDesignMatrix(theta_grid_scalar.flatten(), phi_grid_scalar.flatten(), max_degree)
function_grid_approx_scalar = A_grid_scalar @ coefficients

phi_component = function_grid_approx_scalar[0:len(theta_grid_scalar.flatten())].real.reshape(theta_grid_scalar.shape)
theta_component = function_grid_approx_scalar[len(theta_grid_scalar.flatten()):].real.reshape(theta_grid_scalar.shape)
phi_component_n = (phi_component - phi_component.min())/(phi_component.max()-phi_component.min())
theta_component_n = (theta_component - theta_component.min())/(theta_component.max()-theta_component.min())
function_grid_approx_plot = vsh.convert_to_cartesian(theta_grid.flatten(), phi_grid.flatten(), function_grid_approx)

# Convert grid spherical to Cartesian coordinates for plotting theta and phi components
x_grid_scalar = np.sin(theta_grid_scalar) * np.cos(phi_grid_scalar)
y_grid_scalar = np.sin(theta_grid_scalar) * np.sin(phi_grid_scalar)
z_grid_scalar = np.cos(theta_grid_scalar)

data_phi_component = data[0:num_points].real
data_theta_component = data[num_points:].real
data_phi_component = (data_phi_component - data_phi_component.min())/(data_phi_component.max()-data_phi_component.min())
data_theta_component = (data_theta_component - data_theta_component.min())/(data_theta_component.max()-data_theta_component.min())

function_grid_scalar = vsh.VectorSampleData(theta_grid_scalar.flatten(), phi_grid_scalar.flatten(), function_max_degree, true_coefficients)
function_grid_scalar_phi = function_grid_scalar[0:len(theta_grid_scalar.flatten())].real.reshape(theta_grid_scalar.shape)
function_grid_scalar_theta = function_grid_scalar[len(theta_grid_scalar.flatten()):].real.reshape(theta_grid_scalar.shape)
function_grid_scalar_phi_n = (function_grid_scalar_phi - function_grid_scalar_phi.min())/(function_grid_scalar_phi.max()-function_grid_scalar_phi.min())
function_grid_scalar_theta_n = (function_grid_scalar_theta - function_grid_scalar_theta.min())/(function_grid_scalar_theta.max()-function_grid_scalar_theta.min())

# Plot the phi component of the data
fig = plt.figure(figsize=(12, 12))

# Plot the phi component of the data
ax = fig.add_subplot(221, projection='3d')
sc = ax.scatter(x, y, z, c=data_phi_component, cmap='viridis')
ax.set_title('Phi Component of Data')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
fig.colorbar(sc, ax=ax, shrink=0.5, aspect=5)

# Plot the theta component of the data
ax = fig.add_subplot(223, projection='3d')
sc = ax.scatter(x, y, z, c=data_theta_component, cmap='viridis')
ax.set_title('Theta Component of Data')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
fig.colorbar(sc, ax=ax, shrink=0.5, aspect=5)

# Plot the true function phi component
ax = fig.add_subplot(222, projection='3d')
sc = ax.plot_surface(x_grid_scalar, y_grid_scalar, z_grid_scalar, facecolors=plt.cm.viridis(function_grid_scalar_phi_n), rstride=1, cstride=1, shade=False)
ax.set_title('True Function Phi')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
mappable = plt.cm.ScalarMappable(cmap='viridis')
mappable.set_array(function_grid_scalar_phi)
fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)

# Plot the true function theta component
ax = fig.add_subplot(224, projection='3d')
sc = ax.plot_surface(x_grid_scalar, y_grid_scalar, z_grid_scalar, facecolors=plt.cm.viridis(function_grid_scalar_theta_n), rstride=1, cstride=1, shade=False)
ax.set_title('True Function Theta')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
mappable = plt.cm.ScalarMappable(cmap='viridis')
mappable.set_array(function_grid_scalar_theta)
fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)

plt.show()

# Plot the approximated function
fig = plt.figure(figsize=(12, 6))

# Plot the approximated function phi component
ax = fig.add_subplot(121, projection='3d')
sc = ax.plot_surface(x_grid_scalar, y_grid_scalar, z_grid_scalar, facecolors=plt.cm.viridis(phi_component_n), rstride=1, cstride=1, shade=False)
ax.set_title('Approximated Function Phi')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
mappable = plt.cm.ScalarMappable(cmap='viridis')
mappable.set_array(phi_component)
fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)

# Plot the approximated function theta component
ax = fig.add_subplot(122, projection='3d')
sc = ax.plot_surface(x_grid_scalar, y_grid_scalar, z_grid_scalar, facecolors=plt.cm.viridis(theta_component_n), rstride=1, cstride=1, shade=False)
ax.set_title('Approximated Function Theta')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
mappable = plt.cm.ScalarMappable(cmap='viridis')
mappable.set_array(theta_component)
fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)

plt.show()


fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')
ax.quiver(x_grid.flatten(), y_grid.flatten(), z_grid.flatten(), function_grid_approx_plot[:,0].real, function_grid_approx_plot[:,1].real, function_grid_approx_plot[:,2].real ,length=0.1, normalize=True)
ax.set_title('Approximated Function')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

# Calculate the error
error = np.linalg.norm(true_function - A @ coefficients)
print('Error:', error)

errors_e = []
e_solutions = []
max_degrees = np.arange(1, 20)
errors_max_degree = []
for degree in max_degrees:
    A_new = vsh.VectorDesignMatrix(theta, phi, degree)
    coefficients = vsh.Solve_LSQ(degree, data, A_new, regularization_parameter, grad)
    error = np.linalg.norm(true_function - A_new @ coefficients)
    errors_max_degree.append(error)

# Plot the error vs max degree
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.plot(max_degrees, np.log10(errors_max_degree), marker='o')
ax.set_xlabel('Max Degree', fontsize=14)
ax.set_ylabel('Error', fontsize=14)
ax.set_title('Error vs Max Degree', fontsize=16)
ax.grid(True)

plt.show()


fig = plt.figure(figsize=(14, 6))
ax2 = fig.add_subplot(111)
ax2.plot(np.log10(errors_e), np.log10(e_solutions), label='Error vs Solution Norm')

#[ 10e-9, 10e-8, 10e-7, 10e-6, 10e-5, 10e-4, 10e-3, 10e-2]
# Add points for specific regression parameters
specific_params = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
edge_colors = ['blue', 'green', 'red', 'purple', 'magenta', 'orange']
for idx, param in enumerate(specific_params):
    coefficients = vsh.Solve_LSQ(max_degree, data, A, param, grad)
    error_e = np.linalg.norm(true_function - A @ coefficients)
    solution_norm = np.linalg.norm(vsh.construct_L(max_degree, grad) @ coefficients)
    ax2.scatter(np.log10(error_e), np.log10(solution_norm), label=f'$\lambda$={param}', edgecolor=edge_colors[idx % len(edge_colors)], facecolor='none', marker='o', s=100, linewidth=1.8)
ax2.set_xlabel('Residual norm $Log_{10}||Sf - f||_2$', fontsize=18)
ax2.set_ylabel('Coefficients norm $Log_{10}||v||_2$', fontsize=18)
ax2.legend(fontsize=16)
ax2.grid(True)

plt.show()

