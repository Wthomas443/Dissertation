import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import Spherical_harmonic_functions as sph

r = 6378000
nRowsRead = None
df1 = pd.read_csv('c:/Users/wt057/Documents/University/Dissertation/Data/GlobalLandTemperaturesByMajorCity.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'GlobalLandTemperaturesByMajorCity.csv'


nRow, nCol = df1.shape

# Filter the dataframe for rows where 'dt' is '2003-07-01'

filtered_df = df1[df1['dt'] == '2003-07-01']

theta = filtered_df['Latitude']
phi = filtered_df['Longitude']
data = np.array(filtered_df['AverageTemperature'])


# Convert latitude and longitude to radians
def convert_latitude(lat):
    if lat[-1] == 'N':
        return (90 - float(lat[:-1])) * np.pi / 180
    elif lat[-1] == 'S':
        return (90 + float(lat[:-1])) * np.pi / 180
    else:
        raise ValueError("Invalid latitude format")

theta = np.array([convert_latitude(lat) for lat in theta])

def convert_longitude(long):
    if long[-1] == 'W':
        return (360 - float(long[:-1])) * np.pi / 180
    elif long[-1] == 'E':
        return (float(long[:-1])) * np.pi / 180
    else:
        raise ValueError("Invalid latitude format")

phi = np.array([convert_longitude(long) for long in phi])

# Set max degree, regularisation parameter and what norm is penalised
n = len(data)
max_degree = 9
regularization_parameter = 0.004977023564332114
grad = 2

#Create plot to find optimum regularisation parameter
# num_folds = 10
# fold_error = np.zeros((num_folds,100))
# for i in range(num_folds):
#     np.random.seed(i)
#     test_idx = np.random.randint(0, n, int(0.2*n))
#     test_theta = theta[test_idx]
#     test_phi = phi[test_idx]
#     test_data = data[test_idx]
#     training_data = np.delete(data, test_idx)
#     training_theta = np.delete(theta, test_idx)
#     training_phi = np.delete(phi, test_idx)
#     A_training = sph.design_matrix_vectorized(training_theta, training_phi, max_degree)
#     A_test = sph.design_matrix_vectorized(test_theta, test_phi, max_degree)
#     reg_param = []
#     for idx, e in enumerate(np.logspace(-12,0,100)):
#         reg_param.append(e)
#         fitted_coefficients = sph.Solve_LSQ(max_degree, training_data, A_training, e, grad)
#         error = np.linalg.norm(test_data- A_test @ fitted_coefficients)
#         fold_error[i][idx] = error
#     plt.plot(np.log10(reg_param), np.log10(fold_error[i]), label=f'Fold {i+1}')
#     plt.xlabel('Regularization parameter $Log_{10}(\lambda)$', fontsize=16)
#     plt.ylabel('Test error $Log_{10}||f_{test}-A_{test}v||$', fontsize=16)
#     plt.grid(True)
# mean_error = (fold_error.mean(axis=0))
# min_error = np.min(mean_error)
# best_e = reg_param[np.argmin(mean_error)]
# plt.plot(np.log10(reg_param), np.log10(mean_error), label='Mean error', color='black', linewidth=3)
# plt.legend()
# plt.show()

# print(f'The optimal value of e is {best_e} with a test error of {min_error}')

# Calculate test error over 10 folds
max_degrees_extended = np.arange(0, 41, 4)
num_folds = 1
test_errors_extended = np.zeros((len(max_degrees_extended), num_folds))

for fold in range(num_folds):
    np.random.seed(fold)
    test_idx = np.random.randint(0, n, int(0.2*n))
    test_theta = theta[test_idx]
    test_phi = phi[test_idx]
    test_data = data[test_idx]
    training_data = np.delete(data, test_idx)
    training_theta = np.delete(theta, test_idx)
    training_phi = np.delete(phi, test_idx)
    
    for i, L in enumerate(max_degrees_extended):
        A_training = sph.design_matrix_vectorized(training_theta, training_phi, L)
        A_test = sph.design_matrix_vectorized(test_theta, test_phi, L)
        fitted_coefficients = sph.Solve_LSQ(L, training_data, A_training, regularization_parameter, grad)
        test_error = np.linalg.norm(test_data - A_test @ fitted_coefficients)
        test_errors_extended[i, fold] = test_error

# Calculate mean test error across folds
mean_test_errors_extended = test_errors_extended.mean(axis=1)

# Plot the test error 
plt.plot(max_degrees_extended, np.log10(mean_test_errors_extended))
plt.xlabel('Maximum degree', fontsize=16)
plt.ylabel('Test error $Log_{10}||f_{test}-A_{test}v||$', fontsize=16)
plt.grid(True)
plt.show()

# max_degrees = np.arange(0, 17, 1)
# reg_params = np.logspace(-12, 1, 20)
# training_errors = np.zeros((len(max_degrees), len(reg_params)))

# for i, L in enumerate(max_degrees):
#     for j, e in enumerate(reg_params):
#         A_training = sph.design_matrix_vectorized(training_theta, training_phi, L)
#         fitted_coefficients = sph.Solve_LSQ(L, training_data, A_training, e, grad)
#         training_error = np.linalg.norm(training_data - A_training @ fitted_coefficients)
#         training_errors[i, j] = training_error

# # Create a 3D plot for training errors
# fig = plt.figure(figsize=(14, 6))
# ax = fig.add_subplot(111, projection='3d')
# X, Y = np.meshgrid(np.log10(reg_params), max_degrees)
# ax.plot_surface(X, Y, np.log10(training_errors), cmap='viridis')
# ax.set_xlabel('Log10(Regularization parameter)')
# ax.set_ylabel('Maximum degree')
# ax.set_zlabel('Log10(Training error)')
# plt.show()

# Calculate the training and test errors for different maximum degrees and regularization parameters

# num_folds = 10
# test_errors = np.zeros((len(max_degrees), len(reg_params), num_folds))

# for fold in range(num_folds):
#     np.random.seed(fold)
#     test_idx = np.random.randint(0, n, int(0.2*n))
#     test_theta = theta[test_idx]
#     test_phi = phi[test_idx]
#     test_data = data[test_idx]
#     training_data = np.delete(data, test_idx)
#     training_theta = np.delete(theta, test_idx)
#     training_phi = np.delete(phi, test_idx)
    
#     for i, L in enumerate(max_degrees):
#         for j, e in enumerate(reg_params):
#             print(f"Fold {fold+1}, Degree {L}, Regularization {e}")
#             A_test = sph.design_matrix_vectorized(test_theta, test_phi, L)
#             A_training = sph.design_matrix_vectorized(training_theta, training_phi, L)
#             fitted_coefficients = sph.Solve_LSQ(L, training_data, A_training, e, grad)
#             test_error = np.linalg.norm(test_data - A_test @ fitted_coefficients)
#             test_errors[i, j, fold] = test_error

# # Find the minimum test error and corresponding parameter values
# min_error_idx = np.unravel_index(np.argmin(test_errors), test_errors.shape)
# min_error = test_errors[min_error_idx]
# best_degree = max_degrees[min_error_idx[0]]
# best_reg_param = reg_params[min_error_idx[1]]

# print(f"The minimum test error is {min_error} with degree {best_degree} and regularization parameter {best_reg_param}")

# mean_test_errors = test_errors.mean(axis=2)
# fig = plt.figure(figsize=(14, 6))
# X, Y = np.meshgrid(np.log10(reg_params), max_degrees)
# # Create a 3D plot for mean test errors
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, np.log10(mean_test_errors), cmap='viridis')
# ax.set_xlabel('Regularization parameter')
# ax.set_ylabel('L')
# ax.set_zlabel('Log10(Mean Test error)')

# plt.show()


#Split the data into training and test sets to fit the model on

np.random.seed(0)
test_idx = np.random.randint(0, n, int(0.2*n))
test_theta = theta[test_idx]
test_phi = phi[test_idx]
test_data = data[test_idx]
training_data = np.delete(data, test_idx)
training_theta = np.delete(theta, test_idx)
training_phi = np.delete(phi, test_idx)
A_training = sph.design_matrix_vectorized(training_theta, training_phi, max_degree)
A_test = sph.design_matrix_vectorized(test_theta, test_phi, max_degree)

#Plot the data and the model on a sphere
num_plot_points = 300
theta_grid, phi_grid = np.meshgrid(
    np.linspace(0, np.pi, num_plot_points),
    np.linspace(0, 2*np.pi, num_plot_points)
)

# Fit the model on the training data
A = sph.design_matrix_vectorized(theta, phi, max_degree)
coefficients = sph.Solve_LSQ(max_degree, data, A, regularization_parameter, grad)
residual = np.linalg.norm(data - A @ coefficients)
print(f"Test error: {np.sum(np.abs(test_data - A_test @ coefficients)**2)/len(test_data)}")
print(f"Training error: {np.sum(np.abs(training_data - A_training @ coefficients)**2)/len(training_data)}")

# Compute the fitted values on the grid
A_grid = sph.design_matrix_vectorized(theta_grid.flatten(), phi_grid.flatten(), max_degree)
fitted_grid = (A_grid @ coefficients).reshape(phi_grid.shape)


# Convert spherical coordinates to Cartesian coordinates for plotting
x_grid = r * np.sin(theta_grid) * np.cos(phi_grid)
y_grid = r * np.sin(theta_grid) * np.sin(phi_grid)
z_grid = r * np.cos(theta_grid)

# Normalize the grids and data to [0, 1] for color mapping
fitted_grid_normalized = (fitted_grid.real - fitted_grid.real.min()) / (fitted_grid.real.max() - fitted_grid.real.min())
data_normalized = (data.real - data.real.min()) / (data.real.max() - data.real.min())

# Plot the original and fitted data on the sphere
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
ax1.scatter(r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta), c = data_normalized, cmap='viridis')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

mappable = plt.cm.ScalarMappable(cmap='viridis')
mappable.set_array(data)
fig.colorbar(mappable, ax=ax1, shrink=0.5, aspect=7)

fig = plt.figure()
ax2 = fig.add_subplot(111, projection='3d')
ax2.plot_surface(x_grid, y_grid, z_grid, facecolors=plt.cm.viridis(fitted_grid_normalized), rstride=1, cstride=1, shade=False)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

mappable = plt.cm.ScalarMappable(cmap='viridis')
mappable.set_array(fitted_grid.real)
fig.colorbar(mappable, ax=ax2, shrink=0.5, aspect=7)

plt.show()

#Calculate the errors to plot an L-curve
errors = []
norm = []
for e in np.logspace(-6,-0,50):
    fitted_coefficients = sph.Solve_LSQ(max_degree, training_data, A_training, e, grad)
    error = np.linalg.norm(training_data- A_training @ fitted_coefficients)
    errors.append(error)
    norm.append(np.linalg.norm(sph.construct_L(max_degree, grad) @ fitted_coefficients))

#Plot the 'L-curve'

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(111)
edge_colors = ['blue', 'green', 'red', 'purple', 'magenta', 'orange']
ax1.plot(np.log10(errors), np.log10(norm))

e_values = [10e-6, 10e-5, 10e-4, 10e-3, 10e-2]
for idx, e in enumerate(e_values):
    fitted_coefficients = sph.Solve_LSQ(max_degree, training_data, A_training, e, grad)
    error = np.linalg.norm(training_data - A_training @ fitted_coefficients)
    norm = np.linalg.norm(sph.construct_L(max_degree, grad) @ fitted_coefficients)
    ax1.scatter(np.log10(error), np.log10(norm), label=f'$\lambda$={e_values[idx]}', edgecolors=edge_colors[idx], facecolors='none', marker='o', s=100)
ax1.set_xlabel('Residual norm $Log_{10}||Sf - f||$', fontsize=18)
ax1.set_ylabel('Coefficients norm $Log_{10}||v||$', fontsize=18)
ax1.legend(fontsize=16)
ax1.grid(True)
plt.show()
