import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import Spherical_harmonic_functions as sph

r = 6371000
nRowsRead = None
df1 = pd.read_csv('c:/Users/wt057/Documents/University/Data/GlobalLandTemperaturesByCity.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'GlobalLandTemperaturesByCity.csv'


nRow, nCol = df1.shape

# Filter the dataframe for rows where 'dt' is '2003-07-01'

filtered_df = df1[df1['dt'] == '2003-07-01']

theta = filtered_df['Latitude']
phi = filtered_df['Longitude']
data = np.array(filtered_df['AverageTemperature'])
max_data = max(data)
#data = data/max_data

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

n = len(data)
max_degree = 30
regularization_parameter = 0.38
grad = 0
A = sph.design_matrix_vectorized(theta, phi, max_degree)
coefficients = sph.Solve_LSQ(max_degree, data, A, regularization_parameter, grad)

fold_error = np.zeros((10,100))
for i in range(1):
    np.random.seed(i)
    test_idx = np.random.randint(0, n, int(0.2*n))
    test_theta = theta[test_idx]
    test_phi = phi[test_idx]
    test_data = data[test_idx]
    training_data = np.delete(data, test_idx)
    training_theta = np.delete(theta, test_idx)
    training_phi = np.delete(phi, test_idx)
    A_training = sph.design_matrix_vectorized(training_theta, training_phi, max_degree)
    A_test = sph.design_matrix_vectorized(test_theta, test_phi, max_degree)
    reg_param = []
    for idx, e in enumerate(np.logspace(-6,2,100)):
        reg_param.append(e)
        fitted_coefficients = sph.Solve_LSQ(max_degree, training_data, A_training, e, grad)
        error = np.linalg.norm(test_data- A_test @ fitted_coefficients)
        fold_error[i][idx] = error
    plt.plot(np.log10(reg_param), np.log10(fold_error[i]), label=f'Fold {i+1}')
    plt.xlabel('Regularization parameter $Log_{10}(\lambda)$', fontsize=16)
    plt.ylabel('Test error $Log_{10}||Sf - f||$', fontsize=16)
    plt.grid(True)
mean_error = (fold_error.mean(axis=0))
min_error = np.min(mean_error)
best_e = reg_param[np.argmin(mean_error)]
plt.plot(np.log10(reg_param), np.log10(mean_error), label='Mean error', color='black', linewidth=3)
plt.legend()
plt.show()

print(f'The optimal value of e is {best_e} with a test error of {min_error}')

errors = []
norm = []
for e in np.logspace(-6,-0,30):
    fitted_coefficients = sph.Solve_LSQ(max_degree, data, A, e, grad)
    error = np.linalg.norm(training_data- A_training @ fitted_coefficients)
    errors.append(error)
    norm.append(np.sum( sph.construct_L(max_degree, grad) @ fitted_coefficients**2))


num_plot_points = 300
theta_grid, phi_grid = np.meshgrid(
    np.linspace(0, np.pi, num_plot_points),
    np.linspace(0, 2*np.pi, num_plot_points)
)

# Compute the fitted values on the grid
A_grid = sph.design_matrix_vectorized(theta_grid.flatten(), phi_grid.flatten(), max_degree)
fitted_grid = (A_grid @ coefficients).reshape(phi_grid.shape)


# Convert spherical coordinates to Cartesian coordinates for plotting
x_grid = r * np.sin(theta_grid) * np.cos(phi_grid)
y_grid = r * np.sin(theta_grid) * np.sin(phi_grid)
z_grid = r * np.cos(theta_grid)

# Plot the original and fitted data on the sphere
fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta), c=data, cmap='viridis')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
mappable = plt.cm.ScalarMappable(cmap='viridis')
mappable.set_array(data.real)
fig.colorbar(mappable, ax=ax1, shrink=0.5, aspect=7)

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(x_grid, y_grid, z_grid, facecolors=plt.cm.viridis(fitted_grid.real), rstride=1, cstride=1, shade=False)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

mappable = plt.cm.ScalarMappable(cmap='viridis')
mappable.set_array(fitted_grid.real)
fig.colorbar(mappable, ax=ax2, shrink=0.5, aspect=7)
plt.show()

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(111)
edge_colors = ['blue', 'green', 'red', 'purple', 'magenta', 'orange']
ax1.plot(np.log10(errors), np.log10(norm))

e_values = [10e-6, 10e-5, 10e-4, 10e-3, 10e-2]
for idx, e in enumerate(e_values):
    fitted_coefficients = sph.Solve_LSQ(max_degree, data, A, e, grad)
    error = np.linalg.norm(training_data - A_training @ fitted_coefficients)
    norm = np.sum(sph.construct_L(max_degree, grad) @ fitted_coefficients**2)
    ax1.scatter(np.log10(error), np.log10(norm), label=f'$\lambda$={e_values[idx]}', edgecolors=edge_colors[idx], facecolors='none', marker='o', s=100)
ax1.set_xlabel('Residual norm $Log_{10}||Sf - f||$', fontsize=18)
ax1.set_ylabel('Coefficients norm $Log_{10}||v||$', fontsize=18)
ax1.legend(fontsize=16)
ax1.grid(True)
plt.show()
