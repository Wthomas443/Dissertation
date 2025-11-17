import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import Spherical_harmonic_functions as sph
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import LeaveOneOut
from matplotlib.patches import Patch

print('Processing data...')
r = 6378000
nRowsRead = None
df1 = pd.read_csv('GlobalLandTemperaturesByCity.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'GlobalLandTemperaturesByCity.csv'

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

# Parameters
n = len(data)
max_degree = 32
regularization_parameter = 0.0009326033468832219
grad = 2

# Create plot to find optimum regularisation parameter
num_folds = 5
fold_error = np.zeros((num_folds, 100))
for i in range(num_folds):
    np.random.seed(i)
    test_idx = np.random.randint(0, n, int(0.2 * n))
    test_theta = theta[test_idx]
    test_phi = phi[test_idx]
    test_data = data[test_idx]
    training_data = np.delete(data, test_idx)
    training_theta = np.delete(theta, test_idx)
    training_phi = np.delete(phi, test_idx)
    A_training = sph.design_matrix_vectorized(training_theta, training_phi, max_degree)
    A_test = sph.design_matrix_vectorized(test_theta, test_phi, max_degree)
    reg_param = []
    for idx, e in enumerate(np.logspace(-12, 0, 100)):
        print(f'Fold {i+1}, Regularization parameter {e}')
        reg_param.append(e)
        fitted_coefficients = sph.Solve_LSQ(max_degree, training_data, A_training, e, grad)
        error = np.linalg.norm(test_data - A_test @ fitted_coefficients)
        fold_error[i][idx] = error
    plt.plot(np.log10(reg_param), np.log10(fold_error[i]), label=f'Fold {i+1}')
    plt.xlabel('Regularization parameter $Log_{10}(\lambda)$', fontsize=16)
    plt.ylabel('Test error $Log_{10}||f_{test}-A_{test}v||$', fontsize=16)
    plt.grid(True)
mean_error = (fold_error.mean(axis=0))
min_error = np.min(mean_error)
best_e = reg_param[np.argmin(mean_error)]
plt.plot(np.log10(reg_param), np.log10(mean_error), label='Mean error', color='black', linewidth=3)
plt.legend()
plt.show()

print(f'The optimal value of e is {best_e} with a test error of {min_error}')

# =============================================
# DATA VISUALIZATION
# =============================================

# Convert to Cartesian coordinates
x_data = r * np.sin(theta) * np.cos(phi)
y_data = r * np.sin(theta) * np.sin(phi)
z_data = r * np.cos(theta)

print('Data visualization...')

fig = plt.figure(figsize=(10, 8))
ax_data = fig.add_subplot(111, projection='3d')
sc = ax_data.scatter(x_data, y_data, z_data, c=data, cmap='viridis', s=50)
ax_data.set_title('Original Temperature Data Points', fontsize=14)
ax_data.set_xlabel('X (m)')
ax_data.set_ylabel('Y (m)')
ax_data.set_zlabel('Z (m)')
plt.colorbar(sc, ax=ax_data, shrink=0.5, aspect=7, label='Temperature (°C)')
plt.tight_layout()
plt.show()

# =============================================
# SPHERICAL HARMONIC APPROXIMATION
# =============================================

# Fit model
print('Fitting model...')
A = sph.design_matrix_vectorized(theta, phi, max_degree)
coefficients = sph.Solve_LSQ(max_degree, data, A, regularization_parameter, grad)

# Create grid
print('Making approximation on grid...')
num_plot_points = 200
theta_grid, phi_grid = np.meshgrid(
    np.linspace(0, np.pi, num_plot_points),
    np.linspace(0, 2*np.pi, num_plot_points)
)

A_grid = sph.design_matrix_vectorized(theta_grid.flatten(), phi_grid.flatten(), max_degree)
fitted_grid = (A_grid @ coefficients).reshape(phi_grid.shape)

# Convert grid to Cartesian
x_grid = r * np.sin(theta_grid) * np.cos(phi_grid)
y_grid = r * np.sin(theta_grid) * np.sin(phi_grid)
z_grid = r * np.cos(theta_grid)

