import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import Spherical_harmonic_functions as sph
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn import preprocessing
import warnings
from itertools import product

# Configuration
r = 6378000  # Earth radius in meters

# Load and prepare data
df = pd.read_csv('GlobalLandTemperaturesByCity.csv')
filtered_df = df[df['dt'] == '2003-07-01']

# Coordinate conversion functions
def convert_latitude(lat):
    if lat[-1] == 'N': return (90 - float(lat[:-1])) * np.pi / 180
    elif lat[-1] == 'S': return (90 + float(lat[:-1])) * np.pi / 180
    else: raise ValueError("Invalid latitude format")

def convert_longitude(long):
    if long[-1] == 'W': return (360 - float(long[:-1])) * np.pi / 180
    elif long[-1] == 'E': return float(long[:-1]) * np.pi / 180
    else: raise ValueError("Invalid longitude format")

def uncertainty_measure_LOO(e, grad, theta, phi, data, max_degree, num_plot_points):

    # Use leave one out residuals to train GPR
    loo = LeaveOneOut()
    all_residuals = []
    all_locations = []
    
    print("Calculating LOO residuals...")
    for train_idx, val_idx in loo.split(data):
        
        A_train = sph.design_matrix_vectorized(theta[train_idx], phi[train_idx], max_degree)
        coefficients = sph.Solve_LSQ(max_degree, data[train_idx], A_train, e, grad)
        
        # Predict on the left-out point
        A_val = sph.design_matrix_vectorized(theta[val_idx], phi[val_idx], max_degree)
        val_prediction = (A_val @ coefficients).real
        residual = data[val_idx].real - val_prediction
        
        all_residuals.append(residual[0])
        all_locations.append([theta[val_idx][0], phi[val_idx][0]])
    
    all_residuals = np.array(all_residuals)
    all_locations = np.array(all_locations)
    
    #Train GPR
    print("Training GPR")
    kernel = Matern(nu=1.5, length_scale=1.0)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0, normalize_y=True)
    gp.fit(all_locations, all_residuals)
    
    # Create prediction grid
    theta_grid, phi_grid = np.meshgrid(np.linspace(0, np.pi, num_plot_points),
                                     np.linspace(0, 2*np.pi, num_plot_points))
    X_grid_points = np.column_stack((theta_grid.flatten(), phi_grid.flatten()))
    
    #Predict uncertainty
    _, y_std = gp.predict(X_grid_points, return_std=True)
    uncertainty = 2 * y_std
    norm_uncertainty = sph.spherical_norm(uncertainty.reshape(theta_grid.shape), theta_grid)
    
    return norm_uncertainty


def uncertainty_measure(e, grad, train_idx, val_idx, theta, phi, data, max_degree, num_plot_points):
   
    # Split data
    train_theta, train_phi, train_data = theta[train_idx], phi[train_idx], data[train_idx]
    val_theta, val_phi, val_data = theta[val_idx], phi[val_idx], data[val_idx]
    
    # Train model
    A_train = sph.design_matrix_vectorized(train_theta, train_phi, max_degree)
    coefficients = sph.Solve_LSQ(max_degree, train_data, A_train, e, grad)
    
    # Calculate residuals on test set
    A_val = sph.design_matrix_vectorized(val_theta, val_phi, max_degree)
    val_predictions = A_val @ coefficients
    val_residuals = (val_data - val_predictions).real
    
    # Train GPR
    X_val_points = np.column_stack((val_theta, val_phi))
    
    # Create prediction grid
    theta_grid, phi_grid = np.meshgrid(np.linspace(0, np.pi, num_plot_points),
                                     np.linspace(0, 2*np.pi, num_plot_points))
    X_grid_points = np.column_stack((theta_grid.flatten(), phi_grid.flatten()))
    
    # Fit GPR to validation residuals
    kernel = Matern(nu=1.5, length_scale=1.0)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-8, normalize_y=True)
    gp.fit(X_val_points, val_residuals)
    
    # Predict uncertainty
    _, y_std = gp.predict(X_grid_points, return_std=True)
    uncertainty = 2 * y_std
    norm_uncertainty = sph.spherical_norm(uncertainty.reshape(theta_grid.shape), theta_grid)
    
    return norm_uncertainty

def GridSearch(param_combinations, kf, LOO=False):
    best_score = float('inf')
    best_params = None
    results = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if LOO == True:
            for i, (grad, e) in enumerate(param_combinations):
                print(f"\nTesting combination {i+1}/{len(param_combinations)}: grad={grad}, e={e:.2e}")
                norm_uncertainty = uncertainty_measure_LOO(e, grad, theta, phi, data, max_degree, num_plot_points)
                print(f"  Normalized Uncertainty: {norm_uncertainty:.4f}")
                results.append({'grad': grad, 'reg_param': e, 'uncertainty': norm_uncertainty})
                results_df = pd.DataFrame(results)
            if norm_uncertainty < best_score:
                print("New minimum uncertainty")
                best_score = norm_uncertainty
                best_params = (grad, e)
        elif LOO == False:
            for i, (grad, e) in enumerate(param_combinations):
                print(f"\nTesting combination {i+1}/{len(param_combinations)}: grad={grad}, e={e:.2e}")
                fold_uncertainties = []
                for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
                    norm_uncertainty = uncertainty_measure(e, grad, train_idx, val_idx, theta, phi, data, max_degree, num_plot_points)
                    fold_uncertainties.append(norm_uncertainty)
                    print(f"Fold {fold+1}: Uncertainty = {norm_uncertainty:.4f}")
                results.append({'grad': grad, 'reg_param': e, 'uncertainty': np.mean(fold_uncertainties)})
                results_df = pd.DataFrame(results)
                mean_uncertainty = np.mean(fold_uncertainties)
                print(f"  Mean uncertainty: {mean_uncertainty:.4f}")
                if mean_uncertainty < best_score:
                    print("New minimum uncertainty")
                    best_score = mean_uncertainty
                    best_params = (grad, e)
    return best_params, best_score, results_df


theta = np.array([convert_latitude(lat) for lat in filtered_df['Latitude']])
phi = np.array([convert_longitude(long) for long in filtered_df['Longitude']])
data = np.array(filtered_df['AverageTemperature'])

# Randomly sample data points
np.random.seed(42)  # For reproducibility
sample_indices = np.random.choice(len(data), size=min(1000, len(data)), replace=False)
theta = theta[sample_indices]
phi = phi[sample_indices]
data = data[sample_indices]

max_degree = 40
num_plot_points = 400

# Grid search parameters
grad_values = [1, 2, 3, 4]  # Different grad values to try
reg_params = np.logspace(-8, 1, 20)  # Regularization parameters
param_combinations = list(product(grad_values, reg_params))

# K-Fold Cross Validation
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

print("Starting grid search...")
        
best_params, best_score, results_df = GridSearch(param_combinations, kf, LOO=False)
best_grad, best_reg_param = best_params

# Visualize results
fig= plt.figure(figsize=(15, 12))

grad_unique = sorted(results_df['grad'].unique())
reg_unique = sorted(results_df['reg_param'].unique())

#Uncertainty vs regularization for each grad

ax2 = fig.add_subplot(111)
for grad_val in grad_unique:
    grad_mask = results_df['grad'] == grad_val
    grad_results = results_df[grad_mask]
    ax2.loglog(grad_results['reg_param'], grad_results['uncertainty'], label=f'grad={grad_val}', linewidth=2, markersize=4)
ax2.set_xlabel('Regularization Parameter')
ax2.set_ylabel('Uncertainty')
ax2.set_title('Uncertainty vs Regularization Parameter')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig('Uncertainty_plots\\Uncertainty_vs_Regularization.png')
plt.close()


# Final training with best parameters
print(f"\nTraining final model with best parameters: grad={best_grad}, e={best_reg_param:.2e}")
A = sph.design_matrix_vectorized(theta, phi, max_degree)
coefficients = sph.Solve_LSQ(max_degree, data, A, best_reg_param, best_grad)

all_val_residuals = []
all_val_theta = []
all_val_phi = []

for train_idx, val_idx in kf.split(data):

    # Train 
    A_train = sph.design_matrix_vectorized(theta[train_idx], phi[train_idx], max_degree)
    coefficients_fold = sph.Solve_LSQ(max_degree, data[train_idx], A_train, best_reg_param, best_grad)
    
    # Predict 
    A_val = sph.design_matrix_vectorized(theta[val_idx], phi[val_idx], max_degree)
    val_predictions = A_val @ coefficients_fold
    val_residuals = (data[val_idx] - val_predictions).real
    
    all_val_residuals.extend(val_residuals)
    all_val_theta.extend(theta[val_idx])
    all_val_phi.extend(phi[val_idx])

all_val_residuals = np.array(all_val_residuals)
all_val_theta = np.array(all_val_theta)
all_val_phi = np.array(all_val_phi)

A = sph.design_matrix_vectorized(theta, phi, max_degree)
coefficients = sph.Solve_LSQ(max_degree, data, A, best_reg_param, best_grad)
predictions = A @ coefficients
residuals = (data - predictions).real

print("Creating meshgrid for visualization...")
# Create a grid for visualization
theta_grid, phi_grid = np.meshgrid(np.linspace(0, np.pi, num_plot_points),
                                   np.linspace(0, 2*np.pi, num_plot_points))
A_grid = sph.design_matrix_vectorized(theta_grid.flatten(), phi_grid.flatten(), max_degree)
fitted_grid = (A_grid @ coefficients).reshape(phi_grid.shape).real

x_grid = r * np.sin(theta_grid) * np.cos(phi_grid)
y_grid = r * np.sin(theta_grid) * np.sin(phi_grid)
z_grid = r * np.cos(theta_grid)

print("Preparing data for GPR...")
# Prepare data for GPR
X = np.column_stack((all_val_theta, all_val_phi))
y = all_val_residuals

scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Fitting Gaussian Process Regressor...")
# Initialize and fit GP model
kernel = Matern(nu=1.5, length_scale=1.0, length_scale_bounds=(1e-6, 1e5)) 
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-8, normalize_y=True)
gp.fit(X_scaled, y)

X_grid_points = np.column_stack((theta_grid.flatten(), phi_grid.flatten()))
X_grid_points = scaler.transform(X_grid_points)

y_mean, y_std = gp.predict(X_grid_points, return_std=True)
uncertainty = 2 * y_std
uncertainty = uncertainty.reshape(theta_grid.shape)

print("Creating visualization...")

fig = plt.figure(figsize=(18, 6))

# Temperature Approximation
ax1 = fig.add_subplot(121, projection='3d')
norm_temp = plt.Normalize(vmin=data.min(), vmax=data.max())
surf1 = ax1.plot_surface(x_grid, y_grid, z_grid, 
                        facecolors=plt.cm.viridis(norm_temp(fitted_grid)),
                        rstride=2, cstride=2, shade=False)
ax1.set_title(f'Spherical Harmonic Approximation (grad={best_grad}, e={best_reg_param:.2e})')
fig.colorbar(plt.cm.ScalarMappable(norm=norm_temp, cmap='viridis'), 
             ax=ax1, label='Temperature (°C)')

# GPR Uncertainty
ax3 = fig.add_subplot(122, projection='3d')
norm_uncertainty_plot = plt.Normalize(vmin=np.min(uncertainty), vmax=np.max(uncertainty))
surf3 = ax3.plot_surface(x_grid, y_grid, z_grid,
                        facecolors=plt.cm.plasma(norm_uncertainty_plot(uncertainty)),
                        rstride=2, cstride=2, shade=False)
ax3.set_title('GPR Uncertainty')
fig.colorbar(plt.cm.ScalarMappable(norm=norm_uncertainty_plot, cmap='plasma'), 
             ax=ax3, label='Uncertainty')

plt.tight_layout()
fig.savefig('Uncertainty_plots\\Climate_Data_Uncertainty_Visualization.png')
plt.close()