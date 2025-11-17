import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import Spherical_harmonic_functions as sph
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import KFold
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

def uncertainty_measure(e, grad, train_idx, theta, phi, data, max_degree, num_plot_points):
    """
    Calculate uncertainty for given regularization parameter, grad, and data split
    """
    # Split data using provided indices
    train_theta, train_phi, train_data = theta[train_idx], phi[train_idx], data[train_idx]

    # Train spherical harmonic model
    A_train = sph.design_matrix_vectorized(train_theta, train_phi, max_degree)
    coefficients = sph.Solve_LSQ(max_degree, train_data, A_train, e, grad)

    A_train_pred = sph.design_matrix_vectorized(train_theta, train_phi, max_degree)
    train_predictions = A_train_pred @ coefficients
    train_residuals = (train_data - train_predictions).real

    # Prepare data for GPR
    X_train_points = np.column_stack((train_theta, train_phi))

    theta_grid, phi_grid = np.meshgrid(np.linspace(0, np.pi, num_plot_points),
                                   np.linspace(0, 2*np.pi, num_plot_points))
    X_grid_points = np.column_stack((theta_grid.flatten(), phi_grid.flatten()))

    # Initialize and fit GP model
    kernel = RBF(length_scale=1.0) + WhiteKernel()
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0, normalize_y=True)
    gp.fit(X_train_points, train_residuals)
    
    # Predict uncertainty
    _, y_std = gp.predict(X_grid_points, return_std=True)
    uncertainty = 2 * y_std
    norm_uncertainty = sph.spherical_norm(uncertainty.reshape(theta_grid.shape), theta_grid)

    return norm_uncertainty

theta = np.array([convert_latitude(lat) for lat in filtered_df['Latitude']])
phi = np.array([convert_longitude(long) for long in filtered_df['Longitude']])
data = np.array(filtered_df['AverageTemperature'])

# Remove NaN values
valid_mask = ~np.isnan(data)
theta, phi, data = theta[valid_mask], phi[valid_mask], data[valid_mask]

max_degree = 2
num_plot_points = 50

# Grid search parameters
grad_values = [1, 2, 3, 4]  # Different grad values to try
reg_params = np.logspace(-8, 1, 10)  # Regularization parameters
param_combinations = list(product(grad_values, reg_params))

# K-Fold Cross Validation
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Store results
results = []
best_score = float('inf')
best_params = None

print("Starting grid search...")

for i, (grad, e) in enumerate(param_combinations):
    print(f"\nTesting combination {i+1}/{len(param_combinations)}: grad={grad}, e={e:.2e}")
    
    fold_uncertainties = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
        norm_uncertainty = uncertainty_measure(e, grad, train_idx, theta, phi, data, max_degree, num_plot_points)
        fold_uncertainties.append(norm_uncertainty)
        print(f"Fold {fold+1}: Uncertainty = {norm_uncertainty:.4f}")
    
    mean_uncertainty = np.mean(fold_uncertainties)
    std_uncertainty = np.std(fold_uncertainties)
    
    results.append({
        'grad': grad,
        'reg_param': e,
        'mean_uncertainty': mean_uncertainty,
        'fold_uncertainties': fold_uncertainties
    })
    
    print(f"  Mean uncertainty: {mean_uncertainty:.4f}")
    
    if mean_uncertainty < best_score:
        print("New minimum uncertainty")

        best_score = mean_uncertainty
        best_params = (grad, e)
        

# Convert results to DataFrame for easier analysis
results_df = pd.DataFrame(results)

# Find the best combination
best_result = results_df.loc[results_df['mean_uncertainty'].idxmin()]
best_grad, best_reg_param = best_result['grad'], best_result['reg_param']


print(f"{'='*50}")
print(f"Best grad: {best_grad}")
print(f"Best regularization parameter: {best_reg_param:.2e}")
print(f"Best mean uncertainty: {best_score:.4f}")
print(f"{'='*50}")

# Visualize results
fig= plt.figure(figsize=(15, 12))

grad_unique = sorted(results_df['grad'].unique())
reg_unique = sorted(results_df['reg_param'].unique())

#Uncertainty vs regularization for each grad

ax2 = fig.add_subplot(111)
for grad_val in grad_unique:
    grad_mask = results_df['grad'] == grad_val
    grad_results = results_df[grad_mask]
    ax2.loglog(grad_results['reg_param'], grad_results['mean_uncertainty'], label=f'grad={grad_val}', linewidth=2, markersize=4)
ax2.set_xlabel('Regularization Parameter (e)')
ax2.set_ylabel('Mean Uncertainty')
ax2.set_title('Uncertainty vs Regularization Parameter (by Grad)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Final training with best parameters
print(f"\nTraining final model with best parameters: grad={best_grad}, e={best_reg_param:.2e}")
A = sph.design_matrix_vectorized(theta, phi, max_degree)
coefficients = sph.Solve_LSQ(max_degree, data, A, best_reg_param, best_grad)

# Predict on training data for residual calculation
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
X = np.column_stack((theta, phi))
X_grid_points = np.column_stack((theta_grid.flatten(), phi_grid.flatten()))

print("Fitting Gaussian Process Regressor...")
# Initialize and fit GP model
kernel = RBF(length_scale=0.1, length_scale_bounds=(1e-8, 1e1)) + WhiteKernel()
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0, normalize_y=True)
gp.fit(X, residuals)

y_mean, y_std = gp.predict(X_grid_points, return_std=True)
uncertainty = 2 * y_std

print("Creating visualization...")
# ========== VISUALIZATION ==========
fig = plt.figure(figsize=(18, 6))

# Subplot 1: Temperature Approximation
ax1 = fig.add_subplot(121, projection='3d')
norm_temp = plt.Normalize(vmin=data.min(), vmax=data.max())
surf1 = ax1.plot_surface(x_grid, y_grid, z_grid, 
                        facecolors=plt.cm.viridis(norm_temp(fitted_grid)),
                        rstride=2, cstride=2, shade=False)
ax1.set_title(f'Spherical Harmonic Approximation (grad={best_grad}, e={best_reg_param:.2e})')
fig.colorbar(plt.cm.ScalarMappable(norm=norm_temp, cmap='viridis'), 
             ax=ax1, label='Temperature (°C)')

# Subplot 2: GPR Uncertainty
ax3 = fig.add_subplot(122, projection='3d')
norm_uncertainty_plot = plt.Normalize(vmin=np.min(uncertainty), vmax=np.max(uncertainty))
surf3 = ax3.plot_surface(x_grid, y_grid, z_grid,
                        facecolors=plt.cm.plasma(norm_uncertainty_plot(uncertainty.reshape(x_grid.shape))),
                        rstride=2, cstride=2, shade=False)
ax3.set_title('GPR Uncertainty')
fig.colorbar(plt.cm.ScalarMappable(norm=norm_uncertainty_plot, cmap='plasma'), 
             ax=ax3, label='Uncertainty')

plt.tight_layout()
plt.show()

# Print detailed results table
print("\nDetailed Results Table:")
print("-" * 80)
print(f"{'Grad':<6} {'Reg Param':<15} {'Mean Uncertainty':<18} {'Std Uncertainty':<15}")
print("-" * 80)
for _, row in results_df.sort_values('mean_uncertainty').head(10).iterrows():
    print(f"{row['grad']:<6} {row['reg_param']:<15.2e} {row['mean_uncertainty']:<18.4f} {row['std_uncertainty']:<15.4f}")
print("-" * 80)