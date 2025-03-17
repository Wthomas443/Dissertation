import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from meteostat import Stations, Hourly
from datetime import datetime, timedelta
import Vector_spherical_harmonics as vsh

def get_weather_stations():
    stations = Stations()
    df = stations.fetch()
    return df


def fetch_weather_data(stations, start, end):
    """Fetch weather data for the given stations and time range."""
    data = []
    station_ids = stations.index.astype(str).tolist()
    for i,station in enumerate(station_ids):
        try:
            #print(f"Fetching weather data for station {station}({i+1}/{len(station_ids)})...")
            weather = Hourly(station, start, end)
            weather = weather.fetch()
        except Exception as e:
            print(f"Error fetching weather data for station {station}({i+1}/{len(station_ids)}): {e} ")
        if not weather.empty:
            data.append(weather)
    return pd.concat(data) if data else pd.DataFrame()

def fetch_weather_data_vector(stations, start, end):
    """Fetch weather data for the given stations and time range."""
    data = []
    station_ids = stations.index.astype(str).tolist()
    station_ids.remove('42065')
    try:
        weather = Hourly(station_ids, start, end)
        weather = weather.fetch()
    except Exception as e:
        print(f"Error fetching weather data for station: {e}")
    return weather


def convert_latitude(lat):
    return np.radians(90 - lat)


def convert_longitude(long):
    return np.radians(long)



# Fetch weather stations
# print("Fetching weather stations...")
# df_stations = get_weather_stations()
# start = datetime(2023, 7, 19, 0, 0)
# end = start 
# print("Saving...")
# df_stations.to_csv('stations.csv')
# print("Done.")

# Fetch weather data
# print("Fetching weather data...")
# df_weather = fetch_weather_data_vector(df_stations, start, end)
# print("Saving...")
# df_weather.to_csv('weather.csv')
# print("Done.")

# Merge weather data with station coordinates
# df_stations = pd.read_csv('stations.csv')
# df_weather = pd.read_csv('weather.csv')

# df_stations = df_stations[['id', 'latitude', 'longitude']]
# df_merged = df_weather.merge(df_stations, left_on='station', right_on='id', how='left')

# df_merged.to_csv('merged.csv')

# Plot weather data
df_merged = pd.read_csv('merged.csv')


# Remove rows with latitude -90 or 90
df_merged = df_merged[(df_merged['latitude'] != -90) & (df_merged['latitude'] != 90)]

# Locate rows with NaN wdir and wind speed values and remove them
nan_rows = df_merged[df_merged[['wspd', 'wdir', 'latitude', 'longitude']].isna().any(axis=1)]
df_merged = df_merged.drop(nan_rows.index)

# take a random subset of the data
df_merged = df_merged.sample(150)

theta = df_merged['latitude'].values
phi = df_merged['longitude'].values


theta = np.array([convert_latitude(lat) for lat in theta])
phi = np.array([convert_longitude(long) for long in phi])

# Calculate the wind speed vector
u = df_merged['wspd'].values
v = df_merged['wdir'].values
u = u * np.cos(v)
v = u * np.sin(v)


n = len(theta)
r = 6378000
max_degree = 80
regularization_parameter = 0.3593813663804626
grad = 2

data = np.concatenate((u, v))

# Split the data into a chosen ratio with random indices
np.random.seed(0)
test_idx = np.random.randint(0, n, int(0.2*n))
theta_test = theta[test_idx]
phi_test = phi[test_idx]
data_test = np.concatenate([data[0:n][test_idx], data[n:2*n][test_idx]])
data_train = np.concatenate([np.delete(data[0:n], test_idx), np.delete(data[n:2*n], test_idx)])
theta_train = np.delete(theta, test_idx)
phi_train = np.delete(phi, test_idx)

# Split the design matrix accordingly
A_train = vsh.VectorDesignMatrix(theta_train, phi_train, max_degree)
A_test = vsh.VectorDesignMatrix(theta_test, phi_test, max_degree)

coefficients = vsh.Solve_LSQ(max_degree, data_train, A_train, regularization_parameter, grad)

# Generate a grid of points on the sphere for visualization of vector field
num_plot_points = 50
theta_grid, phi_grid = np.meshgrid(
    np.linspace(0.01, np.pi-0.01, num_plot_points),
    np.linspace(0, 2*np.pi, num_plot_points)
)

# Generate a grid of points on the sphere for visualization of scalar fields
num_plot_points_scalar = 300
theta_grid_scalar, phi_grid_scalar = np.meshgrid(
    np.linspace(0.01, np.pi-0.01, num_plot_points_scalar),
    np.linspace(0, 2*np.pi, num_plot_points_scalar)
)

x =   np.sin(theta_train) * np.cos(phi_train)
y =   np.sin(theta_train) * np.sin(phi_train)
z =   np.cos(theta_train)


#split data_train and normalise to plot scalar fields
data_train_phi = data_train[0:len(theta_train)]
data_train_theta = data_train[len(theta_train):]
data_train_phi_n = (data_train_phi - data_train_phi.min()) / (data_train_phi.max() - data_train_phi.min())
data_train_theta_n = (data_train_theta - data_train_theta.min()) / (data_train_theta.max() - data_train_theta.min())
data_plot = vsh.convert_to_cartesian(theta_train, phi_train, data_train)



#plot the data

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')
ax.quiver(x, y, z, data_plot[:,0].real, data_plot[:,1].real, data_plot[:,2].real, length=0.1, normalize=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


plt.show()

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(x, y, z, c = data_train_phi_n, cmap='viridis')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
mappable = plt.cm.ScalarMappable(cmap='viridis')
mappable.set_array(data_train_phi)
fig.colorbar(mappable, ax=ax1, shrink=0.5, aspect=5)

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(x, y, z, c = data_train_theta_n, cmap='viridis')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
mappable = plt.cm.ScalarMappable(cmap='viridis')
mappable.set_array(data_train_theta)
fig.colorbar(mappable, ax=ax2, shrink=0.5, aspect=5)

plt.show()

# Compute the fitted values on the grid
A_grid = vsh.VectorDesignMatrix(theta_grid.flatten(), phi_grid.flatten(), max_degree)
fitted_grid = A_grid @ coefficients

A_grid_scalar = vsh.VectorDesignMatrix(theta_grid_scalar.flatten(), phi_grid_scalar.flatten(), max_degree)
fitted_grid_scalar = A_grid_scalar @ coefficients

# Reshape and normalize the components in one step
phi_component = fitted_grid_scalar[:len(theta_grid_scalar.flatten())].real.reshape(theta_grid_scalar.shape)
theta_component = fitted_grid_scalar[len(theta_grid_scalar.flatten()):].real.reshape(theta_grid_scalar.shape)
phi_component_n = (phi_component - phi_component.min()) / (phi_component.max() - phi_component.min())
theta_component_n = (theta_component - theta_component.min()) / (theta_component.max() - theta_component.min())

fitted_grid_plot = vsh.convert_to_cartesian(theta_grid.flatten(), phi_grid.flatten(), fitted_grid)

# Convert grid spherical to Cartesian coordinates for plotting theta and phi components
x_grid, y_grid, z_grid = np.sin(theta_grid) * np.cos(phi_grid), np.sin(theta_grid) * np.sin(phi_grid), np.cos(theta_grid)
x_grid_scalar, y_grid_scalar, z_grid_scalar = np.sin(theta_grid_scalar) * np.cos(phi_grid_scalar), np.sin(theta_grid_scalar) * np.sin(phi_grid_scalar), np.cos(theta_grid_scalar)

# Plot the fitted data
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')
ax.quiver(x_grid.flatten(), y_grid.flatten(), z_grid.flatten(), fitted_grid_plot[:,0].real, fitted_grid_plot[:,1].real, fitted_grid_plot[:,2].real, length=0.1, normalize=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(x_grid_scalar, y_grid_scalar, z_grid_scalar, facecolors = plt.cm.viridis(phi_component_n), rstride=1, cstride=1, shade=False)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
mappable = plt.cm.ScalarMappable(cmap='viridis')
mappable.set_array(phi_component)
fig.colorbar(mappable, ax=ax1, shrink=0.5, aspect=5)

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(x_grid_scalar, y_grid_scalar, z_grid_scalar, facecolors = plt.cm.viridis(theta_component_n), rstride=1, cstride=1, shade=False)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
mappable = plt.cm.ScalarMappable(cmap='viridis')
mappable.set_array(theta_component)
fig.colorbar(mappable, ax=ax2, shrink=0.5, aspect=5)

plt.show()

# Compute the error
#training error
error = np.linalg.norm(data_train - A_train @ coefficients)
print(f"Training Error: {error}")
#MSE
MSE = (np.sum((data_train - A_train @ coefficients)**2)/len(data_train)).real
print(f"MSE: {MSE}")
#test error
error = np.linalg.norm(data_test - A_test @ coefficients)
print(f"Test Error: {error}")
#MSE
MSE = (np.sum((data_test - A_test @ coefficients)**2)/len(data_test)).real
print(f"MSE: {MSE}")


# Plot the error for increasing max degree
max_degrees = range(0, 40)
train_errors = []
test_errors = []

for max_degree in max_degrees:
    A_train = vsh.VectorDesignMatrix(theta_train, phi_train, max_degree)
    A_test = vsh.VectorDesignMatrix(theta_test, phi_test, max_degree)
    coefficients = vsh.Solve_LSQ(max_degree, data_train, A_train, regularization_parameter, grad)
    
    train_error = np.linalg.norm(data_train - A_train @ coefficients)
    test_error = np.linalg.norm(data_test - A_test @ coefficients)
    
    train_errors.append(train_error)
    test_errors.append(test_error)

plt.figure(figsize=(10, 6))
plt.plot(max_degrees, np.log10(train_errors), label='Training Error')
plt.plot(max_degrees, np.log10(test_errors), label='Test Error')
plt.xlabel('Max Degree')
plt.ylabel('Error')
plt.title('Error vs Max Degree')
plt.legend()
plt.grid(True)
plt.show()

#Create plot to find optimum regularisation parameter
num_folds = 10
fold_error = np.zeros((num_folds,40))
reg_param = np.logspace(-6, 2, 40)
for i in range(num_folds):
    # Split the data into a chosen ratio with random indices
    np.random.seed(i)
    test_idx = np.random.randint(0, n, int(0.2 * n))
    theta_test = theta[test_idx]
    phi_test = phi[test_idx]
    data_test = np.concatenate([data[0:n][test_idx], data[n:2*n][test_idx]])
    data_train = np.concatenate([np.delete(data[0:n], test_idx), np.delete(data[n:2*n], test_idx)])
    theta_train = np.delete(theta, test_idx)
    phi_train = np.delete(phi, test_idx)

    # Split the design matrix accordingly
    A_train = vsh.VectorDesignMatrix(theta_train, phi_train, max_degree)
    A_test = vsh.VectorDesignMatrix(theta_test, phi_test, max_degree)

    for idx, e in enumerate(reg_param):
        fitted_coefficients = vsh.Solve_LSQ(max_degree, data_train, A_train, e, grad)
        error = np.linalg.norm(data_test - A_test @ fitted_coefficients)
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






