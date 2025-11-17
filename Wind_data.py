import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from meteostat import Stations, Hourly
import Vector_spherical_harmonics as vsh
from datetime import datetime
from sklearn.decomposition import PCA
from scipy.signal import csd, welch

def get_weather_stations():
    stations = Stations()
    df = stations.fetch()
    return df


def fetch_weather_data_vector(stations, start, end):
    """Fetch weather data for the given stations and time range."""
    data = []
    station_ids = stations.index.astype(str).tolist()
    station_ids.remove('42065')
    try:
        weather = Hourly(station_ids, start, end)
        weather = weather.fetch()
        print(f"Weather data fetched successfully for {len(weather)} records.")
    except Exception as e:
        print(f"Error fetching weather data for station: {e}")
    return weather


def convert_latitude(lat):
    return np.radians(90 - lat)


def convert_longitude(long):
    return np.radians(long)



# # Fetch weather stations
# print("Fetching weather stations...")
# df_stations = get_weather_stations()
# print("Saving...")
# df_stations.to_csv('stations.csv')
# print("Done.")

# # Fetch weather data
# print("Fetching weather data...")
# start = datetime(2023, 7, 19, 0, 0)
# end = start + pd.Timedelta(days=1)

# # Only fetch necessary columns to reduce memory usage
# columns = ['wspd', 'wdir', 'station']
# df_weather = fetch_weather_data_vector(df_stations, start, end)
# if not df_weather.empty:
#     df_weather = df_weather.reset_index()
#     df_weather = df_weather[columns + ['time']] if all(col in df_weather.columns for col in columns) else df_weather

# print("Saving...")
# df_weather.to_csv('weather.csv', index=False)
# print("Done.")

# # Merge weather data with station coordinates
# df_stations = pd.read_csv('stations.csv')
# df_weather = pd.read_csv('weather.csv')

# df_stations = df_stations[['latitude', 'longitude']]
# df_merged = df_weather.merge(df_stations, left_on='station', right_on='id', how='left')

# df_merged.to_csv('merged.csv')

# Plot weather data
df_merged = pd.read_csv('merged.csv')


# Remove rows with latitude -90 or 90
df_merged = df_merged[(df_merged['latitude'] > -89.5) & (df_merged['latitude'] < 89.5)]

# Locate rows with NaN wdir and wind speed values and remove them
nan_rows = df_merged[df_merged[['wspd', 'wdir', 'latitude', 'longitude']].isna().any(axis=1)]
df_merged = df_merged.drop(nan_rows.index)

# Pivot the data to create matrices for latitude, longitude, wspd, and wdir
df_merged['time'] = pd.to_datetime(df_merged['time'])
pivot_index = 'station'
pivot_column = 'time'



wspd_pivot = df_merged.pivot(index=pivot_index, columns=pivot_column, values='wspd')
wdir_pivot = df_merged.pivot(index=pivot_index, columns=pivot_column, values='wdir')

# Get the station indices after pivoting
station_indices = wspd_pivot.index

# Create a mapping from station id to latitude and longitude
station_lat_map = df_merged.drop_duplicates('station').set_index('station')['latitude']
station_lon_map = df_merged.drop_duplicates('station').set_index('station')['longitude']

# Align lat/lon arrays to the order of station_indices
theta = np.array([convert_latitude(station_lat_map[station]) for station in station_indices])
phi = np.array([convert_longitude(station_lon_map[station]) for station in station_indices])


wspd_matrix = wspd_pivot.values
wdir_matrix = wdir_pivot.values

data_matrix_theta = wspd_matrix * np.cos(np.radians(wdir_matrix))
data_matrix_phi = wspd_matrix * np.sin(np.radians(wdir_matrix))

data_matrix_theta = data_matrix_theta[~np.isnan(data_matrix_theta).any(axis=1)]
data_matrix_phi = data_matrix_phi[~np.isnan(data_matrix_phi).any(axis=1)]

# take a random subset of the data
np.random.seed(42)  # For reproducibility
sample_size = 1000
sample_indices = np.random.choice(data_matrix_theta.shape[0], sample_size, replace=False)

data_matrix_theta = data_matrix_theta[sample_indices, :]
data_matrix_phi = data_matrix_phi[sample_indices, :]

theta = theta[sample_indices]
phi = phi[sample_indices]

data = np.concatenate((data_matrix_theta, data_matrix_phi), axis=0)
data_centered = data - np.mean(data, axis=0)



pc = PCA(n_components=24)
pc.fit(data_centered)
data = pc.transform(data_centered)

np.savez_compressed(f'wind_data_pca.npz', data=data, theta=theta, phi=phi)





