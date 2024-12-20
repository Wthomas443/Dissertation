import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt

def spherical_harmonics_basis_vectorized(l, m, theta, phi):
    return sph_harm(m, l, phi, theta)


# Generate some sample data (for example purposes, we use a combination of spherical harmonics)
def generate_sample_data_vectorized(theta, phi, function_max_degree, true_coefficients):
    """Generate sample data as a linear combination of spherical harmonics."""
    data = np.zeros_like(theta, dtype=np.complex128)
    idx = 0
    for l in range(function_max_degree + 1):
        for m in range(-l, l + 1):
            data += true_coefficients[idx] * spherical_harmonics_basis_vectorized(l, m, theta, phi)
            idx += 1
    return data 

# Generate the design matrix
def design_matrix_vectorized(theta, phi, max_degree):
    """Generate the design matrix for the least squares fitting."""
    A = np.zeros((len(theta), (max_degree + 1)**2), dtype=np.complex128)
    idx = 0
    for l in range(max_degree + 1):
        for m in range(-l, l + 1):
            A[:, idx] = spherical_harmonics_basis_vectorized(l, m, theta, phi)
            idx += 1
    return A

def construct_L(max_degree, power):
    L = np.zeros(((max_degree + 1)**2, (max_degree + 1)**2), dtype=np.complex128)
    idx = 0
    for l in range(max_degree + 1):
        for m in range(-l, l + 1):
            L[idx][idx] = (l * (l + 1))**power
            idx += 1
    return L

def Solve_LSQ(max_degree, data, A, e, grad):
    L = construct_L(max_degree, grad)
    coefficients = np.linalg.inv(A.conj().T @ A + e * L ) @ A.conj().T @ data
    return coefficients
       
def plot_spherical_harmonics(x_grid, y_grid, z_grid, colourmap):
    fig = plt.figure(figsize=(14, 6))
    ax2 = fig.add_subplot(111, projection='3d')
    ax2.plot_surface(x_grid, y_grid, z_grid, facecolors=plt.cm.viridis(colourmap/colourmap.max()), rstride=1, cstride=1, shade=False)
    ax2.set_xlabel('X', fontsize=16)
    ax2.set_ylabel('Y', fontsize=16)
    ax2.set_zlabel('Z', fontsize=16)

    # Add color bar which maps values to colors
    mappable = plt.cm.ScalarMappable(cmap='viridis')
    mappable.set_array(colourmap)
    fig.colorbar(mappable, ax=ax2, shrink=0.5, aspect=7)

