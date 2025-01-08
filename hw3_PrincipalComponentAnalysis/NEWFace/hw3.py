from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    #https://stackoverflow.com/questions/32207474/changing-numpy-array-to-float
    loaded_data = np.load(filename)
    mean = np.mean(loaded_data, axis=0)
    centered_dataset = loaded_data - mean
    return centered_dataset
    raise NotImplementedError


def get_covariance(dataset):
    dataset_t = np.transpose(dataset)
    n, m= dataset.shape
    covariance_matrix = (1 / (n - 1)) * np.dot(dataset_t, dataset)
    return covariance_matrix
    raise NotImplementedError


def get_eig(S, k):
    eigenvalues, eigenvectors = eigh(S, subset_by_index=[S.shape[0] - k, S.shape[0] - 1])
    #https://www.geeksforgeeks.org/how-to-use-numpy-argsort-in-descending-order-in-python/
    sorted_indices = np.argsort(eigenvalues)[::-1] 
    sorted_eigenvalues = eigenvalues[sorted_indices]
    corresponding_eigenvectors = eigenvectors[:, sorted_indices]
    #https://numpy.org/doc/stable/reference/generated/numpy.diag.html
    top_k_eigenvalues = sorted_eigenvalues[:k]
    top_k_eigenvectors = corresponding_eigenvectors[:, :k]

    Lambda = np.zeros((k, k))  
    np.fill_diagonal(Lambda, top_k_eigenvalues) 

    return Lambda, top_k_eigenvectors
    raise NotImplementedError


def get_eig_prop(S, prop):
    eigenvalues, eigenvectors = eigh(S)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    total_variance = np.sum(sorted_eigenvalues)
    variance_ratio = sorted_eigenvalues / total_variance

    k = np.where(variance_ratio > prop)[0]

    top_k_eigenvalues = sorted_eigenvalues[k]  
    top_k_eigenvectors = sorted_eigenvectors[:, k]

    Lambda = np.diag(top_k_eigenvalues)

    return Lambda, top_k_eigenvectors
    raise NotImplementedError


def project_image(image, U):
    alpha_i = np.dot(np.transpose(U), image)
    reconstruction = np.dot(U, alpha_i)

    return reconstruction
    raise NotImplementedError


def display_image(orig, proj):
    # Your implementation goes here!
    # Please use the format below to ensure grading consistency
    # fig, ax1, ax2 = plt.subplots(figsize=(9,3), ncols=2)
    # return fig, ax1, ax2
    original = np.reshape(orig, (64,64))
    projection = np.reshape(proj, (64,64))
    fig, (ax1, ax2) = plt.subplots(figsize=(9,3), ncols=2)

    im1 = ax1.imshow(original, aspect='equal', cmap='gray')
    ax1.set_title("Original")
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(projection, aspect='equal', cmap='gray')
    ax2.set_title("Projection")
    fig.colorbar(im2, ax=ax2)

    return fig, ax1, ax2
    raise NotImplementedError


def perturb_image(image, U, sigma):
    weights_alpha = np.dot(np.transpose(U), image)
    #https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
    perturbation = np.random.normal(0, sigma, size=weights_alpha.shape)
    perturbed_weights = weights_alpha + perturbation
    perturbed_image = np.dot(U, perturbed_weights)

    return perturbed_image
    raise NotImplementedError


def combine_image(image1, image2, U, lam):
    weight1 = np.dot(np.transpose(U), image1)
    weight2 = np.dot(np.transpose(U), image2)

    weight_comb = lam * weight1 + (1 - lam) * weight2
    comb_image = np.dot(U, weight_comb)

    return comb_image
    raise NotImplementedError