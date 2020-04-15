import numpy as np
import logging

logging.basicConfig()
logger = logging.getLogger('wss_utils')

def extract_vectors(polydata):
    """
        Extract vector from separate columns on polydata and stack them to (n,3) vector 
    """
    u = polydata["u"]
    v = polydata["v"]
    w = polydata["w"]
    vector = np.stack((u,v,w), axis=-1)
    return vector

def get_orthogonal_vectors(vectors, point_normals):
    logger.debug("Get orthogonal vectors")
    logger.debug('Actual vector {}'.format(vectors[0:2]))
    
    # Calculate the scalar v.n
    c = vectors * point_normals
    c = np.sum(c, axis=1)

    # Get normal and tangent vector
    normal_vectors = c[:,np.newaxis] * point_normals
    tangent_vectors = vectors - normal_vectors

    logger.debug('Normal vector {}'.format(normal_vectors[0:2]))
    logger.debug('Tangent vector {}'.format(tangent_vectors[0:2]))

    return normal_vectors, tangent_vectors

def get_vector_magnitude(vectors):
    """
        Calculate vector magnitude |v| from an array of (n,3)
    """
    c = vectors * vectors
    c = np.sum(c, axis=1)
    c = c ** 0.5
    return c

def _calculate_gradient_with_values(pc0_tangent_mag, pc1_tangent_mag, pc2_tangent_mag, inward_distance, use_parabolic):
    """
        Fitting polynomial for multiple rows of set of points, then calculate the gradient
        Based on: https://stackoverflow.com/questions/20202710/numpy-polyfit-and-polyval-in-multiple-dimensions
    """
    logger.info("Calculating gradient for {} points".format(pc1_tangent_mag.shape))
    # Prepare to calculate the slopes for each points
    x = np.array([0,1,2]) # We have 3 points to evaluate
    x = x * inward_distance # Get the correct distance scaling
    
    # Stack them so it has (n x (v0, v1, v2))
    y = np.stack((pc0_tangent_mag, pc1_tangent_mag, pc2_tangent_mag), axis=1)
    
    # Calculate an n-1 polynomial 
    y = np.transpose(y)
    z = np.polynomial.polynomial.polyfit(x, y, len(x)-1)
    
    # Create new X with more points to obtain a smooth curve
    if use_parabolic:
        x_new = np.linspace(x[0], x[-1], len(x) * 5)
    else:
        x_new = x
    
    # Evaluate the fitted function with an evenly distributed x
    y_new = np.polynomial.polynomial.polyval(x_new, z)
    
    # Get all the gradient at once
    gg = np.gradient(y_new, x_new, axis=1)

    # Only return the gradient at the wall
    return gg[:,0], x_new, y_new

def calculate_gradient(pc0_tangent_mag, pc1_tangent_mag, pc2_tangent_mag, inward_distance, use_parabolic=True):
    gradients, x_, y_ =_calculate_gradient_with_values(pc0_tangent_mag, pc1_tangent_mag, pc2_tangent_mag, inward_distance, use_parabolic)
    # Only return the gradients, x_ and y_ for test purpose ONLY
    return gradients