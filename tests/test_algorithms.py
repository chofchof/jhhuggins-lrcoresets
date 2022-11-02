from __future__ import absolute_import
from warnings import warn

import numpy as np
import numpy.random as npr
from numpy.testing import *
import scipy.optimize as opt
from scipy.spatial.distance import euclidean
from scipy.sparse import csr_matrix

import coresets.algorithms as algs
import coresets._algorithms as _algs
from coresets.distributions import logistic_likelihood, logistic_likelihood_grad
from coresets.data import generate_reverse_mixture

npr.seed(1)


def _generate_cluster_data(cluster_centers, points_per_cluster,
                           return_assignments=False):
    assert len(cluster_centers.shape) == 2

    (num_clusters, data_dim) = cluster_centers.shape
    # data uniformly distributed between -1 and 1 around centers
    data = 2 * npr.rand(num_clusters * points_per_cluster, data_dim) - 1
    data = data + np.tile(cluster_centers, (points_per_cluster, 1))
    data = np.asmatrix(data)
    if return_assignments:
        cluster_assignments = np.tile(np.arange(num_clusters),
                                      points_per_cluster)
        return data, cluster_assignments
    else:
        return data


def test_calculate_cluster_information():
    cluster_centers = np.matrix([[0, 0, 0, 0],
                                 [-10, 0, 0, 0],
                                 [0, 5, 5, 0]], dtype=np.float)
    points_per_cluster = 50
    data, true_assignments = \
        _generate_cluster_data(cluster_centers, points_per_cluster, True)

    # Case: assignments not given
    sizes, means, assignments = \
        algs._calculate_cluster_information(data, cluster_centers)

    assert_array_equal(points_per_cluster, sizes)
    # three standard deviations
    tolerance = 3 / np.sqrt(3 * points_per_cluster)
    assert_allclose(cluster_centers, means, atol=tolerance)
    assert_array_equal(true_assignments, assignments)

    # Case: assignments given
    sizes, means, assignments = \
        algs._calculate_cluster_information(data, cluster_centers,
                                            true_assignments)
    assert_array_equal(points_per_cluster, sizes)
    assert_array_equal(cluster_centers, means)
    assert_array_equal(true_assignments, assignments)


def test_calculate_lr_sensitivities():
    # NB: this test will likely fail if test_calculate_cluster_information()
    # fails
    cluster_centers = np.matrix([[0, 0, 0],
                                 [2, 2, 2]], dtype=np.float)
    data = np.matrix([[0, 0, 0],
                      [2, 2, 2]], dtype=np.float)
    dist = euclidean(data[0,:], data[1,:])
    points_per_cluster = 50
    data = np.tile(data, (points_per_cluster, 1))
    R = 2.0
    sensitivity = data.shape[0] / points_per_cluster / (1 + np.exp(-R * dist))

    info = algs._calculate_cluster_information(data, cluster_centers)
    assert_allclose(cluster_centers, info[1])

    sensitivities = algs._calculate_lr_sensitivities(data, R, None, info)
    assert_allclose(sensitivities, sensitivity)


def test_assign_lr_sensitivities_cython():
    cluster_centers = np.matrix([[0, 0, 0],
                                 [-10, 0, 0],
                                 [0, 5, 5]], dtype=np.float)
    num_clusters, num_features = cluster_centers.shape
    points_per_cluster = 100
    R = 2.0

    data = _generate_cluster_data(cluster_centers, points_per_cluster)
    info = algs._calculate_cluster_information(data, cluster_centers)

    py_sensitivities = algs._calculate_lr_sensitivities(data, R, None, info)
    array_sensitivities = np.zeros(data.shape[0])

    _algs._assign_lr_sensitivities_array(data, R, None,
                                         info[0], info[1], info[2],
                                         array_sensitivities)
    csr_sensitivities = np.zeros(data.shape[0])
    _algs._assign_lr_sensitivities_csr(csr_matrix(data), R, None,
                                       info[0], info[1], info[2],
                                       csr_sensitivities)
    assert_allclose(py_sensitivities, array_sensitivities)
    assert_allclose(py_sensitivities, csr_sensitivities)


@dec.slow
def test_construct_lr_coreset_eps_delta():
    cluster_centers = np.matrix([[0, 0, 0],
                                 [-10, 0, 0],
                                 [0, 5, 5]], dtype=np.float)
    num_clusters, num_features = cluster_centers.shape
    eps = .24
    delta = .01
    expected_sensitivity = num_clusters * np.exp(np.sqrt(num_features / 3.0))
    expected_coreset_size =  int(4 * expected_sensitivity / eps**2
                                   * ((num_features + 1)
                                      * np.log(expected_sensitivity)
                                      - np.log(delta)) + 1)

    points_per_cluster = expected_coreset_size * 3 / num_clusters
    if points_per_cluster * num_clusters > 4e5:
        warn("Testing coreset construction on %d samples, "
             "so test may be slow." % (points_per_cluster * num_clusters),
             RuntimeWarning)

    # Should create a coreset
    data = _generate_cluster_data(cluster_centers, points_per_cluster)
    coreset, weights, sensitivities, success = \
        algs.construct_lr_coreset(data, cluster_centers,
                                  output_size_param=(eps, delta),
                                  verbose=True)
    assert success
    mean_sensitivity = np.mean(sensitivities)
    assert_allclose(expected_sensitivity, mean_sensitivity, rtol=.05)
    assert_allclose(expected_coreset_size, coreset.shape[0], rtol=.05)

    close_count = 0
    tests = 100
    for i in range(tests):
        theta = 2 * np.random.rand(num_features) - 1
        data_lik = logistic_likelihood(theta, data)
        core_lik = logistic_likelihood(theta, coreset, weights)
        close_count += abs((data_lik - core_lik) / data_lik) <= eps
    assert_array_equal(tests, close_count)

    # Should not create a coreset
    eps *= .01
    coreset, weights, sensitivities, success = \
        algs.construct_lr_coreset(data, cluster_centers,
                                  output_size_param=(eps, delta),
                                  verbose=True)
    assert not success
    assert_array_equal(1.0, weights)
    assert_allclose(data, coreset)


def test_construct_lr_coreset_coreset_size():
    cluster_centers = np.matrix([[0, 0, 0],
                                 [-10, 0, 0],
                                 [0, 5, 5]], dtype=np.float)
    num_clusters, num_features = cluster_centers.shape
    coreset_size = 300

    points_per_cluster = coreset_size * 10 / num_clusters
    if points_per_cluster * num_clusters > 4e5:
        warn("Testing coreset construction on %d samples, "
             "so test may be slow." % (points_per_cluster * num_clusters),
             RuntimeWarning)

    # Should create a coreset
    data, cluster_assignments = \
        _generate_cluster_data(cluster_centers, points_per_cluster, True)
    coreset, weights, mean_sensitivity, success = \
        algs.construct_lr_coreset(data, cluster_centers, cluster_assignments,
                                  output_size_param=coreset_size,
                                  verbose=True)
    assert success
    assert_array_equal(coreset_size, coreset.shape[0])

    # Should not create a coreset
    coreset_size = data.shape[0] * 2 / 3
    coreset, weights, mean_sensitivity, success = \
        algs.construct_lr_coreset(data, cluster_centers,
                                  output_size_param=coreset_size,
                                  verbose=True)
    assert not success
    assert_array_equal(1.0, weights)
    assert_allclose(data, coreset)


def _generate_reverse_mixture_data(N, dim):
    covar = np.eye(dim)
    means = np.zeros((2, dim))
    pos_prob = .5
    means[0, :dim/2] = 1.
    means[1, dim/2:] = 1.
    X, y = generate_reverse_mixture(N, pos_prob, means, covar)
    return y[:, np.newaxis] * X
