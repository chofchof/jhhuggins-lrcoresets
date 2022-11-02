from __future__ import absolute_import
from warnings import warn

import numpy as np
import numpy.random as npr
from numpy.testing import *

import coresets.kernels as kernels

npr.seed(1)


import pytest
@pytest.mark.slow
def test_polynomial_kernel_equal_distributions():
    sample1 = npr.randn(10000, 2)
    sample2 = npr.randn(10000, 2)

    k = kernels.PolynomialKernel(4)
    mmd = k.estimate_mmd(sample1, sample2)
    assert_allclose(mmd, 0, atol=.2)


def test_polynomial_kernel_equal_moments():
    sample1 = npr.randn(5000, 2)
    sample2 = np.sqrt(12) * (npr.rand(5000, 2) - 0.5)

    k = kernels.PolynomialKernel(2)
    mmd = k.estimate_mmd(sample1, sample2)
    assert_allclose(mmd, 0, atol=.01)


def test_polynomial_kernel_unequal_distributions():
    sample1 = npr.randn(5000, 2)
    sample2 = np.sqrt(5) * (npr.rand(5000, 2) - 0.5)

    k = kernels.PolynomialKernel(2)
    mmd = k.estimate_mmd(sample1, sample2)
    assert_array_less(mmd, .7)
