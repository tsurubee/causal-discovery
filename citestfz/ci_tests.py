#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.stats import norm

from .gauss import zstat


def ci_test_gauss(data_matrix, x, y, s, **kwargs):

    assert 'corr_matrix' in kwargs
    cm = kwargs['corr_matrix']
    n = data_matrix.shape[0]

    z = zstat(x, y, list(s), cm, n)
    p_val = 2.0 * norm.sf(np.absolute(z))
    return p_val


