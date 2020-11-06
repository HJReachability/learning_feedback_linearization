#!/usr/bin/env python

import numpy as np

class ReferenceGenerator(object):
    def __init__(self):
        pass

    def __call__(self, x):
        """ Return a reference trajectory which starts from x """
        raise NotImplementedError()
