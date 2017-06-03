from unittest import TestCase
from tcontrol.poly import *
import numpy as np


class TestPoly(TestCase):
    def test_conv(self):
        ret = conv([1, -1], [1, 1])
        self.assertEqual(all(ret == np.array([1, 0, -1])), True)
        ret = conv([1, 1], [1, 1], [1, 1])
        self.assertEqual(all(ret == np.array([1, 3, 3, 1])), True)

    def test_poly(self):
        ret = poly([1, -1])
        self.assertEqual(all(ret == np.array([1, 0, -1])), True)
        ret = poly([1j, -1j, -1])
        self.assertEqual(all(ret == np.array([1, 1, 1, 1])), True)

    def test_roots(self):
        ret = roots([1, 0, -1])
        self.assertEqual(all(ret == np.array([1, -1])), True)
