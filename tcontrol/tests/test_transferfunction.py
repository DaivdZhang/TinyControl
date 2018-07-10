from unittest import TestCase
from tcontrol.transferfunction import TransferFunction, tf, zpk
from ..exception import *
import numpy as np


class TestTransferFunction(TestCase):
    def setUp(self):
        self.s1 = TransferFunction([1, 1], [1, 0, 1])
        self.s2 = TransferFunction([1, 0, 1], [1, 0, 0, 1])
        self.s3 = TransferFunction([1], [1, 0])
        self.s4 = TransferFunction([2, 0], [1, 4, 3])
        self.s5 = TransferFunction([1, 4, 3], [1, 4, 5, 0])
        self.s6 = TransferFunction([3, 4, 3], [1, 4, 3, 0])
        self.s7 = TransferFunction([2], [1, 4, 3])
        self.s8 = TransferFunction([1], [1, 2, 1])
        self.s9 = TransferFunction([1, 2, 3, 0], [1, 2, 2, 2, 1])

        self.s = TransferFunction([1, 1], [1, 0, 1])
        self.neg_s = TransferFunction([-1, -1], [1, 0, 1])

    def test___init__(self):
        self.assertNotEqual(id(self.s1), id(self.s2))
        self.assertNotEqual(id(self.s1.num), id(self.s2.num))
        self.assertNotEqual(id(self.s1.den), id(self.s2.den))

    def test___neg__(self):
        self.assertEqual(-self.s, self.neg_s)

    def test_feedback(self):
        self.assertEqual(self.s3.feedback(self.s4), self.s5)

    def test___add__(self):
        self.assertEqual(self.s3 + self.s4, self.s6)

    def test___mul__(self):
        self.assertEqual(self.s3*self.s4, self.s7)

    def test___sub__(self):
        self.assertEqual(self.s1 - self.s8, self.s9)

    def test_pole(self):
        s = TransferFunction([1, 2], [1, 2, 1])
        self.assertEqual((s.pole() == np.roots([1, 2, 1])).all(), True)
        s = TransferFunction([1, 2], [1, 1, 1, -1])
        r = s.pole() == np.roots([1, 1, 1, -1])
        self.assertEqual(all(r), True)

    def test_zero(self):
        s = TransferFunction([1, 2], [1, 2, 1])
        self.assertEqual((s.zero() == np.roots([1, 2])).all(), True)

    def test_discretize(self):
        discretize = TransferFunction.discretize
        # test Tustin
        self.assertEqual(discretize(self.s1, 1, 'Tustin'), tf([3, 2, -1], [5, -6, 5], 1))

        # test matched
        d_sys = discretize(self.s1, 1, 'matched')
        error = np.abs(d_sys.num - np.array([1.4545, -0.5351]))
        self.assertTrue(np.all(np.less_equal(error, 1e-4)))

    def test_tf(self):
        self.assertEqual(tf([1], [1, 0]), TransferFunction([1], [1, 0]))
        self.assertEqual(tf(TransferFunction([1], [1, 0])), TransferFunction([1], [1, 0]))
        s1 = tf(TransferFunction([1], [1, 0]))
        s2 = TransferFunction([2], [2, 0])
        self.assertEqual(s1, s2)
        s3 = tf([1, 1], [1, 0, -1])
        s4 = tf([1], [1, -1])
        self.assertEqual(s3, s4)

    def test_zpk(self):
        s1 = TransferFunction([5, 5], [1, 0, -4])
        s2 = zpk([-1], [-2, 2], 5)
        self.assertEqual(s1, s2)

    def test_bad_input(self):
        self.assertRaises(WrongNumberOfArguments, tf, *[[1], 2, 3, 4])
        self.assertRaises(TypeError, tf, [1, 3, 4, 5])
