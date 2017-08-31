from unittest import TestCase
from tcontrol.transferfunction import SISO, tf, zpk
import numpy as np


class TestSISO(TestCase):
    def setUp(self):
        self.s1 = SISO([1, 1], [1, 0, 1])
        self.s2 = SISO([1, 0, 1], [1, 0, 0, 1])
        self.s3 = SISO([1], [1, 0])
        self.s4 = SISO([2, 0], [1, 4, 3])
        self.s5 = SISO([1, 4, 3], [1, 4, 5, 0])
        self.s6 = SISO([3, 4, 3], [1, 4, 3, 0])
        self.s7 = SISO([2], [1, 4, 3])
        self.s8 = SISO([1], [1, 2, 1])
        self.s9 = SISO([1, 2, 3, 0], [1, 2, 2, 2, 1])

        self.s = SISO([1, 1], [1, 0, 1])
        self.neg_s = SISO([-1, -1], [1, 0, 1])


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
        self.assertEqual(self.s3 * self.s4, self.s7)

    def test___sub__(self):
        self.assertEqual(self.s1 - self.s8, self.s9)

    def test_pole(self):
        s = SISO([1, 2], [1, 2, 1])
        self.assertEqual((s.pole() == np.roots([1, 2, 1])).all(), True)
        s = SISO([1, 2], [1, 1, 1, -1])
        r = s.pole() == np.roots([1, 1, 1, -1])
        self.assertEqual(all(r), True)

    def test_zero(self):
        s = SISO([1, 2], [1, 2, 1])
        self.assertEqual((s.zero() == np.roots([1, 2])).all(), True)


class TestTransferFunction(TestCase):
    def test_tf(self):
        self.assertEqual(tf([1], [1, 0]), SISO([1], [1, 0]))
        self.assertEqual(tf(SISO([1], [1, 0])), SISO([1], [1, 0]))
        s1 = tf(SISO([1], [1, 0]))
        s2 = SISO([2], [2, 0])
        self.assertEqual(s1, s2)

    def test_zpk(self):
        s1 = SISO([5, 5], [1, 0, -4])
        s2 = zpk([-1], [-2, 2], 5)
        self.assertEqual(s1, s2)

    def test_bad_input(self):
        self.assertRaises(ValueError, tf, *[[1], 2, 3, 4])
        self.assertRaises(TypeError, tf, [1, 3, 4, 5])
