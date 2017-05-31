from unittest import TestCase
from src.transferfunction import SISO, tf, zpk
import numpy as np


class TestSISO(TestCase):
    def test___init__(self):
        s = SISO([1, 1], [1, 0, 1])
        s1 = SISO([1, 0, 1], [1, 0, 0, 1])
        self.assertNotEqual(id(s), id(s1))
        self.assertNotEqual(id(s.num), id(s1.num))
        self.assertNotEqual(id(s.den), id(s1.den))

    def test___neg__(self):
        s = SISO([1, 1], [1, 0, 1])
        neg_s = SISO([-1, -1], [1, 0, 1])
        self.assertEqual(-s, neg_s)

    def test_feedback(self):
        s1 = SISO([1], [1, 0])
        s2 = SISO([2, 0], [1, 4, 3])
        s5 = SISO([1, 4, 3], [1, 4, 5, 0])
        self.assertEqual(s1.feedback(s2), s5)

    def test___add__(self):
        s1 = SISO([1], [1, 0])
        s2 = SISO([2, 0], [1, 4, 3])
        s3 = SISO([3, 4, 3], [1, 4, 3, 0])
        self.assertEqual(s1 + s2, s3)

    def test___mul__(self):
        s1 = SISO([1], [1, 0])
        s2 = SISO([2, 0], [1, 4, 3])
        s3 = SISO([2], [1, 4, 3])
        self.assertEqual(s1*s2, s3)

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
