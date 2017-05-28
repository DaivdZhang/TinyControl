from unittest import TestCase
from src.transferfunction import SISO, tf


class TestTransferFunction(TestCase):
    def test___init__(self):
        s = SISO([1, 1], [1, 0, 1])
        s1 = SISO([1, 0, 1], [1, 0, 0, 1])
        self.assertNotEqual(id(s), id(s1))
        self.assertNotEqual(id(s.num), id(s1.num))
        self.assertNotEqual(id(s.den), id(s1.den))

    def test_operation(self):
        s1 = SISO([1], [1, 0])
        s2 = SISO([2, 0], [1, 4, 3])
        s3 = SISO([3, 4, 3], [1, 4, 3, 0])
        s4 = SISO([2], [1, 4, 3])
        s5 = SISO([1, 4, 3], [1, 4, 5, 0])
        self.assertEqual(s1 + s2, s3)
        self.assertEqual(s1 * s2, s4)
        self.assertEqual(s1.feedback(s2), s5)

    def test_tf(self):
        self.assertEqual(tf([1], [1, 0]), SISO([1], [1, 0]))
        self.assertEqual(tf(SISO([1], [1, 0])), SISO([1], [1, 0]))
        self.assertEqual(tf(SISO([1], [1, 0])), SISO([2], [2, 0]))
