from unittest import TestCase
from src.transferfunction import SISO, tf


class TestTransferFunction(TestCase):

    def test_create(self):
        s = SISO([1, 1], [1, 0, 1])
        s1 = SISO([1, 0, 1], [1, 0, 0, 1])
        print("s + s: %s" % (s + s))
        print(s1 * s)
        self.assertNotEqual(id(s), id(s1))
        self.assertNotEqual(id(s.num), id(s1.num))
        self.assertNotEqual(id(s.den), id(s1.den))
        s = SISO([1], [1, 1])
        print(s.feedback())

    def test_tf(self):
        tf(1)
