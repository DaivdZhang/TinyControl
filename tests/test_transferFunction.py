from unittest import TestCase
from src.transferfunction import SISO


class TestTransferFunction(TestCase):

    def test_create(self):
        s = SISO([1, 1], [1, 0, 1])
        print(s)
        s1 = SISO([1, 0, 1], [1, 0, 0, 1])
        print(s1)
        self.assertNotEqual(id(s), id(s1))
        self.assertNotEqual(id(s.num), id(s1.num))
        self.assertNotEqual(id(s.den), id(s1.den))
        print(s *s1)
