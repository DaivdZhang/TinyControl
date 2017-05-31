from unittest import TestCase
from src.pzmap import pzmap
from src.transferfunction import tf


class TestPzmap(TestCase):
    def test_pzmap(self):
        sys_ = tf([1, 2, 2], [1, 2, 1, 1])
        pzmap(sys_)
