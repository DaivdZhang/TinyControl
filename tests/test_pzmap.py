from unittest import TestCase
from tcontrol.pzmap import pzmap
from tcontrol.transferfunction import tf


class TestPzmap(TestCase):
    def test_pzmap(self):
        sys_ = tf([1, 2, 2], [1, 2, 1, 1])
        pzmap(sys_)
