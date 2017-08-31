from unittest import TestCase
from tcontrol.pzmap import pzmap
from tcontrol.transferfunction import tf
from tests.data_generator import SYSTEMS
import numpy as np


class TestPzmap(TestCase):
    def test_pzmap(self):
        sys_ = tf([1, 2, 2], [1, 2, 1, 1])
        self.assertTrue(all(np.equal(pzmap(sys_)[0], np.roots(sys_.den))))
        self.assertTrue(all(np.equal(pzmap(sys_)[1], np.roots(sys_.num))))

    def test_pzmap_random_data(self):
        sys_ = [tf(*i) for i in SYSTEMS]
        for i in sys_:
            self.assertTrue(all(np.equal(pzmap(i)[0], np.roots(i.den))))
            self.assertTrue(all(np.equal(pzmap(i)[1], np.roots(i.num))))
