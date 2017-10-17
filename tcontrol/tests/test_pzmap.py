from unittest import TestCase

import numpy as np

from tcontrol.pzmap import pzmap
from tcontrol.tests.data_generator import SYSTEMS
from tcontrol.transferfunction import tf


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
