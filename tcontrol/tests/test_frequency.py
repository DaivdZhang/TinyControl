from unittest import TestCase
from tcontrol import frequency
import tcontrol as tc


class TestFrequency(TestCase):
    def test_nyquist(self):
        frequency.nyquist(tc.tf([0.5], [1, 2, 1, 0.5]))

    def test_bode(self):
        frequency.bode(tc.zpk([], [0, -1, -2], 2))
        frequency.bode(tc.tf([1], [1, 1]))
