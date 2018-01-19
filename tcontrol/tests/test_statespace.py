from unittest import TestCase
import numpy as np
from tcontrol.statespace import *
from tcontrol.transferfunction import tf, ss2tf


class TestStateSpace(TestCase):
    def setUp(self):
        self.A = np.matrix([[0, 1], [-4, -0.5]])
        self.B = np.matrix([[0.], [1.]])
        self.C = np.matrix([[4., 0.]])
        self.D = np.matrix([0.])
        self.tf_ = tf([4], [1, 0.5, 4])
        self.ss_ = StateSpace(self.A, self.B, self.C, self.D)

    def test___init__(self):
        ss = StateSpace(self.A, self.B, self.C, self.D)
        self.assertEqual(ss.A is self.A, True)
        StateSpace(self.A, self.B, self.C, 0)

    def test___str__(self):
        print(self.ss_)

    def test___add__(self):
        ss = StateSpace(self.A, self.B, self.C, self.D)
        ss = ss + ss

    def test___mul__(self):
        # A_ = np.mat([[0, 1, 1, 1], [-4, -0.5, 1, 2], [0, 0, 0, -4], ])
        # print(StateSpace.dual_system(self.ss_)*self.ss_)
        print(self.ss_*StateSpace.dual_system(self.ss_))

    def test_pole(self):
        ss = StateSpace(self.A, self.B, self.C, self.D)
        pass

    def test_controllability(self):
        # print(self.ss_.controllability())
        pass

    def test_is_controllable(self):
        self.assertTrue(self.ss_.is_controllable())

    def test_observability(self):
        # print(self.ss_.observability())
        pass

    def test_is_observable(self):
        self.assertTrue(self.ss_.is_observable())

    def test_dual_system(self):
        _ = StateSpace.dual_system(self.ss_)
        self.assertTrue(np.all(np.equal(_.A.T, self.ss_.A)))
        self.assertTrue(np.all(np.equal(_.C.T, self.ss_.B)))
        self.assertTrue(np.all(np.equal(_.B.T, self.ss_.C)))

    def test_to_controllable_form(self):
        T = self.ss_.to_controllable_form()
        print(T.I*self.A*T)
        print(T.I*self.B)
        print(self.C*T)

    def test_ss(self):
        pass

    def test_tf2ss(self):
        ss = tf2ss(self.tf_)
        self.assertTrue(np.all(ss.A == self.A))
        self.assertTrue(np.all(ss.B == self.B))
        self.assertTrue(np.all(ss.C == self.C))

    def test_ss2tf(self):
        self.assertEqual(ss2tf(self.ss_), self.tf_)

    def test_continuous_to_discrete(self):
        A = np.array([[0, 1], [0, -2]])
        B = np.array([[0],[1]])
        sys_ = StateSpace(A, B, self.C, self.D)
        d_sys_ = continuous_to_discrete(sys_, 0.05)
        continuous_to_discrete(d_sys_, 0.01)
