import timeit


# run step, impulse and ramp
def run1():
    r1 = timeit.timeit("tc.step(system, np.linspace(0, 10, 1000),plot=False)",
                       "import tcontrol as tc;import numpy as np;"
                       "system = tc.tf([5, 25, 30], [1, 6, 10, 8])", number=100)
    r2 = timeit.timeit("tc.impulse(system, np.linspace(0, 10, 1000),plot=False)",
                       "import tcontrol as tc;import numpy as np;"
                       "system = tc.tf([5, 25, 30], [1, 6, 10, 8])", number=100)
    r3 = timeit.timeit("tc.ramp(system, np.linspace(0, 10, 1000), plot=False)",
                       "import tcontrol as tc;import numpy as np;"
                       "system = tc.tf([5, 25, 30], [1, 6, 10, 8])", number=100)
    print("step: {0:.5f} ms impulse: {0:.5f} ms ramp: {0:.5f} ms".format(r1*10, r2*10, r3*10))


# run rlocus
def run2():
    timer = timeit.Timer(
        "rlocus(system, np.linspace(0, 100, 10000), xlim=[-5, 0.5], plot=False)",
        "from tcontrol import tf, rlocus; system = tf([0.5, 1], [0.5, 1, 1]);"
        "import numpy as np")
    r1 = timer.timeit(100)
    print("{0:.5f} ms\n".format(r1*10))


# run bode
def run3():
    timer = timeit.Timer("tc.bode(system, plot=False)",
                         "import tcontrol as tc; system = tc.zpk([], [0, -1, -2], 2)")
    r1 = timer.timeit(100)
    print("{0:.5f} ms\n".format(r1*10))


if __name__ == "__main__":
    run1()
    run2()
    run3()
