import tcontrol as tc
import cProfile

system = tc.tf([10], [0.1, 1, 10])


def main():
    cProfile.run("tc.step(system, plot=False)", sort="cumulative")
    cProfile.run("tc.impulse(system, plot=False)", sort="cumulative")
    cProfile.run("tc.ramp(system, plot=False)", sort="cumulative")
    cProfile.run("tc.pzmap(system, plot=False)", sort="cumulative")
    cProfile.run("tc.rlocus(system, plot=False)", sort="cumulative")
    cProfile.run("tc.bode(system, plot=False)", sort="cumulative")
    cProfile.run("tc.nyquist(system, plot=False)", sort="cumulative")


if __name__ == '__main__':
    main()
