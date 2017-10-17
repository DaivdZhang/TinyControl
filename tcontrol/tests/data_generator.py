import random
import time

random.seed(time.time())
SYSTEMS = []


def gen():
    return ([random.random()*10 for _ in range(2)],
            [random.random()*10 for _ in range(3)])


for i in range(3):
    SYSTEMS.append(gen())
