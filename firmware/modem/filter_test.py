import matplotlib.pyplot as plt

f = 200_000
period = 1 / f

high_t = 500e-3
low_t = 500e-3

data = [180] * (high_t // period)
data += [70] * (high_t // period)


