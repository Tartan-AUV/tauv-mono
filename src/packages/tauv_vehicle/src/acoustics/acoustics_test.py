import numpy as np
import scipy.signal as signal
from backends.adalm_serial import AdalmSerialBackend
import matplotlib.pyplot as plt

def find_direction(times, samples):
    frequency = 40000.0
    sample_rate = 1000000.0

    reference_times = times[0]
    reference_samples = samples[0]

    aligned_samples = np.array([
        np.interp(reference_times, times[1], samples[1]),
        np.interp(reference_times, times[2], samples[2]),
        np.interp(reference_times, times[3], samples[3]),
    ])

    print(aligned_samples)

    correlations = np.array([
        signal.correlate(aligned_samples[0], reference_samples, "full"),
        signal.correlate(aligned_samples[1], reference_samples, "full"),
        signal.correlate(aligned_samples[2], reference_samples, "full"),
    ])

    lags = signal.correlation_lags(reference_samples.shape[0], reference_samples.shape[0], mode="full")

    delays = lags[np.argmax(correlations, axis=1)] / sample_rate

    print(delays * 1.0e6)


def main():
    bnd = AdalmSerialBackend()
    bnd.open()

    while True:
        times, samples = bnd.sample()

        find_direction(times, samples)


if __name__ == "__main__":
    main()