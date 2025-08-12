import csv
import numpy as np
from pathlib import Path


if __name__ == "__main__":
    csv_path = Path(__file__).parent / "data" / "t200-spec-16v.csv"

    rpm = []
    force_kgf = []
    
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            rpm.append(float(row[1]))
            force_kgf.append(float(row[5]) * 9.81)

    rpm = np.array(rpm)
    force_kgf = np.array(force_kgf)

    # Get deadband 
    # Find the smallest and largest index in rpm where the value is zero
    zero_indices = np.where(rpm == 0)[0]
    if zero_indices.size > 0:
        deadband_low = zero_indices[0] - 1
        deadband_high = zero_indices[-1] + 1
    else:
        raise ValueError("No deadband found")

    omega_fwd = rpm[deadband_high:, np.newaxis] * (2 * np.pi / 60)
    omega_rev = -rpm[:deadband_low, np.newaxis] * (2 * np.pi / 60)
    force_fwd = force_kgf[deadband_high:, np.newaxis] * 9.81
    force_rev = force_kgf[:deadband_low, np.newaxis] * 9.81

    k_fwd, *_ = np.linalg.lstsq(omega_fwd ** 2, force_fwd, rcond=None)
    k_rev, *_ = np.linalg.lstsq(omega_rev ** 2, force_rev, rcond=None)

    print(f"k_fwd: {k_fwd[0]} N/(rad/s)^2")
    print(f"k_rev: {k_rev[0]} N/(rad/s)^2")

    omega_deadband_low = omega_rev[-1]
    omega_deadband_high = omega_fwd[0]

    print(f"deadband_low: {omega_deadband_low} rad/s")
    print(f"deadband_high: {omega_deadband_high} rad/s")

    import matplotlib.pyplot as plt

    # Flow overview:
    # Plot original data and fitted quadratic curves for forward and reverse

    # Plot original data
    plt.figure(figsize=(10, 6))
    plt.scatter(omega_fwd, force_fwd, s=10, color='black', label='Measured Data')
    plt.scatter(omega_rev, force_rev, s=10, color='black', label='Measured Data')

    # Plot fitted quadratic for forward region
    force_fwd_fit = k_fwd[0] * (omega_fwd ** 2)
    plt.plot(omega_fwd, force_fwd_fit, color='blue', label='Fitted Forward Quadratic')

    # Plot fitted quadratic for reverse region
    force_rev_fit = k_rev[0] * (omega_rev ** 2)
    plt.plot(omega_rev, force_rev_fit, color='red', label='Fitted Reverse Quadratic')

    # Add deadband region shading for clarity
    plt.axvspan(rpm[deadband_low], rpm[deadband_high], color='gray', alpha=0.2, label='Deadband')

    plt.xlabel("RPM")
    plt.ylabel("Force (N)")
    plt.title("T200 Thruster: Measured vs Fitted Quadratic Force Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
