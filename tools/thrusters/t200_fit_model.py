import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
import csv


class T200Curve:
    def __init__(self, path):
        with open(path, "r") as f:
            df = pd.read_csv(f, header=0, sep="\s+")
        table = df.to_numpy()
        pwm_us = table[:, 0]
        rpm = table[:, 1]
        radps = rpm * (2.0 * np.pi / 60.0)
        force = table[:, 5] * 9.81
        fwd_msk = force > 0.0
        rev_msk = force < 0.0
        self.pwm_us_fwd = pwm_us[fwd_msk]
        self.pwm_us_rev = pwm_us[rev_msk]
        self.radps_fwd = radps[fwd_msk]
        # bluerobotics tables report absolute RPM
        self.radps_rev = -radps[rev_msk]
        self.force_fwd = force[fwd_msk]
        self.force_rev = force[rev_msk]
        V_bus = table[0, 3]

        # estimate deadband
        self.deadband = (np.max(self.radps_rev), np.min(self.radps_fwd))

        # pwm period
        min_pwm = np.min(self.pwm_us_rev)
        max_pwm = np.max(self.pwm_us_fwd)
        center_pwm = (max_pwm + min_pwm) / 2.0
        self.V_eff_fwd = V_bus * (self.pwm_us_fwd - center_pwm) / (max_pwm - center_pwm)
        self.V_eff_rev = V_bus * (self.pwm_us_rev - center_pwm) / (center_pwm - min_pwm)


def estimate_thrust_coeff(radps, F):
    radps_sq = radps * np.abs(radps)
    K_F, res, _, _ = np.linalg.lstsq(radps_sq, F)
    return K_F[0], res


def estimate_rotor_dynamics(radps, V_eff):
    radps_sq = radps * np.abs(radps)
    A = np.hstack([radps, radps_sq])
    (K_v1, K_v2), res, _, _ = np.linalg.lstsq(A, V_eff)
    return K_v1, K_v2, res


if __name__ == "__main__":
    t200 = Path.cwd() / "data" / "t200"

    bus_voltages = [10, 12, 14, 16, 18, 20]
    curves = [T200Curve(t200 / f"{v}v.csv") for v in bus_voltages]

    # Print deadbands
    print("=== Deadbands ===")
    for v, c in zip(bus_voltages, curves):
        dbmin, dbmax = c.deadband
        print(f"V_bus = {v}V: {dbmin}-{dbmax}")

    # Estimate thrust coefficient
    # Stack all rpm + force measurements
    radps_fwd = np.concatenate([c.radps_fwd for c in curves], dtype=np.float64)[:, np.newaxis]
    F_fwd = np.concatenate([c.force_fwd for c in curves], dtype=np.float64)
    K_F_fwd, res_fwd = estimate_thrust_coeff(radps_fwd, F_fwd)

    radps_rev = np.concatenate([c.radps_rev for c in curves], dtype=np.float64)[:, np.newaxis]
    F_rev = np.concatenate([c.force_rev for c in curves], dtype=np.float64)
    K_F_rev, res_rev = estimate_thrust_coeff(radps_rev, F_rev)
    # solve LSq
    print("\n=== Thrust Coefficients ===")
    print(f"Forward: thrust coefficients K_F = {K_F_fwd:.3e}, sum of radps residuals = {res_fwd}")
    print(f"Forward: thrust coefficients K_F = {K_F_rev:.3e}, sum of radps residuals = {res_rev}")

    # Plot results
    fig, ax = plt.subplots()
    c = curves[3]
    radps = np.concatenate([c.radps_rev, c.radps_fwd])
    F = np.concatenate([c.force_rev, c.force_fwd])
    F_est = np.concatenate(
        [c.radps_rev * np.abs(c.radps_rev) * K_F_rev, K_F_fwd * c.radps_fwd**2]
    )
    ax.plot(radps, F, linestyle="--", label=f"16v, GT")
    ax.plot(radps, F_est, linestyle="-", label=f"16v, Model")
    ax.legend()
    ax.set_xlabel("rad/s")
    ax.set_ylabel("Force")
    plt.show()

    # Estimate rotor dynamics
    # Assume that PWM period cycle is mapped linearly to effective voltage
    max_pwm_period = max([np.max(c.pwm_us_fwd) for c in curves])
    min_pwm_period = min([np.min(c.pwm_us_rev) for c in curves])
    print("\n=== PWM periods ===")
    print(f"max_pwm_period = {max_pwm_period}")
    print(f"min_pwm_period = {min_pwm_period}")

    # Stack rpms and effecive voltages
    # Use only 14, 16, 18 V for 4S
    curves_used = curves[2:5]
    V_eff = [c.V_eff_fwd for c in curves_used]
    V_eff.extend([c.V_eff_rev for c in curves_used])
    V_eff = np.concatenate(V_eff)
    radps = [c.radps_fwd for c in curves_used]
    radps.extend([c.radps_rev for c in curves_used])
    radps = np.concatenate(radps)[:, np.newaxis]
    K_v1, K_v2, res = estimate_rotor_dynamics(radps, V_eff)
    print("\n=== Rotor dynamics ===")
    print(f"K_v1 = {K_v1:.3e}")
    print(f"K_v2 = {K_v2:.3e}")
    print(f"residuals = {res}")

    # Plot
    fig, ax = plt.subplots()

    c = curves[3]
    v_bus = bus_voltages[3]

    radps = np.concatenate([c.radps_rev, c.radps_fwd])
    V_eff = np.concatenate([c.V_eff_rev, c.V_eff_fwd])

    V_eff_est = c.radps_fwd * K_v1 + c.radps_fwd * np.abs(c.radps_fwd) * K_v2

    radps_est_fwd = (-K_v1 + np.sqrt(K_v1**2 + 4.0 * K_v2 * c.V_eff_fwd)) / (
        2.0 * K_v2
    )
    radps_est_rev = -(
        -K_v1 + np.sqrt(K_v1**2 + 4.0 * K_v2 * (-c.V_eff_rev))
    ) / (2.0 * K_v2)
    radps_est = np.concatenate([radps_est_rev, radps_est_fwd])

    ax.plot(V_eff, radps, linestyle="--", label=f"16v, GT")
    ax.plot(V_eff, radps_est, linestyle="-", label=f"16v, Model")
    ax.legend()
    ax.set_xlabel("Effective Voltage")
    ax.set_ylabel("rad/s")
    plt.show()
