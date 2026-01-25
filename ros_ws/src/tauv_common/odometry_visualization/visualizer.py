import matplotlib.pyplot as plt
import pandas as pd

# Load your CSV
df = pd.read_csv("./formatted_csvs/sim_2026.01.24_20.51.41.csv")

# ---- Trajectory plot ----
plt.figure(figsize=(6, 6))
plt.plot(df["x"], df["y"], marker='.', linestyle='-', label="EKF path")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("EKF Trajectory")
plt.axis("equal")
plt.grid(True)
plt.legend()

# ---- Pose over time ----
plt.figure()
plt.plot(df["time"], df["x"], label="X")
plt.plot(df["time"], df["y"], label="Y")
plt.plot(df["time"], df["yaw"], label="Yaw")
# plt.plot(df["time"], df["z"], label="Z")
plt.xlabel("Time [s]")
plt.ylabel("Position [m]")
plt.title("EKF Pose")
plt.grid(True)
plt.legend()

# # ---- Yaw over time ----
# plt.figure()
# plt.plot(df["time"], df["yaw"], label="Yaw [rad]")
# plt.xlabel("Time [s]")
# plt.ylabel("Yaw [rad]")
# plt.title("EKF Yaw")
# plt.grid(True)
# plt.legend()

# ---- Linear velocities over time ----
plt.figure()
plt.plot(df["time"], df["vx"], label="Vx")
plt.plot(df["time"], df["vy"], label="Vy")
plt.plot(df["time"], df["vz"], label="Vz")
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")
plt.title("EKF Linear Velocities")
plt.grid(True)
plt.legend()

# ---- Covariance over time ----
plt.figure()
plt.plot(df["time"], df["pose_cov_0"], label="x variance")
plt.plot(df["time"], df["pose_cov_7"], label="y variance")
# plt.plot(df["time"], df["cov_yaw"], label="yaw variance")
plt.xlabel("Time [s]")
plt.ylabel("Covariance")
plt.title("EKF Pose Covariance")
plt.yscale("log")  # Log scale helps visualize differences
plt.grid(True)
plt.legend()

plt.show()
