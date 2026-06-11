#python "C:\Users\SaiPC\OneDrive\Documents\Missile Launch\Missile Defense Program.py"

import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter

print("RUNNING NEWEST VERSION")

# ── Physics ───────────────────────────────────────────────────────────────────
g = 9.81          # m/s²

# ── Simulation ────────────────────────────────────────────────────────────────
dt         = 0.04   # s
totalTime  = 200    # s
NUM_RUNS   = 100

# ── Target initial conditions ─────────────────────────────────────────────────
x0, y0   = 0.0, 0.0
vx0, vy0 = 400.0, 700.0

# ── Radar ─────────────────────────────────────────────────────────────────────
measurementSigma = 50   # m

# ── Interceptor ───────────────────────────────────────────────────────────────
interceptor_speed = 600   # m/s
intercept_radius  = 75    # m
launch_delay      = 8     # s
N_pn              = 3
max_accel         = 500   # m/s²


# ── Helpers ───────────────────────────────────────────────────────────────────

def ballistic_position(px, py, pvx, pvy, tau):
    return px + pvx * tau, py + pvy * tau - 0.5 * g * tau**2


def predict_ground_impact(px, py, pvx, pvy):
    disc = pvy**2 + 2.0 * g * py
    if disc < 0:
        return None, None
    t_impact = (pvy + np.sqrt(disc)) / g
    if t_impact <= 0:
        return None, None
    return px + pvx * t_impact, t_impact


def predict_intercept_point(int_x, int_y, int_speed,
                            tgt_x, tgt_y, tgt_vx, tgt_vy,
                            search_steps=500, max_tof=120.0):
    for tau in np.linspace(0.1, max_tof, search_steps):
        fx, fy = ballistic_position(tgt_x, tgt_y, tgt_vx, tgt_vy, tau)
        if fy < 0:
            break
        dist = np.sqrt((fx - int_x)**2 + (fy - int_y)**2)
        if dist / tau <= int_speed:
            return fx, fy
    return tgt_x, tgt_y


def make_kf():
    kf = KalmanFilter(dim_x=6, dim_z=2)
    kf.x = np.array([[x0], [y0], [vx0], [vy0], [0.], [-g]])
    kf.F = np.array([
        [1, 0, dt, 0,  0.5*dt**2, 0        ],
        [0, 1, 0,  dt, 0,         0.5*dt**2],
        [0, 0, 1,  0,  dt,        0        ],
        [0, 0, 0,  1,  0,         dt       ],
        [0, 0, 0,  0,  1,         0        ],
        [0, 0, 0,  0,  0,         1        ]
    ])
    kf.H = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0]])
    kf.R = np.eye(2) * measurementSigma**2
    kf.P = np.eye(6) * 1000
    kf.Q = np.eye(6) * 0.1
    return kf


# ── Monte Carlo results arrays ────────────────────────────────────────────────
mc_intercepted      = []
mc_avg_track_err    = []
mc_max_track_err    = []
mc_intercept_x      = []
mc_intercept_y      = []

# Save the last run for plotting
last = {}

# ── Monte Carlo loop ──────────────────────────────────────────────────────────
for run in range(NUM_RUNS):

    # Per-run state
    x, y, vx, vy = x0, y0, vx0, vy0
    prev_vy = vy
    kf = make_kf()

    int_x, int_y    = 0.0, 0.0
    int_active      = False
    missile_heading = 0.0
    prev_los_angle  = 0.0

    apogee_reached  = False
    apogee_alt      = 0.0
    apogee_t        = 0.0

    intercepted        = False
    ground_impact      = False
    predicted_impact_x = None
    intercept_idx      = None

    xH, yH         = [], []
    mxH, myH       = [], []
    exH, eyH       = [], []
    int_xH, int_yH = [], []
    track_err      = []

    for t in np.arange(0, totalTime, dt):

        xH.append(x); yH.append(y)

        # Apogee detection
        if not apogee_reached and prev_vy > 0 and vy <= 0:
            apogee_reached = True
            apogee_alt, apogee_t = y, t

        prev_vy = vy

        # Radar measurement
        mx = x + np.random.normal(0, measurementSigma)
        my = y + np.random.normal(0, measurementSigma)
        mxH.append(mx); myH.append(my)

        # Propagate true target
        vy -= g * dt
        x  += vx * dt
        y  += vy * dt

        # Ground impact
        if y <= 0 and t > 0:
            ground_impact = True
            break

        # Kalman predict + update
        kf.predict()
        kf.update(np.array([mx, my]))
        exH.append(kf.x[0,0]); eyH.append(kf.x[1,0])

        track_err.append(np.sqrt((x - kf.x[0,0])**2 + (y - kf.x[1,0])**2))

        # Impact prediction
        predicted_impact_x, _ = predict_ground_impact(
            kf.x[0,0], kf.x[1,0], kf.x[2,0], kf.x[3,0])

        # Interceptor launch
        if not int_active and t >= launch_delay:
            int_active = True
            aim_x, aim_y = predict_intercept_point(
                int_x, int_y, interceptor_speed,
                kf.x[0,0], kf.x[1,0], kf.x[2,0], kf.x[3,0]
            )
            prev_los_angle  = np.arctan2(aim_y - int_y, aim_x - int_x)
            missile_heading = prev_los_angle

        # Guidance
        if int_active:
            tdx = kf.x[0,0] - int_x
            tdy = kf.x[1,0] - int_y
            dist_to_target = np.sqrt(tdx**2 + tdy**2)

            if dist_to_target > 3000:
                # Midcourse — steer toward predicted intercept point
                aim_x, aim_y = predict_intercept_point(
                    int_x, int_y, interceptor_speed,
                    kf.x[0,0], kf.x[1,0], kf.x[2,0], kf.x[3,0]
                )
                desired_heading = np.arctan2(aim_y - int_y, aim_x - int_x)
                heading_error = np.arctan2(
                    np.sin(desired_heading - missile_heading),
                    np.cos(desired_heading - missile_heading)
                )
                turn_rate = np.clip(
                    heading_error / dt,
                    -max_accel / interceptor_speed,
                    max_accel / interceptor_speed
                )
                missile_heading += turn_rate * dt
            else:
                # Terminal — PN against the target
                los_angle = np.arctan2(tdy, tdx)
                los_rate  = np.arctan2(
                    np.sin(los_angle - prev_los_angle),
                    np.cos(los_angle - prev_los_angle)
                ) / dt
                int_vx_now = interceptor_speed * np.cos(missile_heading)
                int_vy_now = interceptor_speed * np.sin(missile_heading)
                rel_vx = kf.x[2,0] - int_vx_now
                rel_vy = kf.x[3,0] - int_vy_now
                closing_vel = -(tdx * rel_vx + tdy * rel_vy) / max(dist_to_target, 1e-3)
                Vc = max(closing_vel, interceptor_speed * 0.1)
                cmd_accel = np.clip(N_pn * Vc * los_rate, -max_accel, max_accel)
                missile_heading += (cmd_accel / interceptor_speed) * dt

            prev_los_angle = np.arctan2(tdy, tdx)

            int_vx = interceptor_speed * np.cos(missile_heading)
            int_vy = interceptor_speed * np.sin(missile_heading)
            int_x += int_vx * dt
            int_y += int_vy * dt
            int_xH.append(int_x); int_yH.append(int_y)

            # Kill check
            if np.sqrt((x - int_x)**2 + (y - int_y)**2) < intercept_radius:
                intercepted   = True
                intercept_idx = len(xH)
                break

    # ── Record run statistics ─────────────────────────────────────────────────
    mc_intercepted.append(intercepted)
    mc_avg_track_err.append(np.mean(track_err) if track_err else 0.0)
    mc_max_track_err.append(max(track_err)     if track_err else 0.0)
    if intercepted:
        mc_intercept_x.append(int_x)
        mc_intercept_y.append(int_y)

    # Save last run for plotting
    if run == NUM_RUNS - 1:
        last = dict(
            xH=xH, yH=yH, mxH=mxH, myH=myH,
            exH=exH, eyH=eyH,
            int_xH=int_xH, int_yH=int_yH,
            int_x=int_x, int_y=int_y,
            track_err=track_err,
            intercepted=intercepted,
            intercept_idx=intercept_idx,
            apogee_reached=apogee_reached,
            apogee_alt=apogee_alt, apogee_t=apogee_t,
            predicted_impact_x=predicted_impact_x,
        )


# ── Monte Carlo summary ───────────────────────────────────────────────────────
intercept_rate = sum(mc_intercepted) / NUM_RUNS * 100

print(f"\n=== Monte Carlo Results ({NUM_RUNS} runs) ===")
print(f"Intercept rate:           {intercept_rate:.1f}%")
print(f"Mean avg tracking error:  {np.mean(mc_avg_track_err):.2f} m")
print(f"Std  avg tracking error:  {np.std(mc_avg_track_err):.2f} m")
print(f"Mean max tracking error:  {np.mean(mc_max_track_err):.2f} m")
print(f"Worst max tracking error: {max(mc_max_track_err):.2f} m")
if mc_intercept_x:
    print(f"Mean intercept altitude:  {np.mean(mc_intercept_y):.1f} m")
    print(f"Std  intercept altitude:  {np.std(mc_intercept_y):.1f} m")


# ── Last-run trajectory plot ──────────────────────────────────────────────────
plot_xH = last['xH'][:last['intercept_idx']] if last['intercept_idx'] else last['xH']
plot_yH = last['yH'][:last['intercept_idx']] if last['intercept_idx'] else last['yH']

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
ax.scatter([x0], [y0], marker="s", s=100, color="green", zorder=5,
           label="Launch Site")
ax.plot(plot_xH, plot_yH, linewidth=2, label="True Trajectory")
ax.scatter(last['mxH'], last['myH'], s=4, alpha=0.3, label="Radar Measurements")
ax.plot(last['exH'], last['eyH'], linestyle="--", label="Kalman Estimate")

if last['int_xH']:
    ax.plot(last['int_xH'], last['int_yH'], color="red", linewidth=2,
            label="Interceptor")
    ax.scatter([last['int_x']], [last['int_y']], marker="x", s=150,
               color="red", zorder=5, label="Final Interceptor Position")

if last['apogee_reached']:
    idx = min(int(last['apogee_t'] / dt), len(plot_xH) - 1)
    ax.scatter([plot_xH[idx]], [plot_yH[idx]], marker="^", s=150,
               color="purple", zorder=5,
               label=f"Apogee ({last['apogee_alt']:.0f} m)")

if last['predicted_impact_x']:
    ax.scatter([last['predicted_impact_x']], [0], marker="X", s=200,
               color="orange", zorder=5,
               label=f"Predicted Impact ({last['predicted_impact_x']:.0f} m)")

ax.axhline(0, color="brown", linewidth=1.5, label="Ground")
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.set_title(f"Ballistic Missile Defense — Last Run (Run {NUM_RUNS})")
ax.legend()
ax.grid(True)

ax2 = axes[1]
ax2.plot(last['track_err'])
ax2.set_xlabel("Time Step")
ax2.set_ylabel("Tracking Error (m)")
ax2.set_title("Kalman Filter Tracking Error — Last Run")
ax2.grid(True)

plt.tight_layout()
plt.show()

# ── Monte Carlo histograms ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(mc_avg_track_err, bins=20, color="steelblue", edgecolor="white")
axes[0].set_xlabel("Average Tracking Error (m)")
axes[0].set_ylabel("Count")
axes[0].set_title("Monte Carlo — Avg Tracking Error")
axes[0].grid(True)

axes[1].hist(mc_max_track_err, bins=20, color="tomato", edgecolor="white")
axes[1].set_xlabel("Max Tracking Error (m)")
axes[1].set_ylabel("Count")
axes[1].set_title("Monte Carlo — Max Tracking Error")
axes[1].grid(True)

if mc_intercept_y:
    axes[2].hist(mc_intercept_y, bins=20, color="mediumseagreen", edgecolor="white")
    axes[2].set_xlabel("Intercept Altitude (m)")
    axes[2].set_ylabel("Count")
    axes[2].set_title(f"Monte Carlo — Intercept Altitude ({intercept_rate:.0f}% rate)")
    axes[2].grid(True)
else:
    axes[2].text(0.5, 0.5, "No intercepts", ha="center", va="center",
                 transform=axes[2].transAxes, fontsize=14)
    axes[2].set_title("Monte Carlo — Intercept Altitude")

plt.tight_layout()
plt.show()
