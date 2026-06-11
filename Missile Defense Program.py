#python "C:\Users\SaiPC\OneDrive\Documents\Missile Launch\Missile Defense Program.py"

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches
from filterpy.kalman import KalmanFilter
from enum import Enum, auto

print("RUNNING NEWEST VERSION")

# Physics
g = 9.81  # m/s²

# Defended asset
# PAC-3 battery defends a point asset at x=50,000m, inside the ~57km impact zone
# PAC-3 defended footprint radius is ~15-20km for SRBM threats
asset_x             = 50000.0  # m
asset_y             = 0.0      # m
asset_threat_radius = 15000.0  # m — engage if predicted impact is within this radius

# Simulation
dt        = 0.04  # s
totalTime = 200   # s
NUM_RUNS  = 100

# Target initial conditions — SRBM targeting the asset at 50km
# Shallow loft angle (~30°) gives a realistic ballistic arc with the asset
# in the predicted impact zone and a descending-arc intercept geometry.
x0, y0   = 0.0, 0.0
vx0, vy0 = 700.0, 400.0

# PAC-3 Radar (AN/MPQ-65)
# Tracking accuracy ~15m 1-sigma at typical engagement ranges
measurementSigma = 15.0  # m

# PAC-3 Interceptor (MIM-104F)
# Max speed ~Mach 5 (~1700 m/s), hit-to-kill warhead
# Battery is co-located with the defended asset
interceptor_speed  = 1700.0  # m/s
interceptor_x0     = 45000.0 # m — PAC-3 battery position (5km short of asset)
interceptor_y0     = 0.0     # m
intercept_radius   = 34.0    # m — half a timestep at 1700m/s (1700*0.04/2); ensures
                              #     closest approach between frames is not missed
N_pn               = 4       # PN constant — PAC-3 uses augmented PN ~4-5
max_accel          = 300.0   # m/s² — PAC-3 ~30g lateral

# Engagement state machine
# Models the real PAC-3 engagement timeline:
#   SEARCHING        — radar scanning, no track yet
#   TRACKING         — track initiated, building track quality over N scans
#   IFF_CLASSIFYING  — IFF interrogation, waiting for friend/foe response
#   THREAT_ASSESSED  — fire control evaluates predicted impact vs defended area
#   AUTHORIZED       — engagement authorised, interceptor ready to fire
#   INTERCEPTOR_AWAY — interceptor in flight

class EngagementState(Enum):
    SEARCHING        = auto()
    TRACKING         = auto()
    IFF_CLASSIFYING  = auto()
    THREAT_ASSESSED  = auto()
    AUTHORIZED       = auto()
    INTERCEPTOR_AWAY = auto()

# Realistic PAC-3 timeline durations (seconds)
# Source: open-source PAC-3 engagement timeline estimates
TRACKING_DURATION       = 2.0   # scans needed to confirm track
IFF_DURATION            = 1.0   # IFF interrogation and response window
THREAT_ASSESS_DURATION  = 1.5   # fire control threat evaluation
AUTHORIZATION_DURATION  = 1.0   # engagement authorisation delay

# IFF misclassification probability
# Small chance the IFF check fails and a hostile track is classified as friendly
IFF_MISCLASSIFY_PROB = 0.02  # 2% chance of IFF failure per engagement


# Helper functions

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
                            search_steps=1000, max_tof=150.0):
    # Find the point where the interceptor and target arrive at the same time.
    # Condition: dist(interceptor, future_target_pos) / int_speed == tau
    # i.e. the interceptor flies straight there and arrives exactly when the target does.
    # We also require the target to be on its descending arc (vy_at_tau < 0).
    best_fx, best_fy = tgt_x, tgt_y
    best_err = 1e9
    for tau in np.linspace(0.1, max_tof, search_steps):
        fx, fy = ballistic_position(tgt_x, tgt_y, tgt_vx, tgt_vy, tau)
        if fy < 0:
            break
        vy_at_tau = tgt_vy - g * tau
        if vy_at_tau >= 0:
            continue  # target still ascending — only intercept on descent
        dist = np.sqrt((fx - int_x)**2 + (fy - int_y)**2)
        tof = dist / int_speed      # time for interceptor to fly straight to that point
        err = abs(tof - tau)        # how well the arrival times match
        if err < best_err:
            best_err = err
            best_fx, best_fy = fx, fy
        if err < 0.5:               # within 0.5s timing accuracy — good enough
            return fx, fy
    return best_fx, best_fy


def compute_launch_time(int_x, int_y, int_speed,
                        tgt_x, tgt_y, tgt_vx, tgt_vy,
                        current_t, search_steps=2000, max_tof=150.0):
    """
    Find the intercept point on the descending arc and return:
      - the absolute time to launch (current_t + delay)
      - the aim point (fx, fy)
    Strategy: scan the target's future positions on the descending arc,
    find the one where dist/speed == time_until_target_arrives.
    The interceptor should launch at (current_t + tau - tof).
    Returns (launch_time, aim_x, aim_y).
    """
    best = None
    best_err = 1e9
    for tau in np.linspace(0.1, max_tof, search_steps):
        fx, fy = ballistic_position(tgt_x, tgt_y, tgt_vx, tgt_vy, tau)
        if fy < 0:
            break
        vy_at_tau = tgt_vy - g * tau
        if vy_at_tau >= 0:
            continue  # ascending — skip
        dist  = np.sqrt((fx - int_x)**2 + (fy - int_y)**2)
        tof   = dist / int_speed           # interceptor flight time to that point
        t_launch_needed = current_t + tau - tof   # absolute time to fire
        if t_launch_needed < current_t:
            continue  # would need to have launched in the past
        err = abs(tof - tau)  # timing residual (should approach 0 at the solution)
        if err < best_err:
            best_err = err
            best = (t_launch_needed, fx, fy, tof)
        # Good enough — within half a second timing accuracy
        if err < 0.5:
            break
    if best:
        return best[0], best[1], best[2]
    # Fallback: launch now, aim at current target position
    return current_t, tgt_x, tgt_y


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


# Component tests

def run_tests():
    """Validates each simulation component. Prints PASS/FAIL and raises on failure."""
    failures = []

    def check(name, condition, detail=""):
        if condition:
            print(f"  PASS  {name}")
        else:
            print(f"  FAIL  {name}" + (f" -- {detail}" if detail else ""))
            failures.append(name)

    print("\n=== Running Component Tests ===")

    # 1. Ballistic physics -- at t=vy0/g the projectile should be at apogee
    t_apogee = vy0 / g
    _, y_apogee = ballistic_position(x0, y0, vx0, vy0, t_apogee)
    expected_apogee = vy0**2 / (2 * g)
    check("Ballistic apogee altitude",
          abs(y_apogee - expected_apogee) < 1.0,
          f"got {y_apogee:.1f}m, expected {expected_apogee:.1f}m")

    # 2. Ground impact prediction -- analytic range = vx * (2*vy/g)
    impact_x, t_impact = predict_ground_impact(x0, y0, vx0, vy0)
    expected_range = vx0 * (2 * vy0 / g)
    check("Impact prediction range",
          impact_x is not None and abs(impact_x - expected_range) < 10.0,
          f"got {impact_x:.1f}m, expected {expected_range:.1f}m")
    check("Impact prediction time is positive",
          t_impact is not None and t_impact > 0,
          f"got {t_impact}")

    # 3. predict_ground_impact returns None for underground start
    ix, it = predict_ground_impact(0, -100, 400, -100)
    check("Impact prediction handles underground position (returns None)",
          ix is None and it is None)

    # 4. Intercept point search finds a reachable point on the descending arc
    aim_x, aim_y = predict_intercept_point(
        interceptor_x0, interceptor_y0, interceptor_speed, x0, y0, vx0, vy0)
    check("Intercept point is above ground",
          aim_y >= 0,
          f"aim_y={aim_y:.1f}m")
    check("Intercept point is reachable at interceptor speed",
          np.sqrt((aim_x - interceptor_x0)**2 + (aim_y - interceptor_y0)**2) / interceptor_speed <= 150.0,
          f"range={np.sqrt((aim_x-interceptor_x0)**2+(aim_y-interceptor_y0)**2):.0f}m, speed={interceptor_speed}m/s")

    # 5. Kalman filter initialises correctly
    kf_test = make_kf()
    check("Kalman filter initial x position",
          abs(kf_test.x[0,0] - x0) < 1e-9)
    check("Kalman filter initial y position",
          abs(kf_test.x[1,0] - y0) < 1e-9)
    check("Kalman filter initial vx",
          abs(kf_test.x[2,0] - vx0) < 1e-9)
    check("Kalman filter initial ay = -g",
          abs(kf_test.x[5,0] - (-g)) < 1e-9,
          f"got {kf_test.x[5,0]:.4f}, expected {-g:.4f}")

    # 6. Defended asset threat detection
    dist_inside = abs(impact_x - asset_x)
    check("Threat confirmed: impact within asset_threat_radius",
          dist_inside <= asset_threat_radius,
          f"impact={impact_x:.0f}m, asset={asset_x:.0f}m, dist={dist_inside:.0f}m, radius={asset_threat_radius:.0f}m")

    dist_outside = abs(0.0 - asset_x)
    check("No threat: impact outside asset_threat_radius",
          dist_outside > asset_threat_radius,
          f"dist={dist_outside:.0f}m, radius={asset_threat_radius:.0f}m")

    # 7. IFF misclassification probability is in valid range
    check("IFF_MISCLASSIFY_PROB in [0, 1]",
          0.0 <= IFF_MISCLASSIFY_PROB <= 1.0,
          f"got {IFF_MISCLASSIFY_PROB}")

    # 8. Engagement latency matches sum of state durations
    expected_latency = TRACKING_DURATION + IFF_DURATION + THREAT_ASSESS_DURATION + AUTHORIZATION_DURATION
    check("Engagement latency matches sum of state durations",
          abs(expected_latency - 5.5) < 1e-9,
          f"got {expected_latency:.1f}s, expected 5.5s")

    print(f"\n  {len(failures)} failure(s)")
    if failures:
        raise AssertionError(f"Tests failed: {failures}")
    print("  All tests passed.\n")

run_tests()


# Monte Carlo results arrays
mc_intercepted   = []
mc_avg_track_err = []
mc_max_track_err = []
mc_intercept_x   = []
mc_intercept_y   = []
mc_launch_t      = []  # when interceptor was fired each run
mc_iff_blocked   = []  # runs where IFF misclassification stopped engagement

last = {}

# Monte Carlo loop
for run in range(NUM_RUNS):

    # Per-run target state
    x, y, vx, vy = x0, y0, vx0, vy0
    prev_vy = vy
    kf = make_kf()

    # Interceptor state — battery is near the defended asset, not at the launch site
    int_x, int_y      = interceptor_x0, interceptor_y0
    int_active        = False
    missile_heading   = 0.0
    prev_los_angle    = 0.0
    aim_x, aim_y      = 0.0, 0.0   # predicted intercept point (updated in midcourse)
    aim_update_t      = -999.0      # time of last aim point refresh
    terminal_phase    = False       # latched True once interceptor enters terminal PN
    scheduled_launch_t = None       # computed optimal launch time (set at AUTHORIZED)

    # Engagement state machine
    eng_state          = EngagementState.SEARCHING
    state_entry_t      = 0.0   # time when current state was entered
    launch_t           = None  # time interceptor actually fired
    iff_blocked        = False  # did IFF misclassification prevent engagement
    scheduled_launch_t = None  # optimal launch time computed at AUTHORIZED

    # Event flags
    apogee_reached     = False
    apogee_alt         = 0.0
    apogee_t           = 0.0
    intercepted        = False
    ground_impact      = False
    predicted_impact_x = None
    intercept_idx      = None

    # History
    xH, yH         = [], []
    mxH, myH       = [], []
    exH, eyH       = [], []
    int_xH, int_yH = [], []
    track_err      = []
    state_history  = []  # (t, state name) for last-run logging

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

        # Propagate true target (ballistic)
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

        # Impact prediction from Kalman estimate
        predicted_impact_x, _ = predict_ground_impact(
            kf.x[0,0], kf.x[1,0], kf.x[2,0], kf.x[3,0])

        # Engagement state machine
        time_in_state = t - state_entry_t

        if eng_state == EngagementState.SEARCHING:
            # First radar return initiates track
            if t > 0:
                eng_state     = EngagementState.TRACKING
                state_entry_t = t
                state_history.append((t, "TRACKING"))

        elif eng_state == EngagementState.TRACKING:
            # Hold for TRACKING_DURATION to build track quality
            if time_in_state >= TRACKING_DURATION:
                eng_state     = EngagementState.IFF_CLASSIFYING
                state_entry_t = t
                state_history.append((t, "IFF_CLASSIFYING"))

        elif eng_state == EngagementState.IFF_CLASSIFYING:
            # IFF interrogation window -- wait IFF_DURATION seconds for response
            # A real system sends a coded interrogation pulse and checks for a
            # valid Mode 4/5 crypto reply. No valid reply = classified HOSTILE.
            if time_in_state >= IFF_DURATION:
                # Simulate small probability of misclassification (HOSTILE read as FRIENDLY)
                if np.random.random() < IFF_MISCLASSIFY_PROB:
                    iff_blocked = True
                    state_history.append((t, "IFF MISCLASSIFY -- NO ENGAGE"))
                    break  # Track classified friendly by mistake -- no engagement
                else:
                    eng_state     = EngagementState.THREAT_ASSESSED
                    state_entry_t = t
                    state_history.append((t, "THREAT_ASSESSED (HOSTILE confirmed)"))

        elif eng_state == EngagementState.THREAT_ASSESSED:
            # Fire control evaluates predicted impact vs defended asset
            if time_in_state >= THREAT_ASSESS_DURATION:
                threat_confirmed = False
                if predicted_impact_x is not None:
                    if abs(predicted_impact_x - asset_x) <= asset_threat_radius:
                        threat_confirmed = True

                if threat_confirmed:
                    eng_state     = EngagementState.AUTHORIZED
                    state_entry_t = t
                    state_history.append((t, "AUTHORIZED (threat confirmed)"))
                else:
                    state_history.append((t, "NO ENGAGE (outside defended area)"))
                    break

        elif eng_state == EngagementState.AUTHORIZED:
            # Fire control has authorised the shot.
            # Compute the optimal launch time so the interceptor arrives at the
            # descending-arc intercept point at the same moment as the target.
            # Hold the interceptor on the rail until that time — then fire.
            if time_in_state >= AUTHORIZATION_DURATION and scheduled_launch_t is None:
                scheduled_launch_t, aim_x, aim_y = compute_launch_time(
                    int_x, int_y, interceptor_speed,
                    kf.x[0,0], kf.x[1,0], kf.x[2,0], kf.x[3,0],
                    current_t=t
                )
                state_history.append((t, f"LAUNCH SCHEDULED for t={scheduled_launch_t:.1f}s  aim=({aim_x:.0f},{aim_y:.0f})"))

            if scheduled_launch_t is not None and t >= scheduled_launch_t:
                eng_state      = EngagementState.INTERCEPTOR_AWAY
                state_entry_t  = t
                launch_t       = t
                int_active     = True
                # Point directly at the aim point at the moment of launch
                missile_heading = np.arctan2(aim_y - int_y, aim_x - int_x)
                prev_los_angle  = missile_heading
                aim_update_t    = t
                terminal_phase  = False
                state_history.append((t, "INTERCEPTOR_AWAY"))

        # Guidance (only when interceptor is away)
        if int_active:
            tdx = kf.x[0,0] - int_x
            tdy = kf.x[1,0] - int_y
            dist_to_target = np.sqrt(tdx**2 + tdy**2)

            # Distance to the aim point — this is what controls phase switching,
            # NOT distance to the current target position.
            adx = aim_x - int_x
            ady = aim_y - int_y
            dist_to_aim = np.sqrt(adx**2 + ady**2)

            los_angle = np.arctan2(tdy, tdx)

            # Switch to terminal PN when interceptor is within 5km of the aim point
            # AND the target is within 6km — ensures PN activates only when
            # both the interceptor and target are converging on the same point.
            if dist_to_aim < 5000.0 and dist_to_target < 6000.0:
                terminal_phase = True

            if not terminal_phase:
                # ---- MIDCOURSE ----
                # Steer toward the predicted intercept point (aim_x, aim_y).
                # Refresh the aim point every 2s using the latest KF estimate so
                # it stays accurate as the target progresses along its arc.
                if t - aim_update_t >= 2.0:
                    aim_x, aim_y = predict_intercept_point(
                        int_x, int_y, interceptor_speed,
                        kf.x[0,0], kf.x[1,0], kf.x[2,0], kf.x[3,0]
                    )
                    aim_update_t = t

                # Heading error toward aim point (not toward current target).
                desired_heading = np.arctan2(aim_y - int_y, aim_x - int_x)
                heading_error   = np.arctan2(
                    np.sin(desired_heading - missile_heading),
                    np.cos(desired_heading - missile_heading)
                )
                # Turn budget per timestep = max lateral accel / speed * dt.
                # Clipping the angle directly (not dividing by dt) prevents
                # the sustained max-rate turning that caused the loop.
                max_turn_per_step = (max_accel / interceptor_speed) * dt
                missile_heading  += np.clip(heading_error, -max_turn_per_step, max_turn_per_step)

            else:
                # ---- TERMINAL PN ----
                # Pure proportional navigation against the actual target.
                # LOS rate = how fast the line-of-sight angle is rotating.
                # PN command = N * closing_velocity * LOS_rate (classic PN law).
                # This is model-free — works for any target motion.
                los_rate  = np.arctan2(
                    np.sin(los_angle - prev_los_angle),
                    np.cos(los_angle - prev_los_angle)
                ) / dt
                int_vx_now  = interceptor_speed * np.cos(missile_heading)
                int_vy_now  = interceptor_speed * np.sin(missile_heading)
                rel_vx      = kf.x[2,0] - int_vx_now
                rel_vy      = kf.x[3,0] - int_vy_now
                closing_vel = -(tdx * rel_vx + tdy * rel_vy) / max(dist_to_target, 1e-3)
                Vc          = max(closing_vel, interceptor_speed * 0.1)
                cmd_accel   = np.clip(N_pn * Vc * los_rate, -max_accel, max_accel)
                missile_heading += (cmd_accel / interceptor_speed) * dt

            # Always update LOS history for smooth PN on phase transition
            prev_los_angle = los_angle

            int_vx = interceptor_speed * np.cos(missile_heading)
            int_vy = interceptor_speed * np.sin(missile_heading)
            int_x += int_vx * dt
            int_y += int_vy * dt
            int_xH.append(int_x); int_yH.append(int_y)

            # Kill check against true target position.
            # intercept_radius is set to half a timestep of travel (1700 * 0.04 / 2 = 34m)
            # to ensure we don't step over the closest approach point between frames.
            if np.sqrt((x - int_x)**2 + (y - int_y)**2) < intercept_radius:
                intercepted   = True
                intercept_idx = len(xH)
                break

    # Record run statistics
    mc_intercepted.append(intercepted)
    mc_avg_track_err.append(np.mean(track_err) if track_err else 0.0)
    mc_max_track_err.append(max(track_err)     if track_err else 0.0)
    mc_iff_blocked.append(iff_blocked)
    if intercepted:
        mc_intercept_x.append(int_x)
        mc_intercept_y.append(int_y)
    if launch_t is not None:
        mc_launch_t.append(launch_t)

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
            state_history=state_history,
            launch_t=launch_t,
            iff_blocked=iff_blocked,
        )


# Monte Carlo summary
intercept_rate = sum(mc_intercepted) / NUM_RUNS * 100
iff_block_rate = sum(mc_iff_blocked) / NUM_RUNS * 100

print(f"\n=== Monte Carlo Results ({NUM_RUNS} runs) ===")
print(f"Intercept rate:               {intercept_rate:.1f}%")
print(f"IFF misclassification rate:   {iff_block_rate:.1f}%")
print(f"Mean avg tracking error:      {np.mean(mc_avg_track_err):.2f} m")
print(f"Std  avg tracking error:      {np.std(mc_avg_track_err):.2f} m")
print(f"Mean max tracking error:      {np.mean(mc_max_track_err):.2f} m")
print(f"Worst max tracking error:     {max(mc_max_track_err):.2f} m")
if mc_launch_t:
    print(f"Mean interceptor launch time: {np.mean(mc_launch_t):.2f} s")
    print(f"  (= {TRACKING_DURATION}s tracking"
          f" + {IFF_DURATION}s IFF"
          f" + {THREAT_ASSESS_DURATION}s threat assessment"
          f" + {AUTHORIZATION_DURATION}s authorization)")
if mc_intercept_x:
    print(f"Mean intercept altitude:      {np.mean(mc_intercept_y):.1f} m")
    print(f"Std  intercept altitude:      {np.std(mc_intercept_y):.1f} m")

print(f"\n=== Last Run Engagement Timeline ===")
for (evt_t, state_name) in last.get('state_history', []):
    print(f"  t={evt_t:6.2f}s  ->  {state_name}")
if last.get('launch_t'):
    print(f"  Total engagement latency: {last['launch_t']:.2f} s")
if last.get('iff_blocked'):
    print("  Engagement blocked by IFF misclassification")
if last.get('intercepted'):
    print(f"  Intercept point:  x={last['int_x']:.0f} m,  y={last['int_y']:.0f} m")
    print(f"  Intercept altitude: {last['int_y']:.0f} m  ({last['int_y']/1000:.2f} km)")


# Last-run trajectory plot
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
            label="PAC-3 Interceptor")
    ax.scatter([last['int_x']], [last['int_y']], marker="x", s=150,
               color="red", zorder=5, label="Final Interceptor Position")
    ax.scatter([interceptor_x0], [interceptor_y0], marker="^", s=150,
               color="red", zorder=5, label="PAC-3 Battery")

if last['apogee_reached']:
    # Use full xH/yH (not the intercept-truncated plot arrays) so the apogee
    # marker is always drawn at the correct position on the arc.
    idx = min(int(last['apogee_t'] / dt), len(last['xH']) - 1)
    ax.scatter([last['xH'][idx]], [last['yH'][idx]], marker="^", s=150,
               color="purple", zorder=5,
               label=f"Apogee ({last['apogee_alt']:.0f} m)")

if last['predicted_impact_x']:
    ax.scatter([last['predicted_impact_x']], [0], marker="X", s=200,
               color="orange", zorder=5,
               label=f"Predicted Impact ({last['predicted_impact_x']:.0f} m)")

asset_circle = matplotlib.patches.Wedge(
    (asset_x, asset_y), asset_threat_radius,
    theta1=0, theta2=180,          # upper half only — ground is not defended
    facecolor="blue", alpha=0.15,
    edgecolor="blue", linewidth=1.5,
    label=f"Defended Area ({asset_threat_radius/1000:.0f}km radius)"
)
ax.add_patch(asset_circle)
ax.scatter([asset_x], [asset_y], marker="D", s=120, color="blue",
           zorder=5, label="Defended Asset")

ax.axhline(0, color="brown", linewidth=1.5, label="Ground")
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.set_title("PAC-3 Engagement Simulation -- Last Run")
ax.legend()
ax.grid(True)

ax2 = axes[1]
ax2.plot(last['track_err'])
ax2.set_xlabel("Time Step")
ax2.set_ylabel("Tracking Error (m)")
ax2.set_title("Kalman Filter Tracking Error -- Last Run")
ax2.grid(True)

plt.tight_layout()
plt.show()


# Monte Carlo histograms
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(mc_avg_track_err, bins=20, color="steelblue", edgecolor="white")
axes[0].set_xlabel("Average Tracking Error (m)")
axes[0].set_ylabel("Count")
axes[0].set_title("Monte Carlo -- Avg Tracking Error")
axes[0].grid(True)

axes[1].hist(mc_max_track_err, bins=20, color="tomato", edgecolor="white")
axes[1].set_xlabel("Max Tracking Error (m)")
axes[1].set_ylabel("Count")
axes[1].set_title("Monte Carlo -- Max Tracking Error")
axes[1].grid(True)

if mc_intercept_y:
    axes[2].hist(mc_intercept_y, bins=20, color="mediumseagreen", edgecolor="white")
    axes[2].set_xlabel("Intercept Altitude (m)")
    axes[2].set_ylabel("Count")
    axes[2].set_title(f"Monte Carlo -- Intercept Altitude ({intercept_rate:.0f}% rate)")
    axes[2].grid(True)
else:
    axes[2].text(0.5, 0.5, "No intercepts", ha="center", va="center",
                 transform=axes[2].transAxes, fontsize=14)
    axes[2].set_title("Monte Carlo -- Intercept Altitude")

plt.tight_layout()
plt.show()
