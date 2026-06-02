#python "C:\Users\SaiPC\OneDrive\Documents\Missile Launch\Missile Defense Program.py"


import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter


print("RUNNING NEWEST VERSION")

x = 0
y = 1000

vx = 300
vy = -20

#seconds
dt = 0.1
totalTime = 20

xHistory = []
yHistory = []

#meters
measurementSigma = 50

#stores noise
measuredXhistory = []
measuredYhistory = []

#dimX is the number of state variables
#dimZ is the number of radar measurements
kf = KalmanFilter(dim_x=4, dim_z=2)
kf.x = np.array([
    [0.],
    [1000.],
    [300.],
    [-20.]
])

#State Transition Matrix
kf.F = np.array([
    [1,0,dt,0],
    [0,1,0,dt],
    [0,0,1,0],
    [0,0,0,1]
])

#Measurement Matrix
kf.H = np.array([
    [1,0,0,0],
    [0,1,0,0]
])

#Variance
kf.R = np.array([
    [2500,0],
    [0,2500]
])

#Initial Covariance
kf.P *= 1000

#Process Noise
kf.Q *= 0.1

#Filter estimate array
estimated_x_history = []
estimated_y_history = []

#print(vy)

#interceptor variables
interceptor_x = 0
interceptor_y = 0
interceptor_speed = 350

dx = 300
dy = 400

#interceptor history array
interceptor_x_history = []
interceptor_y_history = []


intercept_radius = 25

#Navigation Constant
N = 4

#Line of Sight
los_history = []

previous_los_angle = np.arctan2(
    kf.x[1, 0] - interceptor_y,
    kf.x[0, 0] - interceptor_x
)

closing_velocity = interceptor_speed

missile_heading = previous_los_angle

plt.scatter(
    [0],
    [0],
    marker="s",
    s=100,
    label="Launcher"
)

#target acceleration
ax = 0
ay = 0

vx_history = []
vy_history = []

tracking_error_history = []

for t in np.arange(0, totalTime, dt):
    xHistory.append(x)
    yHistory.append(y)

    vx_history.append(vx)
    vy_history.append(vy)

    if t > 2:
        ay = -15

    measuredX = x +np.random.normal(0, measurementSigma)
    measuredY = y +np.random.normal(0, measurementSigma)

    measuredXhistory.append(measuredX)
    measuredYhistory.append(measuredY)

    vx += ax * dt
    vy += ay * dt

    x += vx * dt
    y += vy * dt

    kf.predict()

    measurement = np.array([measuredX, measuredY])
    kf.update(measurement)

    estimated_x_history.append(kf.x[0,0])
    estimated_y_history.append(kf.x[1,0])

    dx = kf.x[0,0] - interceptor_x
    dy = kf.x[1,0] - interceptor_y
    #print(dx)

    distance = np.sqrt(dx**2 + dy**2)

    los_angle = np.arctan2(dy, dx)
    # Wrap difference to [-pi, pi] to avoid spikes at angle boundaries
    los_rate = np.arctan2(np.sin(los_angle - previous_los_angle), np.cos(los_angle - previous_los_angle)) / dt
    previous_los_angle = los_angle

    commanded_acceleration = (
    N
    * closing_velocity
    * los_rate
    )

    missile_heading += (
    commanded_acceleration
    / interceptor_speed
    ) * dt


    interceptor_vx = interceptor_speed * np.cos(missile_heading)
    interceptor_vy = interceptor_speed * np.sin(missile_heading)

    interceptor_x += interceptor_vx * dt
    interceptor_y += interceptor_vy * dt

    interceptor_x_history.append(interceptor_x)
    interceptor_y_history.append(interceptor_y)

    error = np.sqrt(
        (x - kf.x[0,0])**2
        +
        (y - kf.x[1,0])**2
    )

    tracking_error_history.append(error)

    if distance < intercept_radius:
        print("Target intercepted!")
        break

    if len(interceptor_x_history) % 20 == 0:
        print(
        f"Heading={missile_heading:.3f}"
    )


print(interceptor_x, interceptor_y)
print(len(interceptor_x_history))
print(interceptor_speed)


plt.plot(
    xHistory,
    yHistory, 
    label="True Trajectory"
)

plt.scatter(
    measuredXhistory,
    measuredYhistory,
    s=10,
    label="Radar Measurements"
)

plt.plot(
    estimated_x_history,
    estimated_y_history,
    label="Kalman Estimate"
)

plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("Target Tracking Simulation")

plt.plot(
    interceptor_x_history,
    interceptor_y_history,
    label="Interceptor"
)

plt.scatter(
    [interceptor_x],
    [interceptor_y],
    marker="x",
    s=100,
    label="Intercept Point"
)

plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("Target Tracking Simulation")
#print ("Final True Y:", y)

plt.legend()
plt.grid(True)
plt.show()

plt.figure()

plt.plot(tracking_error_history)

plt.xlabel("Time Step")
plt.ylabel("Tracking Error (m)")
plt.title("Kalman Filter Tracking Error")
plt.grid(True)
plt.show()
