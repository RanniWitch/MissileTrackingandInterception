#python "C:\Users\SaiPC\OneDrive\Documents\Missile Defense Program.py"

import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter



x = 0
y = 1000

vx = 300
vy = -80

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

for t in np.arange(0, totalTime, dt):
    xHistory.append(x)
    yHistory.append(y)

    measuredX = x +np.random.normal(0, measurementSigma)
    measuredY = y +np.random.normal(0, measurementSigma)

    measuredXhistory.append(measuredX)
    measuredYhistory.append(measuredY)

    x = x + vx * dt
    y = y + vy * dt

    kf.predict()
    estimated_x_history.append(kf.x[0,0])
    estimated_y_history.append(kf.x[1,0])


plt.plot(
    xHistory,
    yHistory, 
    label="True Trajectory"
)

measurement = np.array([
    measuredX,
    measuredY
])

kf.update(measurement)

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

plt.legend()
plt.grid(True)
plt.show()
