#python "C:\Users\SaiPC\OneDrive\Documents\Missile Defense Program.py"

import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter



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

    dx = kf.x[0,0] - interceptor_x
    dy = kf.x[1,0] - interceptor_y
    #print(dx)

    distance = np.sqrt(dx**2 + dy**2)

    if distance < intercept_radius:
        print("Target intercepted!")
        break

    direction_x = dx / distance
    direction_y = dy / distance

    interceptor_x += direction_x * interceptor_speed * dt
    interceptor_y += direction_y * interceptor_speed * dt

    interceptor_x_history.append(interceptor_x)
    interceptor_y_history.append(interceptor_y)

    

print(interceptor_x, interceptor_y)
print(len(interceptor_x_history))
print(interceptor_speed)


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

plt.plot(
    interceptor_x_history,
    interceptor_y_history,
    label="Interceptor"
)

#print ("Final True Y:", y)

plt.legend()
plt.grid(True)
plt.show()
