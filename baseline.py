from filterpy.kalman import KalmanFilter
from envs_utils.linear_pendulum import LinearPendulum, default_pendulum
import numpy as np

kf = KalmanFilter(dim_x=2, dim_z=2, dim_u=1)
system = LinearPendulum(default_pendulum)
kf.P *= 0.01
kf.F = system.F
kf.B = system.B
kf.H = system.H
kf.Q = system.Q
kf.R = system.R

traj_len = 1000
batch_x, batch_u, batch_y = system.sample(seq=traj_len)
batch_x = batch_x[:,0,:]
# print(batch_x)
batch_u = batch_u[:,0,:]
# print(batch_u[:100])
batch_y = batch_y[:,0,:]
kf.x = system.x0[0]
Mse = 0
for i in range(traj_len):

    kf.predict(u=batch_u[i])
    kf.update(z=batch_y[i])
    if i >= 100:
        Mse += (kf.x - batch_x[i])**2
print(kf.K)
Mse = np.sum(Mse)/(traj_len-100)
print(Mse)
