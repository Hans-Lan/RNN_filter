import numpy as np
from math import cos, sin, atan2
from typing import Tuple, List
from dataclasses import dataclass
from matplotlib import pyplot as plt

@dataclass
class WorldSet:
    room_size: Tuple[float]
    land_marks: np.ndarray

default_world = WorldSet(
    room_size=(10, 10),
    land_marks=np.array([
        [1.2, 1.3],
        [8.5, 1.0],
        [1.2, 9.0],
        [9.2, 8.5]
    ])
)


@dataclass
class RobotParam:
    Q: List
    R: List


default_robot = RobotParam(
    Q = [0.04, 0.04, 0.0016],
    R = [0.04, 0.0016]
)


class CleanRobot:
    
    def __init__(self, world:WorldSet, param:RobotParam) -> None:
        self.Q = np.diag(param.Q)
        self.R = np.diag(param.R)
        self.world = world
        self.sigma_sys = np.array(param.Q) ** 0.5
        self.sigma_mes = np.array(param.R * self.world.land_marks.shape[0]) ** 0.5
        self.state_dim = 3
        self.action_dim = 2
        self.meas_dim = 2 * self.world.land_marks.shape[0]
        self.x = np.array([5.0, 5.0, np.pi/2])
        self.action_space = np.array([0.2, np.pi])

    def sample(self, seq:int=20, batch_size:int=1, x0:np.ndarray=None) -> np.ndarray:
        if x0 is None:
            # randomly init
            x = self.world.room_size[0] * np.random.rand(batch_size, 1)
            y = self.world.room_size[1] * np.random.rand(batch_size, 1)
            heading = 2 * np.pi * np.random.rand(batch_size, 1)
            x0 = np.concatenate((x, y, heading), axis=1)
        else:
            batch_size = 1
        # bias = np.random.rand(self.action_dim)
        batch_x = np.zeros((seq, batch_size, self.state_dim))
        batch_u = np.zeros((seq, batch_size, self.action_dim))
        batch_y = np.zeros((seq, batch_size, self.meas_dim))
        xt = x0
        for i in range(seq):
            ut = (2 * (np.random.rand(batch_size, self.action_dim) - 0.5)) * self.action_space
            batch_u[i] = ut
            xt = self._dynamic_step(xt, ut) + self.sigma_sys * np.random.randn(batch_size, self.state_dim)
            batch_x[i] = xt
            batch_y[i] = self.measure_batch(xt) + self.sigma_mes * np.random.randn(batch_size, self.meas_dim)
        
        return batch_x, batch_u, batch_y
    
    def init(self, x0:np.ndarray):
        assert x0.shape == (self.state_dim, )
        self.x = x0
    
    # idempotent version
    def _dynamic_step(self, xt:np.ndarray, ut:np.ndarray) -> np.ndarray:
        """dynamic system update

        Args:
            xt (np.ndarray): shape: [batch_size, state_dim]
            ut (np.ndarray): shape: [batch_size, action_dim]

        Returns:
            x_t+1 (np.ndarray): shape: [batch_size, state_dim]
        """
        batch_size = xt.shape[0]
        x_t_1 = np.zeros_like(xt)
        for i in range(batch_size):
            xi = xt[i]
            b = np.array([
                [cos(xi[2]), 0.],
                [sin(xi[2]), 0.],
                [0., 1.]
            ])
            x_t_1[i] = xi + b @ ut[i]
        return x_t_1

    def step(self, u_t:np.ndarray):
        xt = self.x
        Done = False
        b_t = np.array([
            [cos(xt[2]), 0.],
            [sin(xt[2]), 0.],
            [0., 1.]
        ])
        self.x += b_t @ u_t
        # if self.x[0] > 10 or self.x[0]
        return self.x, Done
    
    def measure(self, x:np.ndarray):
        measurement = list()
        for mark in self.world.land_marks:
            diff = x[:2] - mark
            measurement.append(np.array([np.sum(diff**2)**0.5, atan2(diff[1], diff[0])]))
        return np.concatenate(measurement)
    
    def measure_batch(self, x:np.ndarray):
        batch_size = x.shape[0]
        meas = np.zeros((batch_size, self.meas_dim))
        for i in range(batch_size):
            meas[i] = self.measure(x[i])
        return meas
    
    def render(self, x:np.ndarray, x_hat:np.ndarray) -> None:
        pass


if __name__ == "__main__":
    myCleanRobot = CleanRobot(default_world, default_robot)
    x = np.ones((1, 3)) * 5.0
    x, u, y = myCleanRobot.sample(seq=10, batch_size=4)
    print(y.shape)
