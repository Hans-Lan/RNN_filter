import numpy as np
from math import cos, sin, atan2
from typing import Tuple, List
from dataclasses import dataclass

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
    Q: np.ndarray
    R: np.ndarray


default_robot = RobotParam(
    Q = np.diag([0.04, 0.04, 0.0016]),
    R = np.diag([0.04, 0.0016])
)


class CleanRobot:
    
    def __init__(self, world:WorldSet, param:RobotParam) -> None:
        self.Q = param.Q
        self.R = param.R
        self.world = world
        self.state_dim = 3
        self.action_dim = 2
        self.meas_dim = 2
        self.x = np.array([5.0, 5.0, np.pi/2])

    def sample(self, batch_size:int=1, x0:np.ndarray=None):
        if x0 is None:
            # randomly init
            x = self.world.room_size[0] * np.random.rand(batch_size, 1)
            y = self.world.room_size[1] * np.random.rand(batch_size, 1)
            heading = 2 * np.pi * np.random.rand(batch_size, 1)
            x0 = np.concatenate((x, y, heading), axis=1)
        else:
            batch_size = 1
    
    def init(self, x0:np.ndarray):
        assert x0.shape == (self.state_dim, )
        self.x = x0
    
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
            diff = x - mark
            measurement.append(np.array([np.sum(diff**2)**0.5, atan2(diff[1], diff[0])]))
        return np.concatenate(measurement)