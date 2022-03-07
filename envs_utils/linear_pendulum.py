from typing import Tuple
import numpy as np
from dataclasses import dataclass

@dataclass
class PendulumParams:
    g: float
    T: float
    mu: float
    m: float
    l: float
    var_sys1: float
    var_sys2: float
    var_mes1: float
    var_mes2: float
    angle_limit: float
    rate_limit: float

default_pendulum = PendulumParams(
    g=9.81,
    T=0.01,
    mu=0.01,
    m=1.0,
    l=1.0,
    var_sys1=0.0005,
    var_sys2=0.001,
    var_mes1=0.01,
    var_mes2=0.09,
    angle_limit=30 / 180 * np.pi,
    rate_limit=10 / 180 * np.pi
)


class LinearPendulum:

    def __init__(self, params:PendulumParams) -> None:
        self.angle_limit = params.angle_limit
        self.rate_limit = params.rate_limit
        T = params.T
        g = params.g
        l = params.l
        m = params.m
        mu = params.mu
        sigma_sys = np.array([params.var_sys1, params.var_sys2]) ** 0.5
        self.sigma_sys = sigma_sys[None, :]
        sigma_mes = np.array([params.var_mes1, params.var_mes2]) ** 0.5
        self.sigma_mes = sigma_mes[None, :]

        self.F = np.array([
            [1., T],
            [-g*T/l, 1.0-mu*T/(m*l**2)]
        ])
        self.B = np.reshape(np.array([0, T/(m*l**2)]), (2,-1))
        self.Q = np.diag([params.var_sys1, params.var_sys2])
        self.R = np.diag([params.var_mes1, params.var_mes2])
        self.H = np.eye(2)
        self.x0 = None
    
    # def _random_x0(self):

    def sample(self, seq:int=20, batch_size:int=1, x0:np.ndarray=None):
        if x0 is None:
            angle = (np.random.rand(batch_size, 1) - 0.5) * 2 * self.angle_limit
            rate = (np.random.rand(batch_size, 1) - 0.5) * 2 * self.rate_limit
            x0 = np.concatenate((angle, rate), axis=1)
        
        batch_x = np.zeros((seq, batch_size, 2))
        batch_u = np.zeros((seq, batch_size, 1))
        batch_y = np.zeros((seq, batch_size, 2))
        self.x0 = x0
        x = x0
        bias = np.random.randn()
        for i in range(seq):
            u = np.random.randn(batch_size, 1) + bias
            batch_u[i] = u
            # x = (F @ x.T + B @ u.T).T + sigma_sys * np.random.randn(batch_size, 2)
            x = x @ self.F.T + u @ self.B.T + self.sigma_sys * np.random.randn(batch_size, 2)
            batch_x[i] = x
            batch_y[i] = x + self.sigma_mes * np.random.randn(batch_size, 2)
        
        return batch_x, batch_u, batch_y


def samlpe_linear_pendulum(
    seq:int=20,
    batch_size:int=1,
    x0:np.ndarray=None,
    params: PendulumParams=default_pendulum) -> Tuple[np.ndarray]:
    """sample from linear system pendulum

    Args:
        seq (int, optional): sequence length. Defaults to 20.
        batch_size (int, optional): batch size. Defaults to 1.
        x0 (np.ndarray, optional): init state, shape of (2, ). Defaults to None.
        params (PendulumParams, optional): paramters for the system. Defaults to default_pendulum.

    Returns:
        batch_x (np.ndarray): array shape of (seq, batch_size, 2)
        batch_u (np.ndarray): array shape of (seq, batch_size, 1)
        batch_y (np.ndarray): array shape of (seq, batch_size, 2)
    """
    angle_limit = 10 / 180 * np.pi
    rate_limit = angle_limit / 10
    T = params.T
    g = params.g
    l = params.l
    m = params.m
    mu = params.mu
    F = np.array([
        [1., T],
        [-g*T/l, 1.0-mu*T/(m*l**2)]
    ])
    B = np.reshape(np.array([0, T/(m*l**2)]), (2,-1))
    sigma_sys = np.array([params.var_sys1, params.var_sys2]) ** 0.5
    sigma_sys = sigma_sys[None, :]
    sigma_mes = np.array([params.var_mes1, params.var_mes2]) ** 0.5
    sigma_mes = sigma_mes[None, :]
    if x0 is None:
        angle = (np.random.rand(batch_size, 1) - 0.5) * 2 * angle_limit
        rate = (np.random.rand(batch_size, 1) - 0.5) * 2 * rate_limit
        x0 = np.concatenate((angle, rate), axis=1)
    
    batch_x = np.zeros((seq, batch_size, 2))
    batch_u = np.zeros((seq, batch_size, 1))
    batch_y = np.zeros((seq, batch_size, 2))
    x = x0
    bias = np.random.randn()
    for i in range(seq):
        u = np.random.randn(batch_size, 1) + bias
        batch_u[i] = u
        batch_x[i] = x
        batch_y[i] = x + sigma_mes * np.random.randn(batch_size, 2)
        # x = (F @ x.T + B @ u.T).T + sigma_sys * np.random.randn(batch_size, 2)
        x = x @ F.T + u @ B.T + sigma_sys * np.random.randn(batch_size, 2)
    
    return batch_x, batch_u, batch_y



if __name__ == "__main__":
    params = default_pendulum
    T = params.T
    g = params.g
    l = params.l
    m = params.m
    mu = params.mu
    F = np.array([
        [1., T],
        [-g*T/l, 1.0-mu*T/(m*l**2)]
    ])
    B = np.reshape(np.array([0, T/(m*l**2)]), (2,-1))
    x, u, y = samlpe_linear_pendulum(seq=20, batch_size=4)
    obs = np.concatenate((y, u), axis=2)
    err = x - y
    print(np.mean(err ** 2))