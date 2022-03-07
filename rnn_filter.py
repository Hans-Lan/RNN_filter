import torch
from torch import nn
from torch import functional as F
import numpy as np
from envs_utils.linear_pendulum import default_pendulum, LinearPendulum

class RNNFilter(nn.Module):

    def __init__(self, rnn_layer, state_dim:int):
        super(RNNFilter, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = self.rnn.hidden_size
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=state_dim)
    
    def forward(self, inputs, state):
        y, state = self.rnn(inputs, state)
        output = self.linear(y.reshape((-1, y.shape[-1])))

        return output, state

    def init_state(self, device, batch_size=1):
        return torch.zeros((self.rnn.num_layers, batch_size, self.hidden_size), device=device)
    
    def predict(self):
        pass


def train_single_epoch(model:RNNFilter, env:LinearPendulum, iters, batch_size, loss, optimizer, device):

    for _ in range(iters):
        x, u, y = env.sample(seq=24, batch_size=batch_size)
        obs = torch.from_numpy(np.concatenate((y, u), axis=2)).to(torch.float32)
        x = torch.from_numpy(x).to(torch.float32)
        state = model.init_state(device, batch_size)
        x_hat, _ = model(obs, state)
        l = loss(x_hat, x.reshape((-1, x.shape[-1])))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print(l)


def train_epochs(model:nn.Module, env, num_epoch, lr, device):
    updater = torch.optim.SGD(model.parameters(), lr)
    loss = nn.MSELoss()
    for _ in range(num_epoch):
        train_single_epoch(model, env, 200, 32, loss, updater, device)


# test
if __name__ == "__main__":
    obs_size = 3
    state_size = 2
    hidden_size = 256
    batch_size = 32
    step_num = 30
    device = torch.device("cpu")
    pendulum_env = LinearPendulum(default_pendulum)
    rnn = nn.RNN(input_size=obs_size, hidden_size=hidden_size)
    rnn2 = nn.GRU(input_size=obs_size, hidden_size=hidden_size)
    filter = RNNFilter(rnn_layer=rnn, state_dim=state_size)
    train_epochs(filter, pendulum_env, 50, 0.002, device)
