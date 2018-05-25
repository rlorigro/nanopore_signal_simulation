"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
torch: 0.1.11
matplotlib
numpy
"""
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
TIME_STEP = 4       # rnn time step
INPUT_SIZE = 1      # rnn input size
LR = 0.02           # learning rate

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,     # rnn hidden unit
            num_layers=2,       # number of rnn layer
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)

        print("x",x.shape)
        print("h_state",h_state.shape)
        print("r_out",r_out.shape)

        r_out = r_out.view(-1, 32)

        print(r_out)
        outs = self.out(r_out)

        return outs, h_state


def run():
    rnn = RNN()
    print(rnn)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
    loss_fn = nn.MSELoss()

    plt.figure(1, figsize=(12, 5))
    plt.ion()           # continuously plot

    train_realtime(model=rnn,
                   loss_fn=loss_fn,
                   optimizer=optimizer)


def train_realtime(model, loss_fn, optimizer, n_steps=12):
    h_state = None      # for initial hidden state

    for step in range(n_steps):
        start, end = step * np.pi, (step+1)*np.pi   # time range
        # use sin predicts cos
        steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)

        print(steps)

        # x_np = np.sin(steps)    # float32 for converting torch FloatTensor
        x_np = steps              # float32 for converting torch FloatTensor
        y_np = np.cos(steps)

        print(x_np[np.newaxis, :, np.newaxis])

        x = Variable(torch.from_numpy(x_np[np.newaxis, :, np.newaxis]))    # shape (batch, time_step, input_size)
        y = Variable(torch.from_numpy(y_np[np.newaxis, :, np.newaxis]))

        prediction, h_state = model(x, h_state)   # rnn output

        # !! next step is important !!
        h_state = Variable(h_state.data)        # repack the hidden state, break the connection from last iteration

        loss = loss_fn(prediction, y)           # cross entropy loss
        optimizer.zero_grad()                   # clear gradients for this training step
        loss.backward()                         # backpropagation, compute gradients
        optimizer.step()                        # apply gradients

        # plotting
        plt.plot(steps, y_np.flatten(), 'r-')
        plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
        plt.draw(); plt.pause(0.05)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    # show data
    # steps = np.linspace(0, np.pi * 2, 100, dtype=np.float32)
    # x_np = np.sin(steps)  # float32 for converting torch FloatTensor
    # y_np = np.cos(steps)
    # plt.plot(steps, y_np, 'r-', label='target (cos)')
    # plt.plot(steps, x_np, 'b-', label='input (sin)')
    # plt.legend(loc='best')
    # plt.show()

    run()