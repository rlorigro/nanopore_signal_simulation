import random
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)    # reproducible

# Hyper Parameters
TIME_STEP = 25       # rnn time step
INPUT_SIZE = 1      # rnn input size
LR = 0.004           # learning rate


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.hidden_size = 24

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=self.hidden_size,     # rnn hidden unit
            num_layers=3,       # number of rnn layer
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            nonlinearity="relu"
        )

        self.layer_sizes = [self.hidden_size, 24, 1]
        D_in, H1, D_out = self.layer_sizes

        # print(D_in, H1, D_out)

        self.linear1 = nn.Linear(D_in, H1)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(H1, D_out)
        self.activation2 = nn.ReLU()

    def out(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        # print(x.size())

        x = self.linear2(x)
        x = self.activation2(x)
        # print(x.size())

        return x

    def step(self, x, h):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)

        r_out, h = self.rnn(x, h)
        r_out = r_out.view(-1, self.hidden_size)

        print(r_out.shape)

        y = self.out(r_out)

        return y, h

    def forward(self, x, h, force=True):
        n_steps = x.shape[1]

        for i in range(n_steps):
            if force or i == 0:
                x_step = x[:,i,:]

            x_step, h = self.step(x_step, h)

        y = x_step
        return y, h


def sample_sin(x):
    y = np.sin(x)*0.9 + 1
    return y


def sample_cos(x):
    y = np.sin(x)
    return y


def run():
    rnn = RNN()
    print(rnn)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all rnn parameters
    loss_fn = nn.MSELoss()

    plt.figure(1, figsize=(12, 5))
    plt.ion()           # continuously plot

    train_realtime(model=rnn,
                   loss_fn=loss_fn,
                   optimizer=optimizer)

    test_realtime(model=rnn)

    # test_recursively(model=rnn)


def train_realtime(model, loss_fn, optimizer, n_iterations=24000, max_pi=16, n_steps_per_pi=50):
    n_steps = n_steps_per_pi*max_pi
    h_state = None      # for initial hidden state
    start, end = 0, max_pi*np.pi  # time range
    steps = np.linspace(start, end, n_steps, dtype=np.float32)
    sine_steps = sample_sin(steps)

    for i in range(n_iterations):
        random_index = int(round(random.random()*(n_steps-TIME_STEP-2)))
        s = steps[random_index:random_index+TIME_STEP]
        x_np = sine_steps[random_index:random_index+TIME_STEP]              # float32 for converting torch FloatTensor
        y_np = sine_steps[random_index+TIME_STEP+1:random_index+TIME_STEP+2]

        x = Variable(torch.from_numpy(x_np[np.newaxis, :, np.newaxis]))    # shape (batch, time_step, input_size)
        y = Variable(torch.from_numpy(y_np[np.newaxis, :, np.newaxis]))

        force = random.random() > 0.5
        prediction, h_state = model(x, h_state, force=force)   # rnn output

        # !! next step is important !!
        h_state = None        # repack the hidden state, break the connection from last iteration

        loss = loss_fn(prediction, y)           # loss
        optimizer.zero_grad()                   # clear gradients for this training step
        loss.backward()                         # backpropagation, compute gradients
        optimizer.step()                        # apply gradients

        if i%100==0:
            print(loss.data[0])

            # plotting
    #         plt.plot(s[-1], y_np.flatten(), marker='o', color='k')
    #         plt.plot(s[-1], prediction.data.numpy().flatten(), marker='o', color='r', alpha=i/n_iterations)
    #         plt.draw()
    #         plt.pause(0.05)
    #
    # plt.ioff()
    # plt.show()


def test_realtime(model, n_iterations=180, max_pi=8, n_steps_per_pi=50):
    h_state = None      # for initial hidden state

    n_steps = n_steps_per_pi*max_pi
    start = random.random()*30
    end = start + max_pi*np.pi  # time range
    steps = np.linspace(start, end, n_steps, dtype=np.float32)
    sine_steps = sample_sin(steps)
    sine_steps_prediction = sample_sin(steps)

    for i in range(n_steps-TIME_STEP-1):
        x_np = sine_steps_prediction[i:i+TIME_STEP]              # float32 for converting torch FloatTensor
        y_np = sine_steps[i+TIME_STEP+1]

        x = Variable(torch.from_numpy(x_np[np.newaxis, :, np.newaxis]))    # shape (batch, time_step, input_size)
        prediction, h_state = model(x, h_state)   # rnn output

        # !! next step is important !!
        h_state = None        # repack the hidden state, break the connection from last iteration

        y_predict = prediction.data.numpy().squeeze()

        # plotting
        if i%4==0:
            plt.plot(steps[i+TIME_STEP+1], y_np.flatten(), marker='o', color='k')
            plt.plot(steps[i+TIME_STEP+1], y_predict, marker='o', color='r')
            plt.draw()
            plt.pause(0.05)

    plt.ioff()
    plt.show()


def test_recursively(model, max_pi=8, n_steps_per_pi=50):
    h_state = None      # for initial hidden state

    n_steps = n_steps_per_pi*max_pi
    start = random.random()*30
    end = max_pi*np.pi  # time range
    steps = np.linspace(start, end, n_steps, dtype=np.float32)
    sine_steps = sample_sin(steps)
    sine_steps_prediction = sample_sin(steps)
    sine_steps_prediction[TIME_STEP+1:] = 0

    for i in range(n_steps-TIME_STEP-1):
        x_np = sine_steps_prediction[i:i+TIME_STEP]              # float32 for converting torch FloatTensor
        y_np = sine_steps[i+TIME_STEP+1]

        x = Variable(torch.from_numpy(x_np[np.newaxis, :, np.newaxis]))    # shape (batch, time_step, input_size)
        prediction, h_state = model(x, h_state)   # rnn output

        # !! next step is important !!
        h_state = None        # repack the hidden state, break the connection from last iteration

        y_predict = prediction.data.numpy().squeeze()

        print(sine_steps_prediction)
        sine_steps_prediction[i+TIME_STEP+1] = y_predict
        print(sine_steps_prediction)

        # plotting
        if i%4==0:
            plt.plot(steps[i+TIME_STEP+1], y_np.flatten(), marker='o', color='k')
            plt.plot(steps[i+TIME_STEP+1], y_predict, marker='o', color='r')
            plt.draw()
            plt.pause(0.05)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    run()