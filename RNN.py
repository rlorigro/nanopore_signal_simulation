import random
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as pyplot
from DataGenerator import SineGenerator


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.hidden_size = 24

        self.rnn = nn.LSTM(
            input_size=1,
            hidden_size=self.hidden_size,     # rnn hidden unit
            num_layers=1,       # number of rnn layer
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            # nonlinearity="relu"
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

    def forward(self, x, h_state, y):
        # x.shape = (batch_size, time_step, input_size)
        r_out, h_n = self.rnn(x, h_state)

        # print(h_n.shape)
        #
        # r_out = r_out[:,-1:,:]
        #
        # print(r_out.shape)

        outs = self.out(h_n)

        return outs


def run():
    data_generator = SineGenerator(pi_fraction_per_training_example=0.5, n_steps_per_pi=40, max_pi=8)

    rnn = RNN()
    print(rnn)

    learning_rate = 0.002

    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)   # optimize all rnn parameters
    loss_fn = nn.MSELoss()

    pyplot.figure(1, figsize=(12, 5))
    pyplot.ion()           # continuously plot

    train_realtime(model=rnn,
                   loss_fn=loss_fn,
                   optimizer=optimizer,
                   data_generator=data_generator,
                   n_batches=1000)


def train_realtime(model, loss_fn, optimizer, data_generator, n_batches):
    h_state = None

    for i in range(n_batches):
        # x.shape = (batch_size, time_step, input_size)
        x_steps, y_steps, x, y = data_generator.generate_data(batch_size=32)

        # prediction.shape = (seq_len, batch, num_directions * hidden_size)
        prediction = model(x=x, y=y, h_state=h_state)   # rnn output

        # !! next step is important !!
        h_state = None                  # repack the hidden state, break the connection from last iteration
        loss = loss_fn(prediction, y)   # loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        # print(x_steps.shape)
        # print(y_steps.shape)
        # print(x.shape)
        # print(y.shape)
        # print(prediction.shape)

        x_sample = x[-1,:,:].data.numpy().squeeze()
        y_sample = y[-1,:,:].data.numpy().squeeze()
        y_prediction_sample = prediction[:,-1,:].data.numpy().squeeze()
        x_sample_steps = x_steps[-1,:,:].squeeze()
        y_sample_steps = y_steps[-1,:,:].squeeze()

        # print(x_sample_steps.shape)
        # print(y_sample_steps.shape)

        # if i % 100 == 0:
        pyplot.plot(y_sample_steps, y_sample, marker='o', color='k')
        pyplot.plot(y_sample_steps, y_prediction_sample, marker='o', color='r', alpha=i/n_batches)

        print(loss.data[0])
        pyplot.draw()
        pyplot.pause(0.1)

    pyplot.ioff()
    pyplot.show()


if __name__ == "__main__":
    run()
