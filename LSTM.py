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
            batch_first=True   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
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

    def forward(self, x, h_state, c_state, y):
        # x.shape = (batch_size, time_step, input_size)
        r_out, (h_n, c_n) = self.rnn(x, (h_state, c_state))

        out = self.out(h_n)

        return out


def train(model, loss_fn, optimizer, data_generator, n_batches):
    batch_size = 4

    # (input size, batch size, hidden size)
    h_state = Variable(torch.zeros((1, batch_size, model.hidden_size,)))
    c_state = Variable(torch.zeros((1, batch_size, model.hidden_size,)))

    for i in range(n_batches):
        # x.shape = (batch_size, time_step, input_size)
        x_steps, y_steps, x, y = data_generator.generate_data(batch_size=batch_size)

        # prediction.shape = (seq_len, batch, num_directions * hidden_size)
        prediction = model(x=x, y=y, h_state=h_state, c_state=c_state)   # rnn output

        # reset hidden states
        h_state = Variable(torch.zeros((1, batch_size, model.hidden_size,)))
        c_state = Variable(torch.zeros((1, batch_size, model.hidden_size,)))

        loss = loss_fn(prediction, y.reshape([1,batch_size,1]))   # loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        x_sample = x[-1,:,:].data.numpy().squeeze()
        y_sample = y[-1,:,:].data.numpy().squeeze()
        y_prediction_sample = prediction[:,-1,:].data.numpy().squeeze()
        x_sample_steps = x_steps[-1,:,:].squeeze()
        y_sample_steps = y_steps[-1,:,:].squeeze()

        # print(x_sample_steps.shape)
        # print(y_sample_steps.shape)

        # if i % 100 == 0:
        pyplot.plot(y_sample_steps, y_sample, marker='o', color='k')
        pyplot.plot(y_sample_steps, y_prediction_sample, marker='o', color='red', alpha=i/n_batches)
        # pyplot.plot(x_sample_steps, x_sample, marker='o', color='blue', alpha=0.2)

        print(i, loss.data[0].numpy())
        pyplot.draw()
        pyplot.pause(0.001)

    pyplot.ioff()
    pyplot.show()


def test(model, data_generator, n_batches):
    batch_size = 60

    # (input size, batch size, hidden size)
    h_state = Variable(torch.zeros((1, batch_size, model.hidden_size,)))
    c_state = Variable(torch.zeros((1, batch_size, model.hidden_size,)))

    for i in range(n_batches):
        # x.shape = (batch_size, time_step, input_size)
        x_steps, y_steps, x, y = data_generator.generate_data(batch_size=batch_size)

        # prediction.shape = (seq_len, batch, num_directions * hidden_size)
        prediction = model(x=x, y=y, h_state=h_state, c_state=c_state)   # rnn output

        # reset hidden states
        h_state = Variable(torch.zeros((1, batch_size, model.hidden_size,)))
        c_state = Variable(torch.zeros((1, batch_size, model.hidden_size,)))

        x_sample = x[-1,:,:].data.numpy().squeeze()
        y_sample = y[-1,:,:].data.numpy().squeeze()
        y_prediction_sample = prediction[:,-1,:].data.numpy().squeeze()
        x_sample_steps = x_steps[-1,:,:].squeeze()
        y_sample_steps = y_steps[-1,:,:].squeeze()

        # print(x_sample_steps.shape)
        # print(y_sample_steps.shape)

        # if i % 100 == 0:
        pyplot.plot(y_sample_steps, y_sample, marker='o', color='k')
        pyplot.plot(y_sample_steps, y_prediction_sample, marker='o', color='red', alpha=i/n_batches)
        # pyplot.plot(x_sample_steps, x_sample, marker='o', color='blue', alpha=0.2)

        pyplot.draw()
        pyplot.pause(0.001)

    pyplot.ioff()
    pyplot.show()


def test_sequential(model, data_generator):
    batch_size = 40
    h_state = Variable(torch.zeros((1, batch_size, model.hidden_size,)))
    c_state = Variable(torch.zeros((1, batch_size, model.hidden_size,)))

    x_steps, y_steps, x, y = data_generator.generate_data(batch_size=batch_size, sequential=True)
    prediction = model(x=x, y=y, h_state=h_state, c_state=c_state)  # rnn output

    n = prediction.shape[1]

    for i in range(n):
        x_sample = x[i,:,:].data.numpy().squeeze()
        y_sample = y[i,:,:].data.numpy().squeeze()
        y_prediction_sample = prediction[:,i,:].data.numpy().squeeze()
        x_sample_steps = x_steps[i,:,:].squeeze()
        y_sample_steps = y_steps[i,:,:].squeeze()

        # if i % 100 == 0:
        pyplot.plot(y_sample_steps, y_sample, marker='o', color='k')
        pyplot.plot(y_sample_steps, y_prediction_sample, marker='o', color='red', alpha=i/n)
        # pyplot.plot(x_sample_steps, x_sample, marker='o', color='blue', alpha=0.2)

        pyplot.draw()
        pyplot.pause(0.001)

    pyplot.ioff()
    pyplot.show()


def run():
    data_generator = SineGenerator(pi_fraction_per_training_example=0.7, n_steps_per_pi=40, max_pi=6)

    rnn = RNN()
    print(rnn)

    n_batches = 500
    learning_rate = 0.02

    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)   # optimize all rnn parameters
    loss_fn = nn.MSELoss()

    pyplot.figure(1, figsize=(12, 5))
    pyplot.ion()           # continuously plot

    train(model=rnn,
          loss_fn=loss_fn,
          optimizer=optimizer,
          data_generator=data_generator,
          n_batches=n_batches)

    test(model=rnn, data_generator=data_generator, n_batches=40)

    test_sequential(model=rnn, data_generator=data_generator)


if __name__ == "__main__":
    run()
