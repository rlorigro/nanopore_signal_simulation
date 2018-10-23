from DataGenerator import *
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as f
import torch
import numpy


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout_rate):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.bidirectional = True
        self.n_directions = int(self.bidirectional)+1

        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=n_layers,
                          batch_first=True,
                          dropout=dropout_rate,
                          bidirectional=self.bidirectional)

        self.tanh = nn.Tanh()
        # self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()

        self.output_1 = nn.Linear(hidden_size*self.n_directions, hidden_size)

    def output_function(self, x):
        x = self.output_1(x)
        x = self.tanh(x)

        return x

    def forward(self, input):
        # input:  (batch, seq_len, input_size)
        # hidden: (num_layers * num_directions, batch, hidden_size)

        batch_size = input.shape[0]

        output, h_n = self.gru(input)
        # output = self.output_function(output)

        # reshape to separate out directions from layers
        h_n = h_n.view(self.n_layers, self.n_directions, batch_size, self.hidden_size)

        # slice out only the top layer
        h_n = h_n[self.n_layers-1,:,:,:]

        # reshape to be single dimension
        h_n = h_n.view([batch_size,1,-1])

        # find a nonlinear transformation to the encoding (from a vector with 2x the length of the encoding)
        encoding = self.output_function(h_n)

        return encoding


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers, dropout_rate):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.bidirectional = True
        self.n_directions = int(self.bidirectional)+1

        # print(output_size)

        self.gru = nn.GRU(input_size=hidden_size,
                          hidden_size=hidden_size,
                          num_layers=n_layers,
                          batch_first=True,
                          dropout=dropout_rate,
                          bidirectional=self.bidirectional)

        # self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.output_1 = nn.Linear(hidden_size*self.n_directions, hidden_size*self.n_directions)
        self.output_2 = nn.Linear(hidden_size*self.n_directions, output_size)

    def output_function(self, x):
        x = self.output_1(x)
        x = self.tanh(x)
        x = self.output_2(x)
        x = self.tanh(x)

        return x

    def forward(self, context_vector, expected_output, force_threshold):
        # input:  (batch, seq_len, input_size)
        # hidden: (batch, num_layers * num_directions, hidden_size) ???
        #   ^ not true even with batch_first=True... is actually: (num_layers * num_directions, batch, hidden_size)

        batch_size = context_vector.shape[0]
        output_length = expected_output.shape[1]

        expanded_input = context_vector.expand(batch_size, output_length, self.hidden_size)

        # print(context_vector.shape)
        # print(expanded_input.shape)
        #
        # print(context_vector)
        # print(expanded_input)

        output, h_n = self.gru(expanded_input)

        outputs = self.output_function(output)

        return outputs, h_n


class RNNAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout_rate):
        super(RNNAutoencoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.encoder = EncoderRNN(input_size=input_size,
                                  hidden_size=hidden_size,
                                  n_layers=n_layers,
                                  dropout_rate=dropout_rate)

        self.decoder = DecoderRNN(hidden_size=hidden_size,
                                  output_size=input_size,
                                  n_layers=n_layers,
                                  dropout_rate=dropout_rate)

    def forward(self, x, force_threshold=1.0):
        encoding = self.encoder(x)
        output, h_n = self.decoder(context_vector=encoding, expected_output=x, force_threshold=force_threshold)

        return output
