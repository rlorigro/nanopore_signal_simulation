from DataGenerator import *
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as f
import torch
import numpy


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.gru = nn.GRU(input_size, hidden_size, n_layers, batch_first=True)

    def forward(self, input):
        output, h_n = self.gru(input)

        return output, h_n


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        # self.lstm = nn.LSTM(hidden_size, output_size, n_layers, batch_first=True)
        self.gru = nn.GRU(hidden_size, output_size, n_layers, batch_first=True)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, encoded_input):
        output, h_n = self.gru(encoded_input)
        output = self.relu(output)

        return output, h_n


class RNNAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super(RNNAutoencoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.encoder = EncoderRNN(input_size, hidden_size, n_layers)
        self.decoder = DecoderRNN(hidden_size, input_size, n_layers)

    def forward(self, input):
        output, h_n = self.encoder(input)
        output, h_n = self.decoder(output)

        return output
