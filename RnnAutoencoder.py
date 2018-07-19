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

        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        h0 = Variable(torch.FloatTensor(self.n_layers, input.size(0), self.hidden_size))
        c0 = Variable(torch.FloatTensor(self.n_layers, input.size(0), self.hidden_size))

        encoded_input, hidden = self.lstm(input, (h0, c0))
        encoded_input = self.relu(encoded_input)

        return encoded_input


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(hidden_size, output_size, n_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, encoded_input):
        h0 = Variable(torch.FloatTensor(self.n_layers, encoded_input.size(0), self.output_size))
        c0 = Variable(torch.FloatTensor(self.n_layers, encoded_input.size(0), self.output_size))

        decoded_output, hidden = self.lstm(encoded_input, (h0, c0))
        decoded_output = self.relu(decoded_output)

        return decoded_output


class RNNAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super(RNNAutoencoder, self).__init__()
        self.encoder = EncoderRNN(input_size, hidden_size, n_layers)
        self.decoder = DecoderRNN(hidden_size, input_size, n_layers)

    def forward(self, input):
        encoded_input = self.encoder(input)
        print(encoded_input)
        decoded_output = self.decoder(encoded_input)

        return decoded_output

