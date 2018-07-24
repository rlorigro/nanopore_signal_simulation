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
        self.n_directions = 1

        # print(output_size)

        self.gru = nn.GRU(input_size=hidden_size+output_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.output_1 = nn.Linear(hidden_size, hidden_size)
        self.output_2 = nn.Linear(hidden_size, output_size)

    def output_function(self, x):
        x = self.output_1(x)
        x = self.relu(x)
        x = self.output_2(x)
        x = self.relu(x)

        return x

    def forward(self, context_vector, expected_output, force):
        # input:  (batch, seq_len, input_size)
        # hidden: (batch, num_layers * num_directions, hidden_size) ???
        #   ^ not true even with batch_first=True... is actually: (num_layers * num_directions, batch, hidden_size)

        batch_size = context_vector.shape[0]
        output_length = expected_output.shape[1]

        outputs = list()
        h_i = torch.zeros((self.n_layers*self.n_directions, batch_size, self.hidden_size))
        prev_output = torch.zeros((batch_size, 1, 1))

        for t in range(output_length):
            input = torch.cat([context_vector, prev_output], dim=2)

            output, h_i = self.gru(input, h_i)
            # input = output[3,:,:]
            # print("batch size:", batch_size)
            # print(output.shape)
            # input = output

            output = self.output_function(output)
            # print(output.shape)

            outputs.append(output)

            prev_output = output

            if force:
                if random.random() > 0.66:
                    prev_output = expected_output[:,t,:].detach().view(batch_size, -1, 1)
                    # print(prev_output)
                    # print(prev_output.shape)

        outputs = torch.cat(outputs, dim=1)

        # print(outputs.shape)

        return outputs, h_i


class RNNAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super(RNNAutoencoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.encoder = EncoderRNN(input_size, hidden_size, n_layers)
        self.decoder = DecoderRNN(hidden_size, input_size, n_layers)

    def forward(self, x, force=False):
        output, h_n = self.encoder(x)
        output = output[:,-1:,:]
        # print(output.shape)
        # length = input.shape[1]
        output, h_n = self.decoder(context_vector=output, expected_output=x, force=force)

        return output
