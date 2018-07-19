from SignalGenerator import *
from matplotlib import pyplot
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as f
from RnnAutoencoder import RNNAutoencoder
from torch import optim
import torch
import numpy


def initialize_signal_generator():
    k = 6
    kmer_table_path = "/home/ryan/data/Nanopore/kmerMeans"
    kmer_table_handler = KmerTableReader(k=k, kmer_table_path=kmer_table_path)

    current_range = kmer_table_handler.get_range()
    n_kmers = kmer_table_handler.get_kmer_count()
    kmer_means = kmer_table_handler.get_kmer_current_dictionary()

    average_current_difference = float(current_range)/n_kmers

    print(current_range, n_kmers)
    print(average_current_difference)

    noise_sigma = 50*average_current_difference
    event_variation_sigma = 0.5*average_current_difference
    interpolate_rate = 1

    noise_model = GaussianNoise(mu=0, sigma=noise_sigma)
    event_variation_model = GaussianNoise(mu=0, sigma=event_variation_sigma)
    duration_model = GaussianDuration(mu=5, sigma=3, minimum_duration=1)

    signal_generator = SignalGenerator(k=6,
                                       kmer_means_dictionary=kmer_means,
                                       noise_model=noise_model,
                                       event_variation_model=event_variation_model,
                                       duration_model=duration_model,
                                       interpolate_rate=interpolate_rate)

    return signal_generator


def run():
    data_loader = initialize_signal_generator()

    # Define signal simulator parameters
    sequence_nucleotide_length = 6

    # Define architecture parameters
    hidden_size = sequence_nucleotide_length
    input_size = 1      # 1-dimensional signal
    n_layers = 1

    # Define the hyperparameters
    batch_size_train = 1
    learning_rate = 1e-2

    # Define training parameters
    n_batches = 1000

    model = RNNAutoencoder(hidden_size=hidden_size, input_size=input_size, n_layers=n_layers)

    # Initialize the optimizer with above parameters
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Define the loss function
    loss_fn = nn.MSELoss()  # mean squared error

    # Train and get the resulting loss per iteration
    train(model=model,
          data_loader=data_loader,
          optimizer=optimizer,
          loss_fn=loss_fn,
          n_batches=n_batches,
          batch_size=batch_size_train,
          sequence_nucleotide_length=sequence_nucleotide_length)

    return


def train_batch(model, x, y, optimizer, loss_fn):
    # Run forward calculation
    y_predict = model.forward(x)

    # Compute loss.
    loss = loss_fn(y_predict, y)

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable weights
    # of the model)
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

    return loss.item(), y_predict


def train(model, data_loader, optimizer, loss_fn, n_batches, batch_size, sequence_nucleotide_length):
    losses = list()

    for i in range(n_batches):
        signals, sequences = data_loader.generate_batch(batch_size=batch_size, sequence_length=sequence_nucleotide_length)

        # shape = (batch_size, seq_len, input_size)
        x = torch.FloatTensor(signals).view((batch_size, -1, 1)).div(200)

        # y_predict shape = (batch_size, seq_len, hidden_size*num_directions)
        loss, y_predict = train_batch(model=model, x=x, y=x, optimizer=optimizer, loss_fn=loss_fn)

        print(i, loss)
        losses.append(loss)

        signal = x.detach().numpy()[0,:,:].squeeze()
        signal_reconstruction = y_predict.detach().numpy()[0,:,:].squeeze()

        if i%200 == 0:
            print(signal)
            print(signal_reconstruction)

            pyplot.plot(signal)
            pyplot.plot(signal_reconstruction)
            pyplot.show()

    pyplot.plot(losses)
    pyplot.show()

if __name__ == "__main__":
    run()