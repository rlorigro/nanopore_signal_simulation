from SignalGenerator import *
from RnnAutoencoder import RNNAutoencoder
from FileManager import FileManager
from matplotlib import pyplot
from torch import nn
from torch import optim
from os import path
import torch
import datetime
import csv

class ResultsHandler:
    def __init__(self):
        self.datetime_string = '-'.join(list(map(str, datetime.datetime.now().timetuple()))[:-1])

        self.output_directory_name = "output/"
        self.directory = path.join(self.output_directory_name, self.datetime_string)

        self.n_checkpoints = 0

        FileManager.ensure_directory_exists(self.directory)

    def save_plot(self, losses):
        loss_plot_filename = path.join(self.directory, "loss.png")

        figure = pyplot.figure()
        axes = pyplot.axes()
        axes.plot(losses)
        pyplot.savefig(loss_plot_filename)

    def save_model(self, model):
        self.n_checkpoints += 1

        model_filename = path.join(self.directory, "model_checkpoint_%d" % self.n_checkpoints)
        torch.save(model.state_dict(), model_filename)

    def save_config(self, model):
        pass


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


def train(model, data_loader, optimizer, loss_fn, n_batches, batch_size, sequence_nucleotide_length, results_handler, checkpoint_interval):
    losses = list()

    for i in range(n_batches):
        signals, sequences = data_loader.generate_batch(batch_size=batch_size, sequence_length=sequence_nucleotide_length)

        # shape = (batch_size, seq_len, input_size)
        x = torch.FloatTensor(signals).view((batch_size, -1, 1))
        x = x.div(200)

        # y_predict shape = (batch_size, seq_len, hidden_size*num_directions)
        loss, y_predict = train_batch(model=model, x=x, y=x, optimizer=optimizer, loss_fn=loss_fn)
        losses.append(loss)

        print(i, loss)

        if i > 0 and i % checkpoint_interval == 0:
            results_handler.save_model(model)

            signal = x.detach().numpy()[0, :, :].squeeze()
            signal_reconstruction = y_predict.detach().numpy()[0, :, :].squeeze()

            plot_prediction(x=signal, y=signal_reconstruction)


def test(model, data_loader, n_batches, batch_size, sequence_nucleotide_length):
    for i in range(n_batches):
        signals, sequences = data_loader.generate_batch(batch_size=batch_size, sequence_length=sequence_nucleotide_length)

        # shape = (batch_size, seq_len, input_size)
        x = torch.FloatTensor(signals).view((batch_size, -1, 1))

        # normalize to fit under 1.0
        x = x.div(200)

        # y_predict shape = (batch_size, seq_len, hidden_size*num_directions)
        y_predict = model.forward(x)

        signal = x.detach().numpy()[0,:,:].squeeze()
        signal_reconstruction = y_predict.detach().numpy()[0,:,:].squeeze()

        plot_prediction(x=signal, y=signal_reconstruction)


def predict_encoding(model, data_loader, n_batches, batch_size, sequence_nucleotide_length):
    for i in range(n_batches):
        signals, sequences = data_loader.generate_batch(batch_size=batch_size, sequence_length=sequence_nucleotide_length)

        # shape = (batch_size, seq_len, input_size)
        x = torch.FloatTensor(signals).view((batch_size, -1, 1))

        # normalize to fit under 1.0
        x = x.div(200)

        # y_predict shape = (batch_size, seq_len, hidden_size*num_directions)
        y_predict = model.enforward(x)

        signal = x.detach().numpy()[0,:,:].squeeze()
        signal_reconstruction = y_predict.detach().numpy()[0,:,:].squeeze()

        plot_prediction(x=signal, y=signal_reconstruction)


def plot_prediction(x,y):
    x = x*200
    y = y*200

    fig = pyplot.figure()
    axes = pyplot.axes()

    axes.set_ylim([0,200])
    pyplot.plot(x)
    pyplot.plot(y)
    pyplot.show()


def run():
    results_handler = ResultsHandler()

    data_loader = initialize_signal_generator()

    # Define signal simulator parameters
    sequence_nucleotide_length = 8

    # Define architecture parameters
    hidden_size = 2*sequence_nucleotide_length
    input_size = 1      # 1-dimensional signal
    n_layers = 3

    # Define the hyperparameters
    batch_size_train = 4
    learning_rate = 1e-3
    weight_decay = 1e-5

    # Define training parameters
    n_batches = 2000

    model = RNNAutoencoder(hidden_size=hidden_size, input_size=input_size, n_layers=n_layers)

    # Define weight initialization
    for param in model.parameters():
        if len(param.shape) >= 2:
            nn.init.orthogonal_(param.data)
        else:
            nn.init.normal_(param.data, mean=0.1, std=0.1)

    # Initialize the optimizer with above parameters
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Define the loss function
    loss_fn = nn.MSELoss()  # mean squared error

    # Train and get the resulting loss per iteration
    train(model=model,
          data_loader=data_loader,
          optimizer=optimizer,
          loss_fn=loss_fn,
          n_batches=n_batches,
          batch_size=batch_size_train,
          sequence_nucleotide_length=sequence_nucleotide_length,
          results_handler=results_handler,
          checkpoint_interval=200)

    test(model=model,
         data_loader=data_loader,
         n_batches=4,
         batch_size=1,
         sequence_nucleotide_length=sequence_nucleotide_length)

    results_handler.save_model(model)

    return model


def test_saved_model(model_state_path):
    data_loader = initialize_signal_generator()

    # Define signal simulator parameters
    sequence_nucleotide_length = 100

    # Define architecture parameters
    hidden_size = 16
    input_size = 1      # 1-dimensional signal
    n_layers = 3

    model = RNNAutoencoder(hidden_size=hidden_size, input_size=input_size, n_layers=n_layers)
    model.load_state_dict(torch.load(model_state_path))

    print(model)

    test(model=model,
         data_loader=data_loader,
         n_batches=4,
         batch_size=4,
         sequence_nucleotide_length=sequence_nucleotide_length)


if __name__ == "__main__":
    run()

    # model_path = "/home/ryan/code/nanopore_signal_simulation/output/2018-7-20-10-52-44-4-201/model_checkpoint_10"
    # test_saved_model(model_path)

