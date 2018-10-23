from models.ResNetAutoencoder import ResNetAutoencoder, BasicDecodeBlock, BasicEncodeBlock
from modules.SignalGenerator import *
from handlers.FileManager import FileManager
from matplotlib import pyplot
from torch import nn
from torch import optim
from os import path
import torch
import datetime
import numpy
import csv


class ResultsHandler:
    def __init__(self):
        self.datetime_string = '-'.join(list(map(str, datetime.datetime.now().timetuple()))[:-1])
        self.subdirectory_name = "training_" + self.datetime_string

        self.output_directory_name = "output/"
        self.directory = path.join(self.output_directory_name, self.subdirectory_name)

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


def plot_prediction(x, y, y_predict):
    fig, axes = pyplot.subplots(nrows=3, gridspec_kw={'height_ratios': [1, 1, 10]})

    x_data = x.data.numpy()[0, :, :, :].squeeze()
    y_target_data = y.data.numpy()[0, :, :].squeeze()
    y_predict_data = y_predict.data.numpy()[0, :, :].squeeze()

    # print(x_data.shape)

    axes[2].imshow(x_data)
    axes[1].imshow(y_target_data)
    axes[0].imshow(y_predict_data)

    axes[2].set_ylabel("x")
    axes[1].set_ylabel("y")
    axes[0].set_ylabel("y*")

    pyplot.show()
    pyplot.close()


# def loss_CE(y_predict, y_target, loss_fn):
#     # x shape = (n, length)
#     # y shape = (n, length)
#
#     n, l = y_predict.shape
#
#     y_target = y_target.view([n])
#
#     # print(y_predict, y_target)
#
#     loss = loss_fn(y_predict, y_target)
#
#     return loss


def train_batch(model, x, y, optimizer, loss_fn):
    # Run forward calculation
    y_predict = model.forward(x)

    # Compute loss.
    # loss = loss_fn(y_predict, y)
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


def convert_to_spectrogram(signals, n_steps_per_fft=20, n_frequencies=127):
    spectrograms = list()
    for signal in signals:
        frequency, time, z_component = stft(signal, nperseg=n_steps_per_fft, nfft=n_frequencies)

        spectrogram = numpy.zeros([len(z_component), len(z_component[0])])
        for i, z_column in enumerate(z_component):
            z_column = numpy.abs(z_column)
            for j in range(len(z_column)):
                spectrogram[i, j] = z_column[j]

        spectrogram = numpy.expand_dims(spectrogram, axis=0)
        spectrogram = numpy.expand_dims(spectrogram, axis=0)
        spectrograms.append(spectrogram)

    spectrograms = numpy.concatenate(spectrograms, axis=0)

    return spectrograms


def train(model, data_generator, optimizer, loss_fn, n_batches, n_epochs, results_handler, checkpoint_interval, batch_size, use_gpu, fixed_length, sequence_length):
    model.train()

    losses = list()

    for e in range(n_epochs):
        for b in range(n_batches):
            signals, sequences = data_generator.generate_fixed_size_batch(batch_size=batch_size,
                                                                          sequence_length=sequence_length,
                                                                          fixed_length=fixed_length)

            x = convert_to_spectrogram(signals)

            if use_gpu:
                x = torch.cuda.FloatTensor(x)
            else:
                x = torch.FloatTensor(x)

            # print("x1", x.shape)
            # print("y1", y.shape)

            n, c, h, w = x.shape
            # x = x.view([n,1,h,w])

            # print("x", x.shape)
            # print("y", y.shape)

            # n, h, w = y.shape
            # y = y.view([n,1,h,w])

            # print("x2", x.shape)
            # print("y2", y.shape)

            # expected convolution input = (batch, channel, H, W)
            # y_predict shape = (batch_size, seq_len, hidden_size*num_directions)
            loss, y_predict = train_batch(model=model, x=x, y=x, optimizer=optimizer, loss_fn=loss_fn)
            losses.append(loss)

            print(b, loss)

            if b % checkpoint_interval == 0:
                results_handler.save_model(model)

                input_example = x[0,0,:,:]
                # input_example = input_example.permute([1,2,0])
                input_example = input_example.cpu().data.numpy()/255

                # print(paths[0])

                output_example = y_predict[0,0,:,:]
                # output_example = output_example.permute([1,2,0])
                output_example = output_example.cpu().data.numpy()/255

                fig, axes = pyplot.subplots(nrows=2)

                axes[0].imshow(input_example)
                axes[1].imshow(output_example)

                pyplot.show()
                pyplot.close()

                pyplot.plot(losses)
                pyplot.show()
                pyplot.close()

    return losses


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

    noise_sigma = 35*average_current_difference
    event_variation_sigma = 0.5*average_current_difference
    interpolate_rate = 1

    noise_model = GaussianNoise(mu=0, sigma=noise_sigma)
    event_variation_model = GaussianNoise(mu=0, sigma=event_variation_sigma)
    duration_model = GaussianDuration(mu=5, sigma=1, minimum_duration=2)

    signal_generator = SignalGenerator(k=k,
                                       kmer_means_dictionary=kmer_means,
                                       noise_model=noise_model,
                                       event_variation_model=event_variation_model,
                                       duration_model=duration_model,
                                       interpolate_rate=interpolate_rate)

    return signal_generator


def run(load_model=False, model_state_path=None):
    signal_generator = initialize_signal_generator()

    results_handler = ResultsHandler()

    # Architecture parameters
    # hidden_size = 16
    # input_channels = 1      # 1-dimensional signal
    # output_size = 5         # '-','A','C','T','G' one hot vector
    # n_layers = 3

    # Hyperparameters
    learning_rate = 1e-5
    weight_decay = 1e-5

    # Training parameters\
    use_gpu = True
    batch_size = 4
    n_batches = 1000
    n_epochs = 1

    checkpoint_interval = 50


    model = ResNetAutoencoder(encode_block=BasicEncodeBlock,
                              decode_block=BasicDecodeBlock,
                              layers=[2, 2, 2, 2])
    print(model)

    if use_gpu:
        model.cuda()

    # Initialize the optimizer with above parameters
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Define the loss function
    loss_fn = nn.MSELoss()
    # loss_fn = nn.CrossEntropyLoss()

    if load_model:
        # get weight parameters from saved model state
        model.load_state_dict(torch.load(model_state_path))

    # Train and get the resulting loss per iteration
    losses = train(model=model,
                   data_generator=signal_generator,
                   optimizer=optimizer,
                   loss_fn=loss_fn,
                   n_batches=n_batches,
                   batch_size=batch_size,
                   sequence_length=100,
                   fixed_length=630,
                   n_epochs=n_epochs,
                   results_handler=results_handler,
                   checkpoint_interval=checkpoint_interval,
                   use_gpu=use_gpu)

    # test(model=model,
    #      data_loader=data_loader,
    #      n_batches=4)

    results_handler.save_model(model)
    results_handler.save_plot(losses)


if __name__ == "__main__":
    run()