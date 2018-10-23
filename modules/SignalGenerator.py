# Author: Ryan Lorig-Roach

from matplotlib import pyplot
import math
import numpy
import random
import sys
from scipy.signal import stft, istft
import copy
import pysam



class Event:
    def __init__(self, mean_current, duration):
        self.mean = mean_current
        self.duration = duration


class GaussianNoise:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def sample(self, x):
        """
        Add noise to a kmer current value, sampled from a gaussian with mean=mu and stdev=sigma
        :param x:
        :return:
        """
        x += numpy.random.normal(loc=self.mu, scale=self.sigma)

        return x


class IdentityFunction:
    def sample(self, x):
        return x


class GaussianDuration:
    """
    Chooses duration for synthetic segments based on any of several models, independent or otherwise
    """
    def __init__(self, mu, sigma, minimum_duration):
        self.mu = mu
        self.sigma = sigma
        self.minimum = minimum_duration

    def sample(self):
        t = numpy.random.normal(loc=self.mu, scale=self.sigma)
        t = max(t, self.minimum)

        return t


class SequenceGenerator:
    def __init__(self):
        self.characters = ['A','C','G','T']

    def generate_sequence(self, length):
        sequence = list()
        for i in range(length):
            sequence.append(random.choice(self.characters))

        return sequence


class SignalGenerator:
    """
    Generates a simulated signal for any given DNA sequence. Requires a table of mean current values for all kmers.
    """

    def __init__(self, k, kmer_means_dictionary, event_variation_model, duration_model, noise_model, interpolate_rate=0.5, sample_rate=1):
        self.k = k
        self.kmer_means = kmer_means_dictionary
        self.event_variation_model = event_variation_model
        self.noise_model = noise_model
        self.duration_model = duration_model
        self.sample_rate = sample_rate
        self.interpolate_rate = interpolate_rate

        self.sequence_generator = SequenceGenerator()

    def rasterize_event_sequence(self, events):
        signal = list()
        for event in events:
            self.append_signal(signal=signal, event=event)

        return signal

    def append_signal(self, signal, event, max_length=sys.maxsize):
        n = round(event.duration / self.sample_rate)

        for i in range(n):
            data_point = self.noise_model.sample(event.mean)

            if len(signal) < max_length:
                signal.append(data_point)

            random_float = random.uniform(a=0, b=1.0)
            # randomly generate a datapoint that interpolates between the last 2 datapoints
            if len(signal) > 1 and len(signal) < max_length and random_float < self.interpolate_rate:
                inter_point = random.uniform(signal[-1], signal[-2])
                inter_point = self.noise_model.sample(inter_point)
                signal.append(signal[-1])
                signal[-2] = inter_point

        return signal

    def generate_signal_from_sequence(self, sequence, pad_signals=False):
        """
        Given a sequence, use standard kmer signal values to estimate its expected signal (over all kmers in sequence).
        Additionally, generate random event durations.
        """
        events = list()
        sequence = ''.join(sequence)

        for i in range(0, len(sequence) - self.k + 1):
            kmer = sequence[i:i + self.k]
            current = self.kmer_means[kmer]
            current = self.event_variation_model.sample(current)
            duration = self.duration_model.sample()

            event = Event(mean_current=current, duration=duration)
            events.append(event)

        signal = self.rasterize_event_sequence(events)

        return signal, events

    def pad_signals(self, event_series, signals, max_length):
        for s,signal in enumerate(signals):
            current = event_series[s][-1].mean   # find last event's current
            duration = max_length - len(signal)
            event = Event(mean_current=current, duration=duration)

            self.append_signal(signal=signal, event=event, max_length=max_length)

        return signals

    def fit_signals_to_length(self, event_series, signals, length):
        for s, signal in enumerate(signals):
            # print("DURATION: ", len(signal),length)

            if len(signal) > length:
                # too long, trim signal
                signal = signal[:length]

            elif len(signal) < length:
                # too short
                current = event_series[s][-1].mean  # find last event's current

                duration = length - len(signal)
                event = Event(mean_current=current, duration=duration)

                signal = self.append_signal(signal=signal, event=event, max_length=length)

            else:
                pass

            signals[s] = signal

        return signals

    def generate_batch(self, batch_size, sequence_length, sequence=None):
        if sequence_length < self.k:
            sys.exit("Sequence length (%d) must be longer than k (%d)"%(sequence_length, self.k))

        # x shape is (batch_size, time_step, input_size)

        if sequence is None:
            sequences = list()
        else:
            sequences = [sequence]*batch_size

        signals = list()
        event_series = list()

        max_signal_length = 0
        for i in range(batch_size):
            sequence = self.sequence_generator.generate_sequence(sequence_length)
            signal, events = self.generate_signal_from_sequence(sequence)

            sequences.append(sequence)
            signals.append(signal)
            event_series.append(events)

            if len(signal) > max_signal_length:
                max_signal_length = len(signal)

        # print(max_signal_length)
        signals = self.pad_signals(event_series=event_series, signals=signals, max_length=max_signal_length)

        return signals, sequences

    def generate_fixed_size_batch(self, batch_size, sequence_length, fixed_length, sequence=None):
        if sequence_length < self.k:
            sys.exit("Sequence length (%d) must be longer than k (%d)" % (sequence_length, self.k))

        # x shape is (batch_size, time_step, input_size)
        # x_data = numpy.zeros(shape=batch_size, )

        if sequence is None:
            sequences = [self.sequence_generator.generate_sequence(sequence_length) for i in range(batch_size)]
        else:
            sequences = [sequence]*batch_size

        signals = list()
        event_series = list()

        max_signal_length = 0
        for i in range(batch_size):
            sequence = sequences[i]
            signal, events = self.generate_signal_from_sequence(sequence)

            sequences.append(sequence)
            signals.append(signal)
            event_series.append(events)

            if len(signal) > max_signal_length:
                max_signal_length = len(signal)

        # print(max_signal_length)
        # self.pad_signals(event_series=event_series, signals=signals, max_length=max_signal_length)
        signals = self.fit_signals_to_length(event_series=event_series, signals=signals, length=fixed_length)

        return signals, sequences


class KmerTableReader:
    def __init__(self, k, kmer_table_path):
        self.k = k
        self.n_kmers = 4**self.k
        self.minimum = sys.maxsize
        self.maximum = -sys.maxsize
        self.kmer_means = self.read_standard_kmer_means(kmer_table_path)

    def read_standard_kmer_means(self, kmer_means_file_path):
        """
        Read a file containing the list of all kmers and their expected signal means (2 columns, with headers), store as
        a dictionary of kmer:mean
        """

        standard_kmer_means = dict()

        with open(kmer_means_file_path, 'r') as file:
            file.readline()

            for line in file:
                kmer, mean = line.strip().split()

                mean = float(mean)

                # update minimum and maximum kmer current values
                if mean > self.maximum:
                    self.maximum = mean

                if mean < self.minimum:
                    self.minimum = mean

                # handle user errors
                if len(kmer)!=self.k:
                    sys.exit("ERROR: kmer length not equal to parameter K")

                try:
                    standard_kmer_means[kmer] = float(mean)
                except:
                    print("WARNING: duplicate kmer found in standard means reference list: %s" % kmer)

        return standard_kmer_means

    def get_kmer_count(self):
        return self.n_kmers

    def get_range(self):
        return self.maximum - self.minimum

    def get_kmer_current_dictionary(self):
        return self.kmer_means


def main():
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

    # master_sequence = "ATCGCCGATTCTGCATGCATAAACTGTGGGACTCATGCCCATGTACTCTTTGTAGCTGCTCCCATGACTGATCGACTTTTGCATGATTCATCTACTATGAATTCTGCTGGACTGCGTATGTTACTGGCCGATTATGTCTGAGTCCGTTCTTAATATCCGTTCGCTAGCTATCTTACTGGACTATTATCGGCCATTCTGATCGT"
    # sequence = "ATCGCCGATTCTGCATGCATAAACTGTGGGACTCATGCCCATGTACTCTTTGTAGCTGCTCCCATGACTGATCGACTTTTGCATGATTCATCTACTATG"
    # sequence = "GCTGCTCCCATGACTGATCGACTTTTGCATGATTCATCTACTATGAATTCTGCTGGACTGCGTATGTTACTGGCCGATTATGTC"
    # sequence = "TGCTGGACTGCGTATGTTACTGGCCGATTATGTCTGAGTCCGTTCTTAATATCCGTTCGCTAGCTATCTTACTGGACTATTATCGGCCATTCTGATCGT"

    batch_size = 2
    fixed_length = 700
    nperseg = 20
    nfft = 64

    signals, sequences = signal_generator.generate_fixed_size_batch(batch_size=batch_size, sequence_length=100, fixed_length=fixed_length)
    # signals, sequences = signal_generator.generate_batch(batch_size=4, sequence_length=8)

    fig, axes = pyplot.subplots(nrows=batch_size, ncols=4)
    for s,signal in enumerate(signals):
        signal = numpy.array(signal)
        frequency, time, z_component = stft(signal, nperseg=nperseg, nfft=nfft)

        reconstructed_signal = istft(z_component, nperseg=nperseg, nfft=nfft)

        length = math.ceil(fixed_length/(nperseg/2))
        print(length, len(frequency), len(time), len(z_component), len(z_component[0]))

        axes[s][0].plot(signal)
        axes[s][1].pcolormesh(time, frequency, numpy.abs(z_component))
        axes[s][3].plot(reconstructed_signal[1][:700])

        spectrogram = numpy.zeros([len(z_component), len(z_component[0])])
        for i,z_column in enumerate(z_component):
            z_column = numpy.abs(z_column)
            for j in range(len(z_column)):
                # print(frequency[j])
                spectrogram[i,j] = z_column[j]
                # spectrogram[i][j] = z_component[j]

        axes[s][2].imshow(spectrogram)

        axes[s][0].get_shared_x_axes().join(axes[s][0], axes[s][3])
        axes[s][0].get_shared_y_axes().join(axes[s][0], axes[s][3])

        # print(frequency)
        # axes[s].axis("off")

    pyplot.show()

if __name__ == "__main__":
    main()