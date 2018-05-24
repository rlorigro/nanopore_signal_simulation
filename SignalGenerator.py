# Author: Ryan Lorig-Roach

from matplotlib import pyplot
import numpy
import random
import sys
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


class SignalGenerator:
    '''
    Generates a simulated signal for any given DNA sequence. Requires a table of mean current values for all kmers.
    '''

    def __init__(self, k, kmer_means_dictionary, event_variation_model, duration_model, noise_model, interpolate_rate=0.5, sample_rate=0.3):
        self.k = k
        self.kmer_means = kmer_means_dictionary
        self.event_variation_model = event_variation_model
        self.noise_model = noise_model
        self.duration_model = duration_model
        self.sample_rate = sample_rate
        self.interpolate_rate = interpolate_rate

    def rasterize_events(self, events):
        signal = list()
        for event in events:
            n = round(event.duration/self.sample_rate)

            for i in range(n):
                data_point = self.noise_model.sample(event.mean)
                signal.append(data_point)

                random_float = random.uniform(a=0, b=1.0)
                if len(signal) > 1 and random_float < self.interpolate_rate:

                    print(signal[-2:])
                    inter_point = random.uniform(signal[-1], signal[-2])
                    inter_point = self.noise_model.sample(inter_point)
                    signal.append(signal[-1])
                    signal[-2] = inter_point

                    print(signal[-3:])

        return signal

    def generate_signal_from_sequence(self, sequence):
        '''
        Given a sequence, use standard kmer signal values to estimate its expected signal (over all kmers in sequence).
        Additionally, generate random event durations.
        '''

        events = list()
        sequence = ''.join(sequence)

        for i in range(0, len(sequence) - self.k + 1):
            kmer = sequence[i:i + self.k]

            current = self.kmer_means[kmer]
            current = self.event_variation_model.sample(current)

            duration = self.duration_model.sample()

            event = Event(mean_current=current, duration=duration)

            events.append(event)

        signal = self.rasterize_events(events)

        return signal


class KmerTableReader:
    def __init__(self, k, kmer_table_path):
        self.k = k
        self.n_kmers = 4**self.k
        self.minimum = sys.maxsize
        self.maximum = -sys.maxsize
        self.kmer_means = self.read_standard_kmer_means(kmer_table_path)

    def read_standard_kmer_means(self, kmer_means_file_path):
        '''
        Read a file containing the list of all kmers and their expected signal means (2 columns, with headers), store as
        a dictionary of kmer:mean
        '''

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


if __name__ == "__main__":
    k = 6
    kmer_table_path = "/Users/saureous/data/nanopore/kmer_means"
    kmer_table_handler = KmerTableReader(k=6, kmer_table_path=kmer_table_path)

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

    sequence = "GATTACAGATTACAGATTACAGATTACA"
    # signals = list()

    n_repeats = 1
    for i in range(n_repeats):
        signal = signal_generator.generate_signal_from_sequence(sequence)
        pyplot.plot(signal)

    pyplot.show()

