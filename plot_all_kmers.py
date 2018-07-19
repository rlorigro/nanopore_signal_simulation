import numpy
import csv
from matplotlib import pyplot


kmer_file_path = "/home/ryan/data/Nanopore/kmerMeans"

with open(kmer_file_path, 'r') as file:
    reader = csv.reader(file, delimiter='\t')

    means = list()
    for l,line in enumerate(reader):
        if l == 0:
            continue

        mean = float(line[1])
        means.append(mean)

bins = numpy.arange(0, 200, step=1)
frequencies, bins = numpy.histogram(means, bins=bins)
center = (bins[:-1]+bins[1:])/2

axes = pyplot.axes()
axes.bar(center, frequencies, align="center")

axes.set_title("Distribution of standard 5-mer means")
axes.set_xlabel("Current (pA)")
axes.set_ylabel("Frequency")
pyplot.show()
