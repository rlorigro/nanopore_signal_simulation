from __future__ import unicode_literals, print_function, division
from io import open
import glob
import unicodedata
import string
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def findFiles(path):
    return glob.glob(path)


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s, all_letters):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn' and c in all_letters)


# Read a file and split into lines
def readLines(filename, all_letters):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line, all_letters) for line in lines]


def read_categorical_names_as_dictionary(path, all_letters):
    # Build the category_lines dictionary, a list of names per language
    categorical_names = {}
    all_categories = []

    for filename in findFiles(path):
        category = filename.split('/')[-1].split('.')[0]
        all_categories.append(category)
        lines = readLines(filename, all_letters)
        categorical_names[category] = lines

    return categorical_names, all_categories


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter, n_letters):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line, n_letters):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return Variable(tensor)


def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = int(top_i[0])
    return all_categories[category_i], category_i


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


def randomChoice(l):
    return l[random.randint(0, len(l)-1)]


def randomTrainingExample(categorical_names):
    category = randomChoice(all_categories)
    line = randomChoice(categorical_names[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = lineToTensor(line, n_letters)

    return category, line, category_tensor, line_tensor


def train(model, category_tensor, name_tensor, loss_fn):
    hidden = rnn.initHidden()

    rnn.zero_grad()
    output = None

    for i in range(name_tensor.size()[0]):
        output, hidden = model.forward(name_tensor[i], hidden)

    loss = loss_fn(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


if __name__ == "__main__":
    # --- PROCESS INPUT DATA ---

    path = 'data/names/*.txt'

    all_letters = string.ascii_letters+" .,;'"
    n_letters = len(all_letters)

    categorical_names, all_categories = read_categorical_names_as_dictionary(path, all_letters)
    n_categories = len(all_categories)

    print(findFiles(path))
    print(unicodeToAscii('Ślusàrski', all_letters))
    print(n_letters)
    print(categorical_names['Irish'][:5])
    print(letterToTensor('J', n_letters))
    print(lineToTensor('Jones', n_letters).size())

    # --- INITIALIZE MODEL AND TRAINING TENSORS ---

    n_hidden = 128
    rnn = RNN(n_letters, n_hidden, n_categories)

    input = lineToTensor('Albert', n_letters)
    hidden = Variable(torch.zeros(1, n_hidden))

    output, next_hidden = rnn.forward(input[0], hidden)

    print(input[0])
    print(output)
    print(categoryFromOutput(output.data))

    for i in range(10):
        category, name, category_tensor, line_tensor = randomTrainingExample(categorical_names)
        print('category =', category, '| name =', name)

    learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn
    loss_fn = nn.NLLLoss

    n_iters = 100000
    print_every = 5000
    plot_every = 1000

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []

    start = time.time()

    for iter in range(1, n_iters + 1):
        category, line, category_tensor, name_tensor = randomTrainingExample(categorical_names)
        output, loss = train(model=rnn, loss_fn=loss_fn, category_tensor=category_tensor, name_tensor=name_tensor)
        current_loss += loss

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    plt.figure()
    plt.plot(all_losses)