import numpy
import random
import torch
from torch.autograd import Variable
from matplotlib import pyplot

class SineGenerator:
    def __init__(self, pi_fraction_per_training_example, n_steps_per_pi, max_pi, min_pi=0):
        self.n_steps_per_pi = n_steps_per_pi
        self.min_pi = min_pi
        self.max_pi = max_pi
        self.start = self.min_pi*numpy.pi
        self.end = self.max_pi*numpy.pi
        self.pi_fraction_per_training_example = pi_fraction_per_training_example
        self.n_steps = int(round(self.pi_fraction_per_training_example*n_steps_per_pi))
        self.step_size = (self.pi_fraction_per_training_example*numpy.pi)/n_steps_per_pi
        # self.n_leading_values = n_leading_values

        print("n_steps", self.n_steps)
        print("step_size", self.step_size)
        print("range", self.step_size*self.n_steps)

        self.x_data_type = torch.FloatTensor
        self.y_data_type = torch.FloatTensor     # for MSE Loss or BCE loss

    def generate_data(self, batch_size):
        """
        Generate a list of training data tuples with x vector and y. x is a series of sequential values of length
        n_leading_values, y is the following value.
        :param x:
        :param n:
        :return:
        """
        # pyplot.figure(1, figsize=(12, 5))
        # pyplot.ion()  # continuously plot

        # x.shape = (batch_size, time_step, input_size)
        x_step_data = numpy.zeros([batch_size, self.n_steps, 1])
        y_step_data = numpy.zeros([batch_size, 1, 1])
        x_data = numpy.zeros([batch_size, self.n_steps, 1])
        y_data = numpy.zeros([batch_size, 1, 1])

        for i in range(batch_size):
            offset = numpy.pi/(self.n_steps_per_pi)
            start = random.random()*(self.max_pi*numpy.pi-self.pi_fraction_per_training_example*numpy.pi-offset)
            end = start + self.pi_fraction_per_training_example*numpy.pi

            # print(self.n_steps,start,end,(end-start)/numpy.pi)

            steps = numpy.linspace(start, end, self.n_steps+1, dtype=numpy.float32)
            x_steps = steps[:-1]
            y_steps = steps[-1:]

            # print(x_steps.shape)
            # print(y_steps.shape)
            # print(numpy.expand_dims(x_steps, axis=1).shape)
            # print(numpy.expand_dims(y_steps, axis=1).shape)

            x_step_data[i,:,:] = numpy.expand_dims(x_steps, axis=1)
            y_step_data[i,:,:] = numpy.expand_dims(y_steps, axis=1)

            sine_steps = numpy.sin(steps)*0.9 + 1
            x = sine_steps[:-1]
            y = sine_steps[-1:]
            x_data[i,:,:] = numpy.expand_dims(x, axis=1)
            y_data[i,:,:] = numpy.expand_dims(y, axis=1)

            # print(x)
            # print(y)
        #     pyplot.plot(steps[:-1],x,color="blue",alpha=i/batch_size,marker='.')
        #     pyplot.plot(steps[-1:],y,color="red",alpha=i/batch_size,marker='.')
        #     pyplot.draw()
        #     pyplot.pause(0.5)
        #
        # pyplot.ioff()
        # pyplot.show()
        # print(x_step_data.shape)
        # print(y_step_data.shape)

        x_data = Variable(torch.from_numpy(x_data).type(self.x_data_type))
        y_data = Variable(torch.from_numpy(y_data).type(self.y_data_type))

        return x_step_data, y_step_data, x_data, y_data


if __name__ == "__main__":
    generator = SineGenerator(pi_fraction_per_training_example=0.5, n_steps_per_pi=40, max_pi=8)

    data = generator.generate_data(2)

    print(data[0].T)
    print(data[1].T)
