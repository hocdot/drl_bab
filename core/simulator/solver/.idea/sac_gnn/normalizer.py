import numpy as np
import torch

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    # Updates the mean, var and count using the previous mean, var, count and batch values.
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

class RunningMeanStd:

    # Tracks the mean, variance and count of values.
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-3, shape=(), is_correct_nan=False):
        # Tracks the mean, variance and count of values.
        self.mean    = torch.zeros(shape, dtype=torch.float)
        self.var     = torch.ones(shape, dtype=torch.float)
        self.count   = epsilon
        self.epsilon = epsilon
        self.shape   = shape
        self.is_correct_nan = is_correct_nan

    def correct_nan(self, x):
        x[torch.where(x[:, 0].isnan())[0]] = self.mean[0] - 3 * torch.sqrt(self.var[0])
        x[torch.where(x[:, 1].isnan())[0]] = self.mean[1] + 3 * torch.sqrt(self.var[1])
        x[torch.where(x[:, 3].isnan())[0]] = self.mean[3] - 3 * torch.sqrt(self.var[3])
        return x

    def update(self, x):
        # wrap scalar reward to vector one-dimensional
        if type(x) in [float, np.float64]:
            x = torch.tensor([x])
        # wrap single observation to batch
        elif len(x.shape) == 1:
            x = x[None, :]
        # wrap 4 dimensional observation to 2 dimensional
        elif len(x.shape) == 4:
            x = x.view(-1, self.shape[0])
        # correct nan
        if self.is_correct_nan:
            x = self.correct_nan(x)
        # updates the mean, var and count from a batch of samples.
        batch_mean  = torch.mean(x, dim=0)
        batch_var   = torch.var(x, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        # updates from batch mean, variance and count moments.
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    def normalize(self, x):
        # wrap single observation to batch
        if type(x) in [float, np.float64]:
            x = torch.tensor([x])
            x_shape = -1
        else:
            x_shape = x.shape
        x       = x.view(-1, self.shape[0])
        norm_x  = (x - torch.tensor(self.mean, dtype=torch.float)) / \
                   torch.sqrt(torch.tensor(self.var, dtype=torch.float) + self.epsilon)
        # correct nan
        if self.is_correct_nan:
            x = self.correct_nan(x)
        if x.shape == -1:
            return x[0].item()
        else:
            return norm_x.view(x_shape)

class RunningMinMax:

    # Tracks the min, max
    def __init__(self, epsilon=1e-3, shape=()):
        # Tracks the min, max
        self.min     = torch.zeros(shape, dtype=torch.float)
        self.max     = torch.ones(shape, dtype=torch.float)
        self.epsilon = epsilon
        self.shape   = shape

    def update(self, x):
        # wrap scalar reward to vector one-dimensional
        if type(x) in [float, np.float64]:
            x = torch.tensor([x])
        # wrap single observation to batch
        elif len(x.shape) == 1:
            x = x[None, :]
        # wrap 4 dimensional observation to 2 dimensional
        elif len(x.shape) == 4:
            x = x.view(-1, self.shape[0])
        # correct nan
        x = self.correct_nan(x)
        # update the max min
        batch_min = torch.min(x, dim=0).values
        batch_max = torch.max(x, dim=0).values
        self.min = torch.min(self.min, batch_min)
        self.max = torch.max(self.max, batch_max)

    def correct_nan(self, x):
        x[torch.where(x[:, 0].isnan())[0]] = self.min[0]
        x[torch.where(x[:, 1].isnan())[0]] = self.max[1]
        x[torch.where(x[:, 3].isnan())[0]] = self.min[3]
        return x

    def normalize(self, x):
        x_shape = x.shape
        x       = x.view(-1, self.shape[0])
        # correct nan
        x       = self.correct_nan(x)
        # normalize
        norm_x  = (x - self.min) / \
                  (self.max - self.min + self.epsilon)
        return norm_x.view(x_shape)
