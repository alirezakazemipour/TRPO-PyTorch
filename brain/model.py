from abc import ABC
from torch import nn
import torch.nn.functional as F  # noqa
from torch.distributions import Categorical


class CNNModel(nn.Module, ABC):
    def __init__(self, input_shape, num_actions):
        super(CNNModel, self).__init__()
        c, w, h = input_shape

        conv1_out_w = self.conv_shape(w, 4, 2)
        conv1_out_h = self.conv_shape(h, 4, 2)
        conv2_out_w = self.conv_shape(conv1_out_w, 4, 2)
        conv2_out_h = self.conv_shape(conv1_out_h, 4, 2)
        flatten_size = conv2_out_w * conv2_out_h * 16

        self.actor = nn.Sequential(nn.Conv2d(in_channels=c, out_channels=16, kernel_size=4, stride=2),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=16, out_channels=16, kernel_size=4, stride=2),
                                   nn.ReLU(),
                                   nn.Flatten(),
                                   nn.Linear(in_features=flatten_size, out_features=20),
                                   nn.ReLU(),
                                   nn.Linear(in_features=20, out_features=num_actions),
                                   nn.Softmax(dim=1)
                                   ).apply(self.init_weights)

        self.critic = nn.Sequential(nn.Conv2d(in_channels=c, out_channels=16, kernel_size=4, stride=2),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=4, stride=2),
                                    nn.ReLU(),
                                    nn.Flatten(),
                                    nn.Linear(in_features=flatten_size, out_features=20),
                                    nn.ReLU(),
                                    nn.Linear(in_features=20, out_features=1),
                                    ).apply(self.init_weights)

    def forward(self, inputs):
        x = inputs / 255.
        value = self.critic(x)
        probs = self.actor(x)
        dist = Categorical(probs)
        return dist, value, probs

    def run_actor(self, inputs):
        x = inputs / 255.
        probs = self.actor(x)
        dist = Categorical(probs)
        return dist, probs

    def run_critic(self, inputs):
        x = inputs / 255.
        return self.critic(x)

    @staticmethod
    def init_weights(layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            layer.bias.data.zero_()
        elif isinstance(layer, nn.Linear):
            if layer.out_features == 20:
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            else:
                nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.zero_()

    @staticmethod
    def conv_shape(x, kernel_size, stride, padding=0):
        return (x + 2 * padding - kernel_size) // stride + 1
