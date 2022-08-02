from abc import ABC
from torch import nn
import torch.nn.functional as F  # noqa
from torch.distributions import Categorical


class CNNModel(nn.Module, ABC):
    def __init__(self, input_shape, num_actions):
        super(CNNModel, self).__init__()
        c, w, h = input_shape

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)

        conv1_out_w = self.conv_shape(w, 8, 4)
        conv1_out_h = self.conv_shape(h, 8, 4)
        conv2_out_w = self.conv_shape(conv1_out_w, 4, 2)
        conv2_out_h = self.conv_shape(conv1_out_h, 4, 2)

        flatten_size = conv2_out_w * conv2_out_h * 32

        self.fc = nn.Linear(in_features=flatten_size, out_features=256)
        self.value = nn.Linear(in_features=256, out_features=1)
        self.logits = nn.Linear(in_features=256, out_features=num_actions)  # noqa

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain("relu"))
                layer.bias.data.zero_()

        nn.init.orthogonal_(self.fc.weight, 1.)
        self.fc.bias.data.zero_()
        nn.init.xavier_uniform_(self.value.weight)
        self.value.bias.data.zero_()
        nn.init.xavier_uniform_(self.logits.weight)
        self.logits.bias.data.zero_()

    def forward(self, inputs):
        x = inputs / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        value = self.value(x)
        probs = F.softmax(self.logits(x), dim=-1)
        dist = Categorical(probs)

        return dist, value

    @staticmethod
    def conv_shape(x, kernel_size, stride, padding=0):
        return (x + 2 * padding - kernel_size) // stride + 1


if __name__ == "__main__":
    model = CNNModel((4, 84, 84), 2)
    for m in model.modules():
        print(type(m))
