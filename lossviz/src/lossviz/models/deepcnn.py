from typing import Callable

import jax
import jax.numpy as jnp
import flax.linen as nn


class ConvBlock(nn.Module):

    stride: int
    in_channels: int
    out_channels: int
    activation: Callable = nn.relu
    kernel_init: Callable = nn.initializers.he_normal()


    @nn.compact
    def __call__(self, x, train):

        x = nn.Conv(features=self.in_channels, kernel_size=(3, 3), strides=1, padding='SAME', use_bias=False, kernel_init=self.kernel_init)(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.1, epsilon=1e-5)(x)
        x = self.activation(x)

        x = nn.Conv(features=self.out_channels, kernel_size=(3, 3), strides=self.stride, padding='SAME', use_bias=False, kernel_init=self.kernel_init)(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.1, epsilon=1e-5)(x)
        return self.activation(x)


class DeepCNN(nn.Module):

    activation: Callable = nn.relu
    kernel_init: Callable = nn.initializers.he_normal()


    @nn.compact
    def __call__(self, x, train: bool):

        x = nn.Conv(features=16, kernel_size=(3, 3), strides=1, padding=1, use_bias=False, kernel_init=self.kernel_init)(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.1, epsilon=1e-5)(x)
        x = self.activation(x)

        for _ in range(N-1):
          x = ConvBlock(stride=1, in_channels=16, out_channels=16)(x, train)
        x = ConvBlock(stride=2, in_channels=16, out_channels=32)(x, train)

        for _ in range(N-1):
          x = ConvBlock(stride=1, in_channels=32, out_channels=32)(x, train)
        x = ConvBlock(stride=2, in_channels=32, out_channels=64)(x, train)

        for _ in range(N):
          x = ConvBlock(stride=1, in_channels=64, out_channels=64)(x, train)

        x = nn.avg_pool(x, window_shape=(x.shape[1], x.shape[2]))
        x = x.reshape((x.shape[0], -1))  # Flatten

        x = nn.Dense(features=10, kernel_init=self.kernel_init)(x)  # Output layer for 10 classes


        return x