from typing import Callable

import jax
import jax.numpy as jnp
import flax.linen as nn


class ResidualBlock(nn.Module):

    in_channels: int
    kernel_init: Callable = nn.initializers.kaiming_normal()

    @nn.compact
    def __call__(self, x, train):
        residual = x

        x = nn.Conv(features=self.in_channels,
                    kernel_size=(3, 3),
                    strides=1,
                    padding="SAME",
                    use_bias=False,
                    kernel_init=self.kernel_init)(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.1, epsilon=1e-5)(x)
        x = nn.relu(x)

        x = nn.Conv(features=self.in_channels,
                    kernel_size=(3, 3),
                    strides=1,
                    padding="SAME",
                    use_bias=False,
                    kernel_init=self.kernel_init)(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.1, epsilon=1e-5)(x)

        x = x + residual

        return nn.relu(x)


class DownSampleResidualBlock(nn.Module):

    in_channels: int
    out_channels: int
    kernel_init: Callable = nn.initializers.kaiming_normal()


    @nn.compact
    def __call__(self, x, train):
        residual = x

        x = nn.Conv(features=self.in_channels,
                    kernel_size=(3, 3),
                    strides=1,
                    padding="SAME",
                    use_bias=False,
                    kernel_init=self.kernel_init)(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.1, epsilon=1e-5)(x)
        x = nn.relu(x)

        x = nn.Conv(features=self.out_channels,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    padding=(((1, 1), (1, 1))),
                    use_bias=False,
                    kernel_init=self.kernel_init)(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.1, epsilon=1e-5)(x)

        x = x + self.pad_identity(residual)

        return nn.relu(x)


    @nn.nowrap
    def pad_identity(self, x):
        # Pad identity connection when downsampling
        return jnp.pad(
            x[:, ::2, ::2, ::],
            ((0, 0), (0, 0), (0, 0), (self.out_channels // 4, self.out_channels // 4)),
            "constant",
        )


class ResidualCNN(nn.Module):

    kernel_init: Callable = nn.initializers.kaiming_normal()

    @nn.compact
    def __call__(self, x, train: bool):

        x = nn.Conv(features=16, kernel_size=(3, 3), strides=1, padding="SAME", use_bias=False, kernel_init=self.kernel_init)(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.1, epsilon=1e-5)(x)
        x = nn.relu(x)

        for _ in range(N-1):
          x = ResidualBlock(in_channels=16)(x, train)
        x = DownSampleResidualBlock(in_channels=16, out_channels=32)(x, train)

        for _ in range(N-1):
          x = ResidualBlock(in_channels=32)(x, train)
        x = DownSampleResidualBlock(in_channels=32, out_channels=64)(x, train)

        for _ in range(N):
          x = ResidualBlock(in_channels=64)(x, train)

        x = nn.avg_pool(x, window_shape=(x.shape[1], x.shape[2]))
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=10, kernel_init=self.kernel_init)(x)  # Output layer for 10 classes

        return x