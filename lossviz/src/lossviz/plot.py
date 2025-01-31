import jax.numpy as jnp
from torch.utils import data

from utils import *

__all__ = ["generate_2D_plot", "generate_3D_plot"]

def generate_2D_plot(model_state,
                      batch_stats,
                      train_loader_unshuffled: data.DataLoader,
                      num_batches, x1=-0.03, x2=0.03):

    """
    Generates a 2D plot of the loss landscape for a given model state and batch statistics.

    Args:
        model_state: The current state of the model.
        batch_stats: The current batch statistics of the model.
        train_loader_unshuffled: The training data loader, which should not be shuffled.
        num_batches: The number of batches to compute the loss over.
        x1: The lower bound of the x-axis.
        x2: The upper bound of the x-axis.

    Returns:
        A tuple containing the x-coordinates and y-coordinates of the plot.
    """ 

    direction = _generate_random_weights(model_state.params, key)

    init_weights = model_state.params

    coords = jnp.linspace(x1, x2, 500)
    losses = []

    for x in coords:
      new_params = _perturb_params(init_weights, direction, x)
      loss = _compute_loss(model_state, new_params, batch_stats, train_loader_unshuffled, num_batches)
      losses.append(loss)

    return coords, losses



def generate_3D_plot(model_state, 
                      batch_stats, 
                      train_loader_unshuffled: data.DataLoader, 
                      num_batches, x1=-0.03, x2=0.03):
    """
    Generates a 3D plot of the loss landscape for a given model state and batch statistics.

    Args:
      model_state: The current state of the model.
      batch_stats: The current batch statistics of the model.
      train_loader_unshuffled: The training data loader, which should not be shuffled.
      num_batches: The number of batches to compute the loss over.
      x1: The lower bound of the x-axis.
      x2: The upper bound of the x-axis.

    Returns:
      A tuple containing the x-coordinates, y-coordinates, and z-coordinates of the plot.
    """

    direction_x = _generate_random_weights(model_state.params, key)
    sub, _ = jax.random.split(key)
    direction_y = _generate_random_weights(model_state.params, sub)

    init_weights = model_state.params

    X, Y = jnp.meshgrid(jnp.linspace(x1, x2, 20), jnp.linspace(x1, x2, 20))

    losses = jnp.empty_like(X)
    count = 0
    total = X.shape[0] * X.shape[1]

    for i in range(X.shape[0]):
      for j in range(X.shape[1]):
        new_params = _perturb_params(init_weights, [direction_x, direction_y], [X[i,j], Y[i,j]])
        loss = _compute_loss(model_state, new_params, batch_stats, train_loader_unshuffled, num_batches)
        losses = losses.at[i,j].set(loss)
        count += 1


    return X, Y, losses