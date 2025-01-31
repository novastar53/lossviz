import jax
import optax
import jax.numpy as jnp
from torch.utils import data

import losscape
import matplotlib.pyplot as plt


__all__ = ["compute_loss", "generate_random_weights", "normalize_direction", "perturb_params", ]


@jax.jit
def _loss(model_state, params, batch_stats, x, y):
    """
    Computes the loss for a given batch of data.

    Args:
        model_state: The current state of the model.
        params: The current parameters of the model.
        batch_stats: The current batch statistics of the model.
        x: The input data.
        y: The target labels.

    Returns:
        The loss for the given batch of data.
    """

    variables = {"params": params, "batch_stats": batch_stats}
    logits = model_state.apply_fn(variables, x, train=False)
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == y)
    one_hot = jax.nn.one_hot(y, num_classes=10)
    return jnp.mean(optax.softmax_cross_entropy(logits, one_hot))


def compute_loss(model_state, params, batch_stats, train_loader_unshuffled: data.DataLoader, num_batches):
    """
    Computes the average loss over a specified number of batches from a training data loader.

    Args:
        model_state: The current state of the model.
        params: The current parameters of the model.
        batch_stats: The current batch statistics of the model.
        train_loader_unshuffled: The training data loader, which should not be shuffled.
        num_batches: The number of batches to compute the loss over.

    Returns:
        The average loss over the specified number of batches.
    """

    batch_idx = 0
    loss = 0

    for images, labels in train_loader_unshuffled:

      loss += _loss(model_state, params, batch_stats, images, labels)
      batch_idx += 1

      if batch_idx >= num_batches:
        break

    loss = loss / num_batches

    return loss


def generate_random_weights(weights, key):

    return jax.tree_util.tree_map(lambda x: jax.random.normal(key, x.shape), weights)


def normalize_direction(direction, weights):
    """
    Normalizes the direction of the model parameters by scaling the direction vector by the relative magnitude of the parameter weights.

    This function is used to ensure that the perturbation direction has the same relative magnitude as the parameter weights, which can help improve the stability and convergence of optimization algorithms.

    Args:
        direction: A tree-like structure containing the direction vectors for each parameter.
        weights: A tree-like structure containing the current parameter values.

    Returns:
        A tree-like structure containing the normalized direction vectors.
    """

    w_norms = jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x.flatten(), ord=2), weights)
    d_norms = jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x.flatten(), ord=2), direction)

    norm_weights = jax.tree_util.tree_map(lambda d, w_norm, d_norm: d*w_norm/d_norm, direction, w_norms, d_norms)

    return norm_weights


def perturb_params(params, direction, step):
    """
    Perturbs the model parameters by adding a step in the given direction(s).

    If the direction is a tuple of two elements, the step is applied in both directions. Otherwise, the step is applied in the single direction.

    Args:
        params: The model parameters to be perturbed.
        direction: The direction(s) in which to perturb the parameters.
        step: The step size(s) to apply in the given direction(s).

    Returns:
        The perturbed model parameters.
    """

    if len(direction) == 2:
      new_params = jax.tree_util.tree_map(lambda w, d0, d1: w + step[0]*d0 + step[1]*d1, params, direction[0], direction[1])
    else:
      new_params = jax.tree_util.tree_map(lambda w, d: w + step*d, params, direction)

    return new_params


