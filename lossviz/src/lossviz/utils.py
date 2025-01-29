import jax
import optax
import jax.numpy as jnp
from torch.utils import data

import losscape
import matplotlib.pyplot as plt



@jax.jit
def _loss(model_state, params, batch_stats, x, y):

    variables = {"params": params, "batch_stats": batch_stats}
    logits = model_state.apply_fn(variables, x, train=False)
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == y)
    one_hot = jax.nn.one_hot(y, num_classes=10)
    return jnp.mean(optax.softmax_cross_entropy(logits, one_hot))

def compute_loss(model_state, params, batch_stats, train_loader_unshuffled: data.DataLoader, num_batches):
    """
    Computes the loss on the dataset
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

def _generate_random_weights(weights, key):
  return jax.tree_util.tree_map(lambda x: jax.random.normal(key, x.shape), weights)

def _normalize_direction(direction, weights):

  w_norms = jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x.flatten(), ord=2), weights)
  d_norms = jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x.flatten(), ord=2), direction)

  norm_weights = jax.tree_util.tree_map(lambda d, w_norm, d_norm: d*w_norm/d_norm, direction, w_norms, d_norms)

  return norm_weights

def _perturb_params(params, direction, step):

  if len(direction) == 2:
    new_params = jax.tree_util.tree_map(lambda w, d0, d1: w + step[0]*d0 + step[1]*d1, params, direction[0], direction[1])
  else:
    new_params = jax.tree_util.tree_map(lambda w, d: w + step*d, params, direction)

  return new_params


