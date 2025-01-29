import jax.numpy as jnp
from torch.utils import data

from .utils import *

def _generate_2D_plot(model_state,
                      batch_stats,
                      train_loader_unshuffled: data.DataLoader,
                      num_batches, x1=-0.03, x2=0.03):

  direction = _generate_random_weights(model_state.params, key)

  init_weights = model_state.params

  coords = jnp.linspace(x1, x2, 500)
  losses = []

  for x in coords:
    new_params = _perturb_params(init_weights, direction, x)
    loss = compute_loss(model_state, new_params, batch_stats, train_loader_unshuffled, num_batches)
    losses.append(loss)

  return coords, losses



def _generate_3D_plot(model_state, 
                      batch_stats, 
                      train_loader_unshuffled: data.DataLoader, 
                      num_batches, x1=-0.03, x2=0.03):

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
      loss = compute_loss(model_state, new_params, batch_stats, train_loader_unshuffled, num_batches)
      losses = losses.at[i,j].set(loss)
      count += 1


  return X, Y, losses