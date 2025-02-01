import pickle

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax.training import checkpoints

import matplotlib.pyplot as plt

from models import resnet
from dataloaders.cifar10 import get_dataloaders
from plot import generate_2D_plot, generate_3D_plot


BATCH_SIZE = 128
LEARNING_RATE = 0.1
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 180


# Create the train state
def create_train_state(rng, model, train_size):
    
    variables = model.init(rng, jnp.ones([1, 32, 32, 3]), True)
    params = variables['params']

    batch_stats = None
    if "batch_stats" in variables:
        batch_stats = variables['batch_stats']

    steps_per_epoch = train_size//BATCH_SIZE
    scales = [81*steps_per_epoch, 122*steps_per_epoch]

    lr_schedule = optax.schedules.piecewise_constant_schedule(init_value=LEARNING_RATE,
                                                              boundaries_and_scales={scales[0]: 0.1, scales[1]: 0.1})

    tx = optax.sgd(learning_rate=lr_schedule,
                   momentum=0.9, nesterov=True)


    return train_state.TrainState.create(apply_fn=model.apply,
                                         params=params, tx=tx), batch_stats


rng = jax.random.PRNGKey(0)
model = resnet.ResidualCNN(N=9)
resnet_state, _ = create_train_state(rng, model, 50000)
resnet_state = checkpoints.restore_checkpoint(ckpt_dir=f"/lossviz/models/ckpts/resnet/resnet-56", target=resnet_state, step=NUM_EPOCHS)

f = open(f"/lossviz/models/ckpts/resnet/resnet-56.pkl", "rb")
resnet_batch_stats = pickle.load(f)

train_dataloader_unshuffled, _ = get_dataloaders()

X, Y = generate_2D_plot(rng, resnet_state, resnet_batch_stats, train_dataloader_unshuffled, num_batches=4, x1=-0.03, x2=0.03)


plt.plot(X, Y)
plt.show()