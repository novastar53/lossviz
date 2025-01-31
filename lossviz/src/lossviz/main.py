import matplotlib.pyplot as plt

from models.resnet import ResidualCNN
from dataloaders.cifar10 import get_dataloaders
from plot import generate_2D_plot, generate_3D_plot


# Create the train state
def create_train_state(rng, model, train_size):
    
    variables = model.init(rng, jnp.ones([1, 32, 32, 3]), True)
    params = variables['params']

    num_params = count_parameters(params)
    print(f"Initialized model with {num_params} parameters")

    batch_stats = None
    if "batch_stats" in variables:
        batch_stats = variables['batch_stats']

    steps_per_epoch = train_size//BATCH_SIZE
    scales = [81*steps_per_epoch, 122*steps_per_epoch]

    lr_schedule = optax.schedules.piecewise_constant_schedule(init_value=LEARNING_RATE,
                                                              boundaries_and_scales={scales[0]: 0.1, scales[1]: 0.1})

    tx = optax.sgd(learning_rate=lr_schedule,
                   momentum=0.9, nesterov=True)


    return train_state.TrainState.create(apply_fn=cnn.apply,
                                         params=params, tx=tx), batch_stats


model = ResidualCNN()
resnet_state, _ = create_train_state(rng, model, 50000)
resnet_state = checkpoints.restore_checkpoint(ckpt_dir=f"models/ckpts/resnet/resnet-56", target=resnet_state, step=180)
f = open(f"models/ckpts/resnet/resnet-56.pkl", "rb")
resnet_batch_stats = pickle.load(f)
train_dataloader_unshuffled, _ = get_dataloaders()

X, Y = generate_2D_plot(model_state, batch_stats, train_loader_unshuffled, num_batches, x1=-0.03, x2=0.03)


plt.plot(X, Y)
plt.show()