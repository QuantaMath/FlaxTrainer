import jax
import flax.linen as nn
from sklearn.datasets import make_regression


dataset = make_regression(
    n_samples=10000,
    n_features=200,
    n_targets= 10,)

from .train import MLPTrainer
from .model import MLP



trainer = MLPTrainer(
    hidden_dims=[128, 128, 64],
    hidden_activations=[nn.relu, nn.tanh, nn.swish],
    output_dim=10,
    output_activaton=lambda x: x,
    optimizer_params={'lr', 4e-3},
    logger_params={'base_log_dir': "./checkpoints"},
    check_val_envery_n_epoch=5
)

