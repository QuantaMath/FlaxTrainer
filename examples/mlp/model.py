
from typing import Sequence, Callable
import flax
import flax.linen as nn

import jax
import jax

class MLP(nn.Module):
    hidden_features: Sequence[int]          # Sequence that contain dimensions of hidden layers
    hidden_activation: Sequence[Callable]   # Sequence that contain activatoin of hidden layers
    output_dim: int                         # output dimension
    output_activaton: Callable              # output activation

    @nn.compact
    def __call__(self, x):
        
        for feat_dim, activation in zip(self.hidden_features, self.hidden_activation):
            x = nn.Dense(feat_dim)(x)
            x = nn.activation(x)
        
        output = nn.Dense(self.output_dim)
        return self.output_activaton(x)
