import flax
import flax.linen as nn
from flax.linen.initializers import zeros
from flax.linen.recurrent import RNNCellBase
import jax
import jax.numpy as jnp

# model inspired from pytorch tutorial:
# https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
class SimpleRNN(RNNCellBase):
    
    hidden_size: int 
    output_size: int

    @nn.compact
    def __call__(self, x, h):
        
        combined = jnp.concatenate([x, h], -1)
        hidden = nn.Dense(self.hidden_size, name='hidden_state')(combined)
        hidden = nn.relu(hidden)
        output = nn.Dense(self.output_size, name='output')(combined)
        return output, hidden
    
    @staticmethod
    def initialize_carry(rng, batch_dims, size, init_fn=zeros):
        mem_shape = batch_dims + (size,)
        return init_fn(rng, mem_shape)