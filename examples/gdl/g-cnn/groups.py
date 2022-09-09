from this import d
import jax
import jax.numpy as jnp

import flax
import flax.linen as nn


class GroupBase(nn.Module):
    
    def identity(self):
        raise NotImplementedError
    
    def elements(self):
        raise NotImplementedError
    
    def product(self, h, h_prime):
        """ Obtain group product on two group elements in group."""
    
    def inverse(self, h):
        """"""

    def determinant(self, h):
        """ Calculate the determinant of the representation of a group h"""