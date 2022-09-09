from typing import Sequence

import flax
import flax.linen as nn

import jax
import jax.numpy as jnp


class AutoEncoderBlock(nn.Module):
    module: nn.Module

    @nn.compact
    def __call__(self, x, **kwargs):

        output = self.encoder_module(x)
        return output


class AutoEncoder(nn.Module):
    encoder: nn.Module
    decoder: nn.Module
    bottleneck: nn.Module

    @nn.compact
    def __call__(self, x, **kwargs):

        data_size = x.shape[-1]
        latent = self.encoder(x)
        latent = self.bottleneck(latent)
        output = self.decoder(latent)
        output = nn.Dense(
            data_size,
            use_bias=False
        )(output)

        return output

    

class MLPBlock(nn.Module):
    hidden_size: Sequence[int]
    output_size: int
    hidden_activation: str = 'relu'
    
    
    @nn.compact
    def __call__(self, x, **kwargs):

        for layer_size in self.hidden_size:
            x = nn.Dense(layer_size)(x)
            x = getattr(nn, self.hidden_activation)(x)
        o = nn.Dense(self.output_size)(x)
        return o


class AutoEncoder(nn.Module):
    encoder: nn.Module
    decoder: nn.Module
    bottleneck: nn.Module

    @nn.compact
    def __call__(self, x, **kwargs):

        data_size = x.shape[-1]
        latent = self.encoder(x)
        latent = self.bottleneck(latent)
        output = self.decoder(latent)
        output = nn.Dense(
            data_size,
            use_bias=False
        )(output)
        
        return output