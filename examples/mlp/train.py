from FlaxTrainer.trainer import TrainerModule
from typing import Sequence, Callable

import jax
import flax
import flax.linen as nn

from model import MLP

class MLPTrainer(TrainerModule):
    def __init__(
        self,
        hidden_features: Sequence[int],          
        hidden_activation: Sequence[Callable],   
        output_dim: int,                         
        output_activaton: Callable,
        **kwargs
    ):
        super().__ini__(
            model_class=MLP,
            model_hparams={
                'hidden_features': hidden_features,
                'hidden_activation': hidden_activation,   
                'output_dim': output_dim,
                'output_activaton': output_activaton,
            },
            **kwargs
        )

    def create_function(self):
        def mse_loss(params, batch):
            x, y = batch,
            pred = self.model.apply({'params': params}, x)
            loss = ((pred - y) ** 2).mean()
            return loss
        
        def train_step(state, batch):
            loss_fn = lambda params: mse_loss(params, batch)
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            metrics = {'loss': loss}
            return state, metrics
        
        def eval_step(state, batch):
            loss = mse_loss(state.params, batch)
            return {'loss': loss}
        
        return train_step, eval_step
    
