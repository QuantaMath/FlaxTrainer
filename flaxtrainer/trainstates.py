from typing import Any
from flax.training import train_state


# TODO: Generalize the TrainStateClass
class TrainState(train_state.TrainState):
    # A simple extension of TrainState to also include batch statistics
    # If a model has no batch statistics, it is None
    batch_stats : Any = None
    # You can further extend the TrainState by any additional part here
    # For example, rng to keep for init, dropout, etc.
    rng : Any = None