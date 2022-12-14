# MIT License

# Copyright (c) 2022 Gholamhossin Eslami

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# This code inspired from the notebook:
# https://github.com/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/guide4/Research_Projects_with_JAX.ipynb
# Authored by Phillip Lippe

import os
import time
import json
from copy import copy

from tqdm import tqdm

from typing import Any, Callable, Dict, Iterator, Tuple, List
from collections import defaultdict

# JAX ecosytems import 
import jax
from jax._src.api import eval_shape
import jax.numpy as jnp
from jax import random

from FlaxTrainer.callbacks import Callback

## Flax (NN in JAX)
try:
    import flax
    import flax.linen as nn
    from flax.training import train_state, checkpoints
except ModuleNotFoundError:
    print("flax not found, run 'pip install --upgrade git+https://github.com/google/flax.git'")
## Optax (Optimizers in JAX)

try:
    import optax
except ModuleNotFoundError:
    print("optax not found, run 'pip install --upgrade git+git://github.com/deepmind/optax.git'")

# temporary
import torch
import torch.utils.data as data


from pytorch_lightning.loggers import WandbLogger

from .trainstates import TrainState
from FlaxTrainer.loggers import TensorboardLogger

class TrainerBaseModule(object):
    """ Base class of trainer module fo training flax based artificial neural network"""
    def __init__(self):
        super().__init__()
        self.debug=False
        self.train = True

    def init_model(self):
        raise NotImplementedError

    def bind_model(self):
        raise NotImplementedError
    
    def create_functions(self): #-> Callable[..., Any]:
        raise NotImplementedError

    def create_jitted_functions(self):
        
        if self.debug: # Skip jitted
            print("Skipped jitted due to debug=True")
            {
                setattr(self, func.__name__, func) for func in self.create_functions()
            }
        else:
            {
                setattr(self, func.__name__, jax.jit(func)) for func in self.create_functions()
            }


    def init_logger(self):
        raise NotImplementedError

    def stop_train(self):
        self.train = False


class TrainerModule(TrainerBaseModule):

    def __init__(
        self,
        #model_class: nn.Module,
        #model_hparams: Dict[str, Any],
        optimizer_hparams: Dict[str, Any],
        #exmp_input: Any,
        callbacks: set[Callback] = set(),
        seed: int = 42,
        logger_params: Dict[str, Any] | None = None,
        enable_progress_bar: bool = True,
        debug: bool = False,
        check_val_every_n_epoch: int = 1,
        **kwargs
    ):
        """
        A basic Trainer module summarizing most common training functionalities
        like logging, model initialization, training loop, etc,

        Inputs:
            model_class: The class of the model that should be trained.
            model_hparams: A dictionary of all hyperparameters of the model. Is
              Used as input to the model when created.
            optimizer_hparams - A dictionary of all hyperparameters of the optimizer.
              Use during initialization of the optimizer.
            exmp_input: Input to the model for initialization and tabulate.
            seed - Seed to initialize PRNG
            logger_params: A dictionary containing the specification of the logger.
            enable_progress_bar: If False, no progress bar is shown.
            debug: If True, no jitting is applied. Can be helpful for debugging.
            check_val_every_n_epoch: The frequency with which the is evaluated
            on the validation set.
        """
        super(TrainerModule, self).__init__()
        #self.model_class = model_class
        #self.model_hparams = model_hparams
        self.optimizer_hparams = optimizer_hparams
        self.enable_progress_bar = enable_progress_bar
        self.debug = debug
        self.seed = seed
        self.check_val_every_n_epoch = check_val_every_n_epoch
        #self.exmp_input = exmp_input
        
        self.callbacks = callbacks
        [callback.set_trainer(self) for callback in callbacks]
        [print(callback.trainer) for callback in callbacks]
        # Set of  hyperparameters to save
        self.config = {
            #'model_class': model_class.__name__,
            #'model_hparams': model_hparams,
            'optimizer_hparams': optimizer_hparams,
            'logger_params': logger_params,
            'enable_progress_bar': self.enable_progress_bar,
            'debug': self.debug,
            'check_val_every_n_epoch': check_val_every_n_epoch,
            'seed': self.seed
        }
        self.config.update(kwargs)
        # Create empty model: no parameters yet
        #self.model = self.model_class(**self.model_hparams)
        # Init trainer parts
        #self.init_logger(logger_params)
        self.create_jitted_functions()
        # self.init_model(exmp_input)

    def init_logger(
        self,
        state: TrainState,
        logger_params: Dict | None = None,
        ):
        """3
        Initializes of logger and creates a logging directory.

        Args:
          logger_params: A dictionary containing the specification of the logger.
        """
        if logger_params is None:
            logger_params = dict()
        # Determine logging directory
        log_dir = logger_params.get('log_dir', None)
        if not log_dir:
            base_log_dir = logger_params.get('base_log_dir', 'checkpoints/')
            # Prepare logging
            log_dir = os.path.join(base_log_dir, state.model_class)
            if 'logger_name' in logger_params:
                log_dir = os.path.join(log_dir, logger_params['logger_name'])
            version = None
        else:
            version = ''
        # Create logger object
        logger_type = logger_params.get('logger_type', 'TensorBoard').lower()
        if logger_type == 'tensorboard':
            self.logger = TensorboardLogger(save_dir=log_dir,
                                            version=version,
                                            name='')
        elif logger_type == 'wandb':
            self.logger = WandbLogger(name=logger_params.get('project_name', None),
                                      save_dir=log_dir,
                                      version=version,
                                      config=self.config)
        else:
            assert False, f'Unknown logger type \"{logger_type}\"'
        # Save hyperparameters
        log_dir = self.logger.log_dir
        if not os.path.isfile(os.path.join(log_dir, 'hparams.json')):
            os.makedirs(os.path.join(log_dir, 'metrics/'), exist_ok=True)
            with open(os.path.join(log_dir, 'hparams.json'), 'w') as f:
                json.dump(self.config, f, indent=4)
        self.log_dir = log_dir

    def init_model(
        self,
        model: nn.Module,
        exmp_input: Any,
        tabulated: bool= True
    ):
        """
        Create an initial training state with newly generated network parameters.

        Args:
          model: flax NN module class which 
          exmp_input: An input to the model with which the shape are inferred.
          tabulated:
        """
        # Prepare PRNG and input

        model_rng = random.PRNGKey(self.seed)
        model_rng, init_rng = random.split(model_rng)
        exmp_input = [exmp_input] if not isinstance(exmp_input, (list, tuple)) else exmp_input
        # Run model initialization
        variables = self.run_model_init(
            model, exmp_input, init_rng
        )
        # Create default state. Optimizers is initialized later
        new_state = TrainState(step = 0,
                                apply_fn=model.apply,
                                params=variables['params'],
                                batch_stats=variables.get('batch_stats'),
                                rng=model_rng,
                                model_class=model.__class__.__name__,
                                tx=None,
                                opt_state=None)
        
        self.init_logger( new_state, self.config['logger_params'])
        if tabulated: 
            self.print_tabulate(model, exmp_input)

        return new_state
        

    def run_model_init(
        self,
        model: nn.Module,
        exmp_input: Any,
        init_rng: Any
    ) -> Dict:
        """
        The model initialization call

        Args:
          exmp_input: An input to the model with which the shape are inferred.
          init_rng: A jax.random.PRNGKey.

        Returns:
          The initialized variable dictionary
        """

        return model.init(init_rng, *exmp_input, train=True)

    def print_tabulate(
        self,
        model: nn.Module,
        exmp_input: Any
    ):
        """
        Print a summary of the Module represent as table

        Args: exmpt_input: An input to the model with which the shape are inferred.
        """
        print(model.tabulate(random.PRNGKey(0), *exmp_input, train=True))

    def init_optimizer(
        self,
        state: TrainState,
        num_epochs: int,
        num_steps_per_epoch: int
    ):
        """
        Initializes the optimizer and learning_rate_shedular.

        Args:
          num_epochs: Number of epochs the model will be trained for.
          num_steps_per_epoch: Number of training step per epoch.
        """

        hparams = copy(self.optimizer_hparams)

        # Initialize optimizer
        optimizer_name = hparams.pop('optimizer', 'adamw')
        if optimizer_name.lower() == 'adam':
            opt_class = optax.adam
        elif optimizer_name.lower() == 'adamw':
            opt_class = optax.adamw
        elif optimizer_name.lower() == 'sgd':
            opt_class = optax.sgd
        else:
            assert False, f'Unknown optimizer "{opt_class}"'
        # Initialize learning rate schedular
        # A cosine decay schedular is use, but other are also possible

        lr = hparams.pop('lr', 1e-3)
        warmup = hparams.pop('warmup', 0)
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=warmup,
            decay_steps=int(num_epochs * num_steps_per_epoch),
            end_value=0.01 * lr
        )
        # Clip gradients at max value, and evt. apply weight decay
        transf = [optax.clip_by_global_norm(hparams.pop('gradient_clip', 1.0))]
        if opt_class == optax.sgd and 'weight_decay' in hparams:  # wd is integrated in adamw
            transf.append(optax.add_decayed_weights(hparams.pop('weight_decay', 0.0)))
        optimizer = optax.chain(
            *transf,
            opt_class(lr_schedule, **hparams)
        )
        # initialize training stata
        new_state = TrainState.create(apply_fn=state.apply_fn,
                                       params=state.params,
                                       batch_stats=state.batch_stats,
                                       tx=optimizer,
                                       rng=state.rng)
        
        return new_state


    # def create_jitted_functions(self):
    #     """
    #     Create jitted version of the training and evaluation functions.
    #     If self.debug is True, not jitted is applied.
    #     """

    #     train_step, eval_step = self.create_functions()
    #     if self.debug: # Skip jitted
    #         print("Skipped jitted due to debug=True")
    #         self.train_step = train_step
    #         self.eval_step = eval_step
    #     else:
    #         self.train_step = jax.jit(train_step)
    #         self.eval_step = jax.jit(eval_step)


    def create_functions(self) -> Tuple[Callable[[TrainState, Any], Tuple[TrainState, Dict]],
                                        Callable[[TrainState, Any], Tuple[TrainState, Dict]]]:
        """
        Creates and returns functions for the training and evaluation step. The
        functions take as input the training state and a batch from the train/
        val/test loader. Both functions are expected to return a dictionary of
        logging metrics, and the training function a new train state. This
        function needs to be overwritten by a subclass. The train_step and
        eval_step functions here are examples for the signature of the functions.
        """

        def train_step(state: TrainState,
                       bach: Any):
            metrics = {}
            return state, metrics
        def eval_step(state: TrainState,
                      batch: Any):
            metrics = {}
            return metrics
        raise NotImplementedError

    def train_model(
        self,
        model: nn.Module,
        state: TrainState,
        train_loader : Iterator,
        val_loader : Iterator,
        test_loader : Iterator | None = None,
        num_epochs : int = 500) -> Dict[str, Any]:
        """
        Starts a training loop for the given number of epochs.

        Args:
          train_loader: Data loader of the training set.
          val_loader: Data loader of the validation set.
          test_loader: If given, best model will be evaluated on the test set.
          num_epochs: Number of epochs for which to train the model.

        Returns:
          A dictionary of the train, validation and evt. test metrics for the
          best model on the validation set.
        """
        # Create optimizer and the scheduler for the given number of epochs
        self.logger.initialize()

        new_state = self.init_optimizer(state, num_epochs, len(train_loader))
        # Prepare training loop
        self.on_training_start()

        
        best_eval_metrics = None
        for epoch_idx in self.tracker(range(1, num_epochs+1), desc='Epochs'):

            if not self.train:
                break
            train_metrics, new_state = self.train_epoch(new_state, train_loader, epoch_idx=epoch_idx)
            self.logger.log_metrics(value=train_metrics, step=epoch_idx)
            self.on_training_epoch_end(epoch_idx)
            # Validation every N epochs
            # FIXME: fix validation steps and total epochs steps problem 
            if epoch_idx % self.check_val_every_n_epoch == 0:
                eval_metrics = self.eval_model(new_state, val_loader, log_prefix='val/')
                self.on_validation_epoch_end(epoch_idx, eval_metrics, val_loader)
                self.logger.log_metrics(eval_metrics, step=epoch_idx)
                self.save_metrics(f'eval_epoch_{str(epoch_idx).zfill(3)}', eval_metrics)
                # Save best model
                if self.is_new_model_better(eval_metrics, best_eval_metrics):
                    best_eval_metrics = eval_metrics
                    best_eval_metrics.update(train_metrics)
                    self.save_model(new_state, step=epoch_idx)
                    self.save_metrics('best_eval', eval_metrics)
        # Test best model if possible
        if test_loader is not None:
            self.load_model(model, new_state)
            test_metrics = self.eval_model(new_state, test_loader, log_prefix='test/')
            self.logger.log_metrics(test_metrics, step=epoch_idx)
            self.save_metrics('test', test_metrics)
            best_eval_metrics.update(test_metrics)
        # Close logger
        self.logger.finalize('success')
        [callback.on_train_end() for callback in self.callbacks]
        return best_eval_metrics, new_state


    def train_epoch(
        self,
        state: TrainState,
        train_loader: Iterator,
        epoch_idx: int) -> Dict[str, Any]:
        """
        Trains a model for one epoch.

        Args:
          train_loader: Data loader of the training set.

        Returns:
          A dictionary of the average training metrics over all batches
          for logging.
        """

        # Train model for one epoch, and log avg loss and accuracy
        
        [callback.on_train_epoch_start() for callback in self.callbacks]

        metrics = defaultdict(float)
        num_train_steps = len(train_loader)
        start_time = time.time()
        for batch in self.tracker(train_loader, desc='Training', leave=False):
            state, step_metrics = self.train_step(state, batch)
            for key in step_metrics:
                metrics['train/' + key] += step_metrics[key] / num_train_steps
        metrics = {key: metrics[key].item() for key in metrics}
        metrics['epochs_time'] = time.time() - start_time
        
        [callback.on_train_epoch_end(trainer=self, epoch_idx=epoch_idx) for callback in self.callbacks]
        return metrics, state


    def eval_model(
        self,
        state: TrainState,
        data_loader: Iterator,
        log_prefix: str | None = '') -> Dict[str, Any]:
        """
        Evaluate the model of a dataset.

        Args:
          data_loader: Data loader of the dataset to evaluate on.
          log_prefix: Prefix to add to all metrics (e.g 'val/' or 'test/')

        Returns:
          A dictionary of the evaluation metrics, averaged over data points
          in the dataset.
        """
        # Test model on all images of a data loader and return svg loss
        metrics = defaultdict(float)
        num_elements = 0
        for batch in data_loader:
            step_metrics = self.eval_step(state, batch)
            batch_size = batch[0].shape[0] if isinstance(batch, (list, tuple)) else batch.shape[0]
            for key in step_metrics:
                metrics[key] += step_metrics[key] * batch_size
            num_elements +=batch_size
        metrics = {(log_prefix + key): (metrics[key] / num_elements).item() for key in metrics}
        return metrics

    def is_new_model_better(self,
                     new_metrics: Dict[str, Any],
                     old_metrics: Dict[str, Any]) -> bool:
        """
        Compares two sets of evaluation metrics to decide whether the
        new model is better than the previous one or not.

        Args:
          new_metrics: A dictionary of the evaluation metrics of the new model
          old_metrics: A dictionary of the evaluation metrics of the previously
            best model, i.e. the one to compare to.
        """
        if old_metrics is None:
            return True

        for key, is_larger in [('val/val_metric', False), ('val/acc', True), ('val/loss', False)]:
            if key in new_metrics:
                if is_larger:
                    return new_metrics[key] > old_metrics[key]
                else:
                    return new_metrics[key] < old_metrics[key]

        assert False, f"No known metrics to log on: {new_metrics}"

    def tracker(self,
                Iterator: Iterator,
                **kwargs) -> Iterator:
        """
        Wraps an iterator in a progress bar tracker (tqdm) if the progress bar
        is enabled.

        Args:
          iterator: Iterator to wrap in tqdm
          kwargs: Additional arguments to tqdm

        Returns:
          Wrapped iterator if progress bar is enabled, otherwise same iterator
          as input.
        """
        if self.enable_progress_bar:
            return tqdm(Iterator, **kwargs)
        else:
            return Iterator

    def save_metrics(self,
                     filename: str,
                     metrics: Dict[str, Any]):
        """
        Saves a dictionary of metrics to file. Can be used as a textual
        representation of the validation performance for checking in the terminal.

        Args:
          filename: Name of the metrics file without folders and postfix
          metrics: A dictionary of metrics to save in the file.
        """
        with open(os.path.join(self.log_dir, f'metrics/{filename}.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        pass

    def on_training_start(self):
        """
        Method called before training is started. Can be used for additional
        initialization operations etc.
        """
        pass

    def on_training_epoch_end(self,
                              epch_idx: int):
        """
        Method called at the end of each training epoch. Can be used for additional
        logging or similar

        Args:
          epoch_idx: Index of the training epoch that has finished.
        """
        pass

    def on_validation_epoch_end(self,
                                epoch_idx: int,
                                eval_metrics: Dict[str, Any],
                                val_loader: Iterator):
        """
        Method called at the end of each validation epoch. Can be used for additional
        logging and evaluation.

        Args:
          epoch_idx: Index of the training epoch at which validation was performed.
          eval_metrics: A dictionary of the validation metrics. New metrics added to
            this dictionary will be logged as well
          val_loader: Data loader of the validation set, to support additional
            evaluation
        """
        pass

    def save_model(
        self,
        state: TrainState,
        step: int = 0):
        """
        Save current training state at certain training iteration. Only the model
        parameters and batch statistics are saved to reduce memory footprint. To
        support the training to be continued from acheckpoint, this method can be
        extended to include the optimizer state as well

        Args:
          step: Index of the step to save the model at, e.g. epoch.
        """
        #[callback.on_save_checkpoint() for callback in callbacks]
        checkpoints.save_checkpoint(ckpt_dir=self.log_dir,
                                    target={'params': state.params,
                                            'batch_stats': state.batch_stats},
                                    step=step,
                                    overwrite=True
                                    )


    def load_model(self, model: nn.Module, state: TrainState):
        """
        Load model parameters and batch statistics from the logging directory.
        """
        # [callback.on_load_checkpoint() for callback in callbacks]
        state_dict = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=None)
        new_state = TrainState.create(apply_fn=model.apply,
                                       params=state_dict['params'],
                                       batch_stats=state_dict['batch_stats'],
                                       #Optimizer will be overwritten when training, starts
                                       tx=state.tx if state.tx else optax.sgd(0.1),
                                       rng=state.rng
                                       )
        return new_state

    def bind_model(self, model: nn.Module, state: TrainState):
        """
        Return a model with parameters bound to it. Enable an easier inference
        access

        Returns:
          The model with parameters and evt. batch statistics bound to it.
        """
        params = {'params': state.params}
        if state.batch_stats:
            params['batch_stats'] = self.batch_stats
        return model.bind(params)

    @classmethod
    def load_from_checkpoint(cls,
                             checkpoint: str,
                             exmp_input: Any) -> Any:
        """
        Creates a Trainer object with same hyperparameters and loaded model from
        a checkpoint directory

        Args:
          checkpoint: Folder in which the checkpoint and hyperparameter file is stored.
          exmp_input: An input to the model for shape inference.

        Returns:
          A Trainer object with model loaded from the checkpoint folder.
        """
        hparams_file = os.path.join(checkpoint, 'hparams.json')
        assert os.path.isfile(hparams_file)
        with open(hparams_file, 'r') as f:
            hparams = json.load(f)
        hparams.pop("model_class")
        hparams.update(hparams.pop("model_hparams"))
        if not hparams['logger_params']:
            hparams["logger_params"] = dict()
        hparams["logger_params"]["log_dir"] = checkpoint
        trainer = cls(exmp_input=exmp_input,
                      **hparams)
        return trainer

