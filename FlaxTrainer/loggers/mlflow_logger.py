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

from asyncio.log import logger
from loggers import LoggerBase

import mlflow


class MLFlowLogger(LoggerBase):
    
    def __init__(
        self,
        log_dir: str,
        auto_flush: bool = True
    ):
        """
        
        Args:
          log_dir: 
          auto_flush: 
        """  

        self.log_dir = log_dir

        with mlflow.start_run():
            pass
        
    def log_params(self, args, **kwargs):
        pass

    def log_hyperparams(self, args, **kwargs):
        pass

    def log_metrics(
        self, name: str, value:float , epoch: int
    ):
        """
        
        Args:
          name: metric name
          value: metric value
          epoch: train epoch number
        """
        with self.logger:
            mlflow.log_metric(name, value, epoch)

    def log_artifacts(self, args, **kwargs):
        pass

    def hparams(self, hparams):
        """
        
        Args:
          hparams: Flat mapping from hyper parameter name to value.
        """
        self.logger.hparams(
            hparams=hparams
        )