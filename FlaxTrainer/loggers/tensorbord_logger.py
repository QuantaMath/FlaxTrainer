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

import os
import re

from FlaxTrainer.loggers import LoggerBase
from flax.metrics.tensorboard import SummaryWriter


class TensorboardLogger(LoggerBase):
    
    def __init__(
        self,
        save_dir: str,
        version: str,
        name:str,
        auto_flush: bool = True
    ):
        """
        
        Args:
          log_dir: 
          auto_flush: 
        """  
        
        self.log_dir = save_dir
        self.version = 'version' if version is None else version
        self.name = name
        self.auto_flush = auto_flush



        
    def initialize(self):
        self.version_i = 0
        dir_list = os.listdir(self.log_dir)
        version_list = [
            int(d.split('_')[1]) for d in dir_list
            if d.split('_')[0]==self.version 
            and re.match(r'^\d*$', d.split('_')[1])
        ]
        
        if version_list:
            self.version_i = max(version_list) + 1

        v = ''.join([self.version,'_', str(self.version_i)])
        self.logger = SummaryWriter(
            log_dir=os.path.join(self.log_dir,v),
            auto_flush=self.auto_flush
        )

    def log_params(self, args, **kwargs):
        pass

    def log_hyperparams(self, args, **kwargs):
        pass

    def log_metrics(
        self,value:dict , step: int
    ):
        """
        Record metric data at epoch b
        
        Args:
          name: metric name
          value: metric value
          epoch: train epoch number

        """
        name = list(value)[0]
        self.logger.scalar(name, value[name], step)

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
    #FIXME: after finalize new folder created therfore it should be create the folder after 
    # new train process
    def finalize(self, msg):
        #self.logger.close()
        self.logger.close()
        