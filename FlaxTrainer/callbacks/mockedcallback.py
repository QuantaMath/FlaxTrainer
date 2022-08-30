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


from unittest.mock import Mock
from FlaxTrainer.callbacks import Callback

class MockedCallback(object):
    def __init__(self, stop_train=True):
        super(MockedCallback, self).__init__()
        self.stop_train = stop_train
        self.trainer = None

    def on_train_start(self, **kwargs):
        print("Start training....")
    
    def on_train_end(self, **kwargs):
        #print(self.trainer.state)
        print("Finsih training....")

    def on_train_step_start(self, **kwargs):
        print("Start training step....")

    def on_train_step_end(self, **kwargs):
        print("Finsih training step ....")
    
    def on_train_epoch_start(self, **kwargs):
        print("Start training epoch....")
        
    
    def on_train_epoch_end(self, **kwargs):
        if kwargs['epoch_idx'] == 10:
            self.trainer.stop_train()
        print(" Finsih training epoch....")

    def on_train_batch_start(self, **kwargs):
        print("Start training batch....")

    def on_train_batch_end(self, **kwargs):
        print("Finsih training batch....")
        pass


    def on_validation_start(self, **kwargs):
        print("Start validation....")
    
    def on_validation_stop(self, **kwargs):
        print("Finish validation....")
    
    def on_validation_step_start(self, **kwargs):
        print("Start validation step....")

    def on_validation_step_end(self, **kwargs):
        print("Start validation step end....")

    def on_validation_epoch_start(self, **kwargs):
        print("Start validation epoch....")
    
    def on_validation_epoch_end(self, **kwargs):
        print("Finish validation epoch....")

    def on_validation_batch_start(self, **kwargs):
        print("Start validation batch....")

    def on_validation_batch_stop(self, **kwargs):
        print("Finish validation batch....")


    def on_test_start(self, **kwargs):
        print("Start test....")
    
    def on_test_end(self, **kwargs):
        print("Finish test....")
    
    def on_test_step_start(self, **kwargs):
        print("Start test step....")

    def on_test_step_end(self, **kwargs):
        print("Finish test step....")
    # 
    def on_test_epoch_start(self, **kwargs):
        print("Start test epoch....")
    
    def on_test_epoch_end(self, **kwargs):
        print("Finish test epoch....")

    def on_test_batch_start(self, **kwargs):
        print("Start test batch....")

    def on_test_batch_stop(self, **kwargs):
        print("Finish test batch....")


    def on_predit_start(self, **kwargs):
        print("Start prediction....")

    def on_predit_end(self, **kwargs):
        print("Finish prediction....")
        pass

    
    def on_load_checkpoint(self, **kwargs):
        print("Checkpoint load....")

    def on_save_checkpoint(self, **kwargs):
        print("Checkpoint save....")

    def set_trainer(self, trainer):
        self.trainer = trainer