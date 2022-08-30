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

# TODO: Documenting class
class Callback(object):
    def __init__(self):
        NotImplementedError
    

    def on_train_start(self, **kwargs):
        pass
    
    def on_train_end(self, **kwargs):
        pass

    def on_train_step_start(self, **kwargs):
        pass

    def on_train_step_end(self, **kwargs):
        pass
    
    def on_train_epoch_start(self, **kwargs):
        pass
    
    def on_train_epoch_end(self, **kwargs):
        pass

    def on_train_batch_start(self, **kwargs):
        pass

    def on_train_batch_end(self, **kwargs):
        pass


    def on_validation_start(self, **kwargs):
        pass
    
    def on_validation_end(self, **kwargs):
        pass
    
    def on_validation_step_start(self, **kwargs):
        pass

    def on_validation_step_end(self, **kwargs):
        pass
    
    def on_validation_epoch_start(self, **kwargs):
        pass
    
    def on_validation_epoch_end(self, **kwargs):
        pass

    def on_validation_batch_start(self, **kwargs):
        pass

    def on_validation_batch_end(self, **kwargs):
        pass


    def on_test_start(self, **kwargs):
        pass
    
    def on_test_end(self, **kwargs):
        pass
    
    def on_test_step_start(self, **kwargs):
        pass

    def on_test_step_end(self, **kwargs):
        pass
    # 
    def on_test_epoch_start(self, **kwargs):
        pass
    
    def on_test_epoch_end(self, **kwargs):
        pass

    def on_test_batch_start(self, **kwargs):
        pass

    def on_test_batch_end(self, **kwargs):
        pass


    def on_predit_start(self, **kwargs):
        pass

    def on_predit_end(self, **kwargs):
        pass

    
    def on_load_checkpoint(self, **kwargs):
        pass

    def on_save_checkpoint(self, **kwargs):
        pass

