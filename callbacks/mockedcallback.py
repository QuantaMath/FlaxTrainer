from unittest.mock import Mock
from callbacks import Callback
#from ..FlaxTrainer.trainer import TrainerBaseModule

class MockedCallback(object):
    def __init__(self, stop_train=True):
        super(MockedCallback, self).__init__()
        self.stop_train = stop_train
        self.trainer = None

    def on_train_start(self, **kwargs):
        print("Start training....")
    
    def on_train_end(self, **kwargs):
        print(self.trainer.state)
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