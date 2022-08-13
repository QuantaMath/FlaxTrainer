
# TODO: Documenting class
class Callback(object):
    def __init__(self):
        NotImplementedError
    

    def on_train_start(self, **kwargs):
        pass
    
    def on_train_stop(self, **kwargs):
        pass

    def on_train_step_start(self, **kwargs):
        pass

    def on_train_step_stop(self, **kwargs):
        pass
    
    def on_train_epoch_start(self, **kwargs):
        pass
    
    def on_train_epoch_end(self, **kwargs):
        pass

    def on_train_batch_start(self, **kwargs):
        pass

    def on_train_batch_stop(self, **kwargs):
        pass


    def on_validation_start(self, **kwargs):
        pass
    
    def on_validation_stop(self, **kwargs):
        pass
    
    def on_validation_step_start(self, **kwargs):
        pass

    def on_validation_step_stop(self, **kwargs):
        pass
    
    def on_validation_epoch_start(self, **kwargs):
        pass
    
    def on_validation_epoch_end(self, **kwargs):
        pass

    def on_validation_batch_start(self, **kwargs):
        pass

    def on_validation_batch_stop(self, **kwargs):
        pass


    def on_test_start(self, **kwargs):
        pass
    
    def on_test_stop(self, **kwargs):
        pass
    
    def on_test_step_start(self, **kwargs):
        pass

    def on_test_step_stop(self, **kwargs):
        pass
    # 
    def on_test_epoch_start(self, **kwargs):
        pass
    
    def on_test_epoch_end(self, **kwargs):
        pass

    def on_test_batch_start(self, **kwargs):
        pass

    def on_test_batch_stop(self, **kwargs):
        pass


    def on_predit_start(self, **kwargs):
        pass

    def on_predit_end(self, **kwargs):
        pass

    
    def on_load_checkpoint(self, **kwargs):
        pass

    def on_save_checkpoint(self, **kwargs):
        pass

