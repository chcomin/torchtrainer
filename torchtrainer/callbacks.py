
"""Initial sketch of a callback system que may be added to the training loop.
Not used anywhere yet.
The callbacks here were copied from the fast.ai codebase.
"""

class BaseCallback:

    def before_epoch(self): 
        """called at the beginning of each epoch, useful for any behavior you need to reset at each epoch."""
        pass

    def before_train(self): 
        """called at the beginning of the training part of an epoch."""
        pass

    def before_batch(self): 
        """called at the beginning of each batch, just after drawing said batch. It can be used to do any setup necessary 
        for the batch (like hyper-parameter scheduling) or to change the input/target before it goes in the model (change 
        of the input with techniques like mixup for instance)."""
        pass
    
    def after_pred(self): 
        """called after computing the output of the model on the batch. It can be used to change that output before it's 
        fed to the loss."""
        pass

    def after_loss(self): 
        """called after the loss has been computed, but before the backward pass. It can be used to add any penalty to the 
        loss (AR or TAR in RNN training for instance)."""
        pass

    def before_backward(self): 
        """called after the loss has been computed, but only in training mode (i.e. when the backward pass will be used)"""
        pass

    def before_step(self): 
        """called after the backward pass, but before the update of the parameters. It can be used to do any change to the 
        gradients before said update (gradient clipping for instance)."""
        pass

    def after_step(self): 
        """called after the step and before the gradients are zeroed."""
        pass

    def after_batch(self): 
        """called at the end of a batch, for any clean-up before the next one."""
        pass

    def after_train(self): 
        """called at the end of the training phase of an epoch."""
        pass

    def before_validate(self): 
        """called at the beginning of the validation phase of an epoch, useful for any setup needed specifically for validation."""
        pass
    
    def after_validate(self): 
        """called at the end of the validation part of an epoch."""
        pass

    def after_epoch(self): 
        """called at the end of an epoch, for any clean-up before the next one."""
        pass

    #def on_epoch_end(self):