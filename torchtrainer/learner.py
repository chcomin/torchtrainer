'''
Class used for training a Pytorch neural network.
'''

import torch
from torch.optim.lr_scheduler import LambdaLR

class Learner:
    """Class used for training a Pytorch neural network.

    The class is initialized with a CNN model, a loss function, an optimizer and train
    and validation datasets. The main methods are:

    fit(epochs) : train the network for the given number of epochs
    pred(xb) : apply model to a batch
    save_state() and load_state() : save and load learner state
    get_history() : return the train and validation losses, accuracies and learning rates
                    for each epoch

    Parameters
    ----------
    model : torch.nn
        The neural network to be trained
    loss_func : callable
        The function used for calculating the loss. Should have signature loss_func(input,
        target, weight=None, epoch=None). `input` has shape (batch size, num classes, height, width)
        and target has shape (batch size, height, width). `weight` can be used for weighting the loss
        for each pixel (same shape as `target`) and `epoch` is the current training epoch.
    optm : torch.optim
        Optimizer used for updating the parameters. 
    train_dl : torch.Dataset
        Dataloader used for training
    valid_dl : torch.Dataset
        Dataloader used for validation
    scheduler : torch.optim.lr_scheduler
        Scheduler used for updating the learning rate of the optimizer
    acc_funcs: dict
        Dict of functions to be used for measuring result accuracy. Each key is a string containing
        the name of the accuracy measure and respective values are functions with signature f(input, target)
        containing the prediction of the model and the ground truth. Shapes of input and target are the same
        as in `loss_func`
    main_acc_func : string
        Accuracy score used for checking if the model has improved. At the end of each epoch, if this
        accuracy is larger than any other previously recorded, the parameters of the model are saved
    checkpoint_file : string
        File to save the model when the accuracy have improved. Also used as default file for saving
        the model when function save_state is called
    scheduler_step_epoch : bool
        If True, the scheduler will call step() after every epoch. If False, step() will be called after
        every batch.
    callbacks : list of callable
        List of callback functions to call on the validation data after each epoch. Functions must have signature 
        callback(val_batch, val_label_batch, model_output, epoch), where val_batch and val_label_batch are
        and image and label batch, model_output is the output of the model for the batch and epoch is the current
        epoch.
    device : torch.device
        Device used for training
    verbose : bool
        If True, prints information regarding the model performance after each epoch to the standard output .

    TODO: Model should be moved to `device` before constructing the optimizer. Only solution is to receive and optm
    class instead of an instance?
    """

    def __init__(self, model, loss_func, optm, train_dl, valid_dl, scheduler=None,
                 acc_funcs=None, main_acc_func='loss', checkpoint_file='./learner.tar',
                 scheduler_step_epoch=True, callbacks=None, device=None, verbose=True):

        
        if scheduler is None:
            LambdaLR(optm, lambda x: 1)  # Fixed learning rate
        if acc_funcs is None:
            acc_funcs = {}
        if callbacks is None:
            callbacks = []
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

        self.model = model
        self.loss_func = loss_func
        self.optm = optm
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.scheduler = scheduler
        self.scheduler_init_state = scheduler.state_dict()
        self.acc_funcs = acc_funcs
        self.main_acc_func = main_acc_func
        self.checkpoint_file = checkpoint_file
        self.scheduler_step_epoch = scheduler_step_epoch
        self.callbacks = callbacks
        self.device = device 
        self.verbose = verbose

        self.train_loss_history = []
        self.valid_loss_history = []
        acc_funcs_history = {}
        for k, v in acc_funcs.items():
            acc_funcs_history[k] = []
        self.acc_funcs_history = acc_funcs_history

        #nb, nc, h, w = self.get_output_shape()

        self.lr_history = []
        self.epoch = 0
        self.checkpoint = {}       # Will store best model found
        self.best_score = None

    def fit(self, epochs, lr=None):
        """Train model for the given number of epochs. Each epoch consists in
        updating the weights for one pass in the training set and measuring loss
        and accuracies for one pass in the validation set.

        Parameters
        ----------
        epochs : int
            Number of epochs for training
        lr : float
            If a learning rate is passed, this new value is used for training indepently of the
            learning rate used when instantiating the class. Note that in this case learning rate 
            schedulers are ignored.
        """

        if lr is not None:
            # Fix the learning rate
            self.scheduler = None
            for pg in self.optm.param_groups:
                pg['lr'] = lr

        self.model.to(self.device)
        if self.verbose:
            self._print_epoch_info_header()
        for epoch in range(epochs):
            self._train_one_epoch()
            self._validate()

            if (self.scheduler is not None) and self.scheduler_step_epoch:
                self.lr_history.append(self.scheduler.get_last_lr())
                self.scheduler.step()
            if self.verbose:
                self._print_epoch_info()

            self._check_if_better_score()
            self.epoch += 1

    def _train_one_epoch(self):
        """Train model for one epoch."""

        self.model.train()
        train_loss = 0.
        for item_collection in self.train_dl:
            loss, _, _ = self._apply_model_to_batch(*item_collection)
            self.optm.zero_grad()
            loss.backward()
            self.optm.step()

            if (self.scheduler is not None) and (not self.scheduler_step_epoch):
                self.lr_history.append(self.scheduler.get_last_lr())
                self.scheduler.step()

            with torch.no_grad():
                train_loss += loss.item()

        self.train_loss_history.append(train_loss/len(self.train_dl))

    def _validate(self):
        """Validate the model for one epoch."""

        self.model.eval()
        valid_loss = 0.
        valid_acc = dict(zip(self.acc_funcs.keys(), [0.]*len(self.acc_funcs)))
        with torch.no_grad():
            for item_collection in self.valid_dl:
                loss, predb, yb = self._apply_model_to_batch(*item_collection)

                valid_loss += loss.item()
                accs = self._apply_acc_funcs(predb, yb)
                for key in accs:
                    valid_acc[key] += accs[key]

            self.valid_loss_history.append(valid_loss/len(self.valid_dl))
            for idx, (func_name, acc_func) in enumerate(self.acc_funcs.items()):
                self.acc_funcs_history[func_name].append(valid_acc[func_name]/len(self.valid_dl))

            for cb in self.callbacks:
                cb.on_epoch_end(item_collection[0], item_collection[1], predb, self.epoch)

    def _apply_model_to_batch(self, xb, yb, wb=None):
        """Given an input and target batch, and optionaly a loss weights batch, apply
        the model to the data and calculates loss.

        Parameters
        ----------
        xb : torch.Tensor
            Input data
        yb : torch.Tensor
            Target data
        wb : torch.Tensor
            Weights for each pixel

        Returns
        -------
        loss : torch.float
            The calculated loss
        predb : torch.Tensor
            The predictions of the model.
        yb : torch.Tensor
            Target data converted to long and on the correct self.evice.
        """

        device = self.device
        xb, yb = xb.to(device, torch.float32), yb.to(device, torch.long)
        predb = self.model(xb)

        if wb is None:
            loss = self.loss_func(predb, yb)
        else:
            wb = wb.to(device, torch.float32)
            loss = self.loss_func(predb, yb, wb)
        
        return loss, predb, yb

    def _print_epoch_info_header(self):
        """Print table header shown during training"""

        print_str = f'{"Epoch":<7}{"Train loss":>15}{"Valid loss":>15}'
        for func_name in self.acc_funcs:
            print_str += f'{func_name:>15}'
        print(print_str)

    def _print_epoch_info(self):
        """Print training and validation loss and accuracies calculated for the current epoch."""

        print_str = f'{self.epoch:5}{self.train_loss_history[-1]:17.3f}{self.valid_loss_history[-1]:15.3f}'
        for func_name in self.acc_funcs:
            acc_func_h = self.acc_funcs_history[func_name]
            print_str += f'{acc_func_h[-1]:15.3f}'
        print(print_str)

    def _apply_acc_funcs(self, predb, yb):
        """Apply each accuracy function to the data.

        Parameters
        ----------
        predb : torch.Tensor
            The model predictions
        yb : torch.Tensor
            The target

        Returns
        -------
        valid_acc : dict
            Dictionary containing the accuracy calculated for each function. Each key is the name
            of a function in self.acc_funcs.
        """

        valid_acc = {}
        for idx, (func_name, acc_func) in enumerate(self.acc_funcs.items()):
            valid_acc[func_name] = acc_func(predb, yb)
        return valid_acc

    def _check_if_better_score(self):
        """Check if the value of the main accuracy function has improved. If True,
        the model is saved in the file given by self.checkpoint_file."""

        score_improved = False
        prev_score = self.best_score

        if self.main_acc_func=='loss':
            score = self.valid_loss_history[-1]
            if (prev_score is None) or (score < prev_score):
                score_improved = True
        else:
            score = self.acc_funcs_history[self.main_acc_func][-1]
            if (prev_score is None) or (score > prev_score):
                score_improved = True

        if score_improved:
            #print(f'Score improved from {prev_score} to {score} checkpoint saved')
            self.best_score = score
            self.update_checkpoint()
            self.save_state(True)

    def get_state_dict(self):
        """Returns dictionary containing all relevant information about this class.

        Returns
        -------
        state_dict : dict
            Dictionary containing relevant class attributes
        """

        state_dict = {
                        'model_state' : self.model.state_dict(),
                        'optm_state' : self.optm.state_dict(),
                        'scheduler_state' : self.scheduler.state_dict(),
                        'epoch' : self.epoch,
                        'best_score' : self.best_score,
                        'model' : str(self.model),
                        'train_loss_history' : self.train_loss_history,
                        'valid_loss_history' : self.valid_loss_history,
                        'lr_history': self.lr_history,
                        'acc_funcs_history' : self.acc_funcs_history
                     }

        return state_dict

    def update_checkpoint(self):
        """Updates chekpoint of the model using current parameters."""

        self.checkpoint = self.get_state_dict()

    def save_state(self, checkpoint=False, filename=None):
        """Saves all the relevant information about this class.

        Parameters
        ----------
        checkpoint : bool
            If True, saves the parameters associated with the best model found during training, that is,
            the model providing the largest value of function acc_funcs[main_acc_func]. If False, saves
            the current parameters of the model.
        filename : string
            Filename to save the information. If None, it is given by self.checkpoint_file
        """

        if filename is None:
            filename = self.checkpoint_file

        if checkpoint:
            torch.save(self.checkpoint, filename)
        else:
            torch.save(self.get_state_dict(), filename)

    def load_state(self, filename=None):
        """Loads all the relevant information about this class from a file. Attributes of the
        class are updated with the information read.

        If you just need the trained model for making new predictions, it is possible to do:

        checkpoint = torch.load(filename)
        model = checkpoint['model_state']
        pred = model(img)

        instead of using this function.

        Parameters
        ----------
        filename : string
            Filename to load the information. If None, it is given by self.checkpoint_file
        """

        if filename is None:
            filename = self.checkpoint_file

        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optm.load_state_dict(checkpoint['optm_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.epoch = checkpoint['epoch']
        self.train_loss_history = checkpoint['train_loss_history']
        self.valid_loss_history = checkpoint['valid_loss_history']
        self.lr_history = checkpoint['lr_history']
        self.acc_funcs_history = checkpoint['acc_funcs_history']
        self.best_score = checkpoint['best_score']
        self.checkpoint = checkpoint

    def save_history(self, filename, sep=';'):
        """Save the loss and accuracy history to a file."""

        train_loss_history = self.train_loss_history
        valid_loss_history = self.valid_loss_history
        acc_funcs_history = self.acc_funcs_history

        header = f'Epoch{sep}Train loss{sep}Valid loss'
        for func_name in acc_funcs_history:
            header += f'{sep}{func_name}'

        with open(filename, 'w') as fd:
            fd.write(header+'\n')
            for epoch in range(len(train_loss_history)):
                line_str = f'{epoch+1}{sep}{self.train_loss_history[epoch]:.5f}{sep}{valid_loss_history[epoch]:.5f}'
                for func_name, acc_func_h in acc_funcs_history.items():
                    line_str += f'{sep}{acc_func_h[epoch]:.5f}'
                fd.write(line_str+'\n')

    def pred(self, xb, yb=None, return_classes=False):
        """Apply model to a batch, model parameters are not updated.

        If `yb` is provided, also returns the accuracy of the prediction for the
        given target.

        The possible returned values of this function are:

        if yb is None:
            if return_classes:
                Predicted classes
            else:
                Output of the model
        else:
            if return_classes:
                (Predicted classes, accuracies)
            else:
                (Output of the model, accuracies)

        Parameters
        ----------
        xb : torch.Tensor
            Input of the model. Must have shape (batch size, channels, height, width)
        yb : torch.Tensor
            The target. Must have shape (batch size, height, width)
        return_classes : bool
            If True, returns classes instead of probabilities. Also returns accuracy of
            the prediction

        Returns
        -------
        predb : torch.Tensor
            The predicted class probabilities. Only returned if `return_classes` is False
        bin_predb : torch.Tensor
            The predicted segmentation. Returned in place of `predb` if `return_classes` is True
        predb_acc : float
            Accuracies calculated for functions self.acc_funcs. Only returned if `yb` is not None

        # TODO: Implement TTA augmentation
        """

        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            xb = xb.to(self.device, torch.float32)
            predb = self.model(xb).to('cpu')

            if return_classes:
                classes_predb = torch.argmax(predb, dim=1).to('cpu', torch.uint8)

            if yb is None:
                if return_classes:
                    return classes_predb
                else:
                    return predb
            else:

                predb_acc = self._apply_acc_funcs(predb, yb)

                if return_classes:
                    return classes_predb, predb_acc
                else:
                    return predb, predb_acc

    def test(self, test_dl):
        """Calculate accuracies for a dataset. Calculated accuracies are given by the functions
        in self.acc_funcs.

        Parameters
        ----------
        test_dl : torch.Dataset
            The input dataset. Usually a dataset used for the testing phase.

        Returns
        -------
        test_acc : dict
            Dictionary of calculated accuracies with the same keys as self.acc_funcs
        """

        test_acc = dict(zip(self.acc_funcs.keys(), [0.]*len(self.acc_funcs)))
        with torch.no_grad():
            for xb, yb in test_dl:
                _, pred_acc = self.pred(xb, yb)
                for idx, (k, v) in enumerate(pred_acc.items()):
                    test_acc[k] += v

        test_acc = {k:v/len(test_dl) for k, v in test_acc.items()}

        return test_acc

    def get_output_shape(self):
        """Calculate the output shape of the model from one of the items in the dataset.

        Returns
        -------
        tuple
            Shape of the output from the model
        """

        xb, *_ = next(iter(self.train_dl.dataset[0]))

        self.model.to(self.device)
        xb = xb.to(self.device)
        with torch.no_grad():
            pred = self.model(xb)
        self.model.to('cpu')

        return pred.shape

    def get_history(self):
        """Return the recorded history for some parameters and evaluations of this learner.

        Returned values are:
        train_loss : the training loss
        valid_loss : the validation loss
        acc : accuracies calculated for functions stored in self.acc_funcs
        lr : learning rate

        Returns
        -------
        history : dict
            Dictionary keyed by the name of the property.
        """

        history = {
                    'train_loss':self.train_loss_history,
                    'valid_loss':self.valid_loss_history,
                    'acc':self.acc_funcs_history,
                    'lr':self.lr_history
        }
        return history

    def reset_history(self):
        """Reset the history stats for this learner."""

        self.train_loss_history = []
        self.valid_loss_history = []
        self.lr_history = []
        acc_funcs_history = {}
        for k, v in self.acc_funcs.items():
            acc_funcs_history[k] = []
        self.acc_funcs_history = acc_funcs_history

    def reset_training(self, optm, scheduler):
        """Reset parameters and history for this learner. Notice that this does not reset the optimizer and scheduler
        parameters! You can use functions set_optimizer() and set_scheduler() or reset_scheduler() for that."""

        self.model.reset_parameters()     # Will probably not work for fastai model
        self.reset_history()

    def reset_scheduler(self):
        """Reset the scheduler to its initial state."""

        self.scheduler.load_state_dict(self.scheduler_init_state)

    def set_optimizer(self, optm, *args, **kwargs):
        """Set or update the optimizer.

        Parameters
        ----------
        optm : torch.optim.Optimizer
            New optimizer
        """

        self.optm = optm(self.model, *args, **kwargs)

    def set_scheduler(self, scheduler, *args, **kwargs):
        """Set or update the learning rate scheduler.

        Parameters
        ----------
        optm : torch.optim.lr_scheduler._LRScheduler
            New scheduler
        """

        self.scheduler = scheduler(self.optm, *args, **kwargs)
