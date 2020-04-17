'''
Class used for training a Pytorch neural network.
'''

import torch

class Learner:
    """Class used for training a Pytorch neural network.

    The class is initialized with a CNN model, a loss function, an optimizer and train
    and validation datasets. The main mehods are:

    fit(epochs) : train the network for the given number of epochs
    pred(xb) : apply model to a batch
    save_state() and load_state() : save and load learner state
    get_history() : return the train and validation losses, accuracies and learning rates
                    for each epoch

    Note: the shape of the targets used for training and the prediction returned by pred() may
    be different than the shape of the input tensor, depending on the model used. A center crop
    is done on the tensors if the shapes of the tensor returned by the model and the target are different.

    Parameters
    ----------
    model : torch.nn
        The neural network to be trained
    loss_func : function
        The function used for calculating the loss. Should have signature loss_func(input,
        target, weight=None, epoch=None). `input` has shape (batch size, num classes, height, width)
        and target has shape (batch size, height, width). `weight` can be used for weighting the loss
        for each pixel (same shape as `target`) and `epoch` is the current training epoch.
    optm : torch.optim
        Optimizer used for updating the parameters
    train_dl : torch.Dataset
        Dataset used for training
    valid_dl : torch.Dataset
        Dataset used for validation
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
    device : torch.device
        Device used for training
    """

    def __init__(self, model, loss_func, optm, train_dl, valid_dl, scheduler=None,
                 acc_funcs=None, main_acc_func='loss', checkpoint_file='./learner.tar', device=None):

        if acc_funcs is None:
            self.acc_funcs = {}
        else:
            self.acc_funcs = acc_funcs
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
            self.device = device

        self.model = model
        self.loss_func = loss_func
        self.optm = optm
        self.train_dl = train_dl
        self.scheduler = scheduler
        self.valid_dl = valid_dl
        self.device = device
        self.main_acc_func = main_acc_func
        self.checkpoint_file = checkpoint_file

        self.train_loss_history = []
        self.valid_loss_history = []
        acc_funcs_history = {}
        for k, v in acc_funcs.items():
            acc_funcs_history[k] = []
        self.acc_funcs_history = acc_funcs_history

        nb, nc, h, w = self.get_output_shape()
        self.crop_shape = (nb, h, w)

        self.lr_history = []
        self.epoch = 0
        self.checkpoint = {}       # Will store best model found
        self.best_score = None

    def fit(self, epochs):
        '''Train model for the given number of epochs. Each epoch consists in
        updating the weights for one pass in the training set and measuring loss
        and accuracies for one pass in the validation set.

        Parameters
        ----------
        epochs : int
            Number of epochs for training
        '''

        # Returns model, but it is not necessary to assign to new variable since the layers
        # will be converted and copied to the GPU
        self.model.to(self.device)
        self._print_epoch_info_header()
        for epoch in range(epochs):
            self._fit_one_epoch()
            self._print_epoch_info()

            self._check_if_better_score()
            self.epoch += 1


    def _fit_one_epoch(self):
        '''Train model for one epoch. Also applies validation.
        '''

        self._train_one_epoch()
        self._validate_one_epoch()

        if self.scheduler is not None:
            self.lr_history.append(self.scheduler.get_lr())
            self.scheduler.step()

    def _train_one_epoch(self):
        '''Train model for one epoch.
        '''

        self.model.train()
        train_loss = 0.
        for xb, yb, wb in self.train_dl:
            loss = self._apply_to_batch(xb, yb, wb)
            loss.backward()
            self.optm.step()
            self.optm.zero_grad()

            with torch.no_grad():
                train_loss += loss.item()

        self.train_loss_history.append(train_loss/len(self.train_dl))

    def _validate_one_epoch(self):
        '''Validate the model for one epoch.
        '''

        self.model.eval()
        valid_loss = 0.
        valid_acc = [0.]*len(self.acc_funcs)
        with torch.no_grad():
            for xb, yb, wb in self.valid_dl:
                loss, predb, yb = self._apply_to_batch(xb, yb, wb, ret_data=True)

                valid_loss += loss.item()
                accs = self._apply_acc_funcs(predb, yb)
                for idx, v in enumerate(accs):
                    valid_acc[idx] += v

            self.valid_loss_history.append(valid_loss/len(self.valid_dl))
            for idx, (func_name, acc_func) in enumerate(self.acc_funcs.items()):
                self.acc_funcs_history[func_name].append(valid_acc[idx]/len(self.valid_dl))

    def _apply_to_batch(self, xb, yb, wb=None, ret_data=False):
        '''Given an input and target batch, and optionaly a loss weights batch, apply
        the model to the data and calculates loss.

        Parameters
        ----------
        xb : torch.Tensor
            Input data
        yb : torch.Tensor
            Target data
        wb : torch.Tensor
            Weights for each pixel
        ret_data : bool
            If False, only returns loss. If True, also returns predictions and the cropped
            target (in case the model returned a smaller output than the target)

        Returns
        -------
        loss : torch.float
            The calculated loss
        predb : torch.Tensor
            The predictions of the model. Only returned if ret_data is True
        yb_cropped : torch.Tensor
            The target cropped with the same size as the values returned by the model. Only returned
            if ret_data is True
        '''

        device = self.device
        xb, yb = xb.to(device, torch.float32), yb.to(device, torch.long)
        if wb is None:
            wb_cropped = None
        else:
            wb = wb.to(device, torch.float32)
            wb_cropped = self.center_crop_tensor(wb.squeeze(1), self.crop_shape)
        predb = self.model(xb)
        yb_cropped = self.center_crop_tensor(yb.squeeze(1), self.crop_shape)
        loss = self.loss_func(predb, yb_cropped, wb_cropped, self.epoch)

        if ret_data:
            return loss, predb, yb_cropped
        else:
            return loss

    def _print_epoch_info_header(self):
        '''Print table header shown during training'''

        print_str = f'{"Epoch":<7}{"Train loss":>15}{"Valid loss":>15}'
        for func_name in self.acc_funcs:
            print_str += f'{func_name:>15}'
        print(print_str)

    def _print_epoch_info(self):
        '''Print training and validation loss and accuracies calculated for the current epoch.'''

        print_str = f'{self.epoch:5}{self.train_loss_history[-1]:17.3f}{self.valid_loss_history[-1]:15.3f}'
        for func_name in self.acc_funcs:
            acc_func_h = self.acc_funcs_history[func_name]
            print_str += f'{acc_func_h[-1]:15.3f}'
        print(print_str)

    def _apply_acc_funcs(self, predb, yb):
        '''Apply each accuracy function to the data.

        Parameters
        ----------
        predb : torch.Tensor
            The model predictions
        yb : torch.Tensor
            The target

        Returns
        -------
        valid_acc : list
            List containing the accuracy calculated by each function
        '''

        valid_acc = []
        for idx, (func_name, acc_func) in enumerate(self.acc_funcs.items()):
            valid_acc.append(acc_func(predb, yb))
        return valid_acc

    def _check_if_better_score(self):
        '''Check if the value of the main accuracy function has improved. If True,
        the model is saved in the file given by self.checkpoint_file.'''

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
        '''Returns dictionary containing all relevant information about this class.

        Returns
        -------
        state_dict : dict
            Dictionary containing relevant class attributes
        '''

        state_dict = {
                        'model_state' : self.model.state_dict(),
                        'optm_state' : self.optm.state_dict(),
                        'scheduler_state' : self.scheduler.state_dict(),
                        'epoch' : self.epoch,
                        'best_score' : self.best_score,
                        'model' : str(self.model),
                        'train_loss_history' : self.train_loss_history,
                        'valid_loss_history' : self.valid_loss_history,
                        'acc_funcs_history' : self.acc_funcs_history
                     }

        return state_dict

    def update_checkpoint(self):
        '''Updates chekpoint of the model using current parameters.'''

        self.checkpoint = self.get_state_dict()

    def save_state(self, checkpoint=False, filename=None):
        '''Saves all the relevant information about this class.

        Parameters
        ----------
        checkpoint : bool
            If True, saves the parameters associated with the best model found during training, that is,
            the model providing the largest value of function acc_funcs[main_acc_func]. If False, saves
            the current parameters of the model.
        filename : string
            Filename to save the information. If None, it is given by self.checkpoint_file
        '''

        if filename is None:
            filename = self.checkpoint_file

        if checkpoint:
            torch.save(self.checkpoint, filename)
        else:
            torch.save(self.get_state_dict(), filename)

    def load_state(self, filename=None):
        '''Loads all the relevant information about this class from a file. Attributes of the
        class are updated with the information read.

        Parameters
        ----------
        filename : string
            Filename to load the information. If None, it is given by self.checkpoint_file
        '''

        if filename is None:
            filename = self.checkpoint_file

        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optm.load_state_dict(checkpoint['optm_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.epoch = checkpoint['epoch']
        self.train_loss_history = checkpoint['train_loss_history']
        self.valid_loss_history = checkpoint['valid_loss_history']
        self.acc_funcs_history = checkpoint['acc_funcs_history']
        self.best_score = checkpoint['best_score']
        self.checkpoint = checkpoint

    def pred(self, xb, yb=None, return_classes=False):
        '''Apply model to a batch, model parameters are not updated.

        If `y` is provided, also returns the accuracy of the prediction for the
        given target.

        Parameters
        ----------
        xb : torch.Tensor
            Input of the model. Must have shape (batch size, channels, height, width)
        yb : torch.Tensor
            The target. Must have shape (batch size, 1, height, width)
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
            Accuracies calculated for functions self.acc_funcs. Only returned if `yb` is True
        '''
        # TODO: Implement TTA augmentation

        self.model.eval()
        with torch.no_grad():
            xb = xb.to(self.device, torch.float32)
            predb = self.model(xb)

            if return_classes:
                classes_predb = torch.argmax(predb, dim=1).to('cpu', torch.uint8)

            if yb is None:
                if return_classes:
                    return classes_predb, None
                else:
                    return predb, None
            else:
                yb = yb.to(self.device, torch.long)
                yb_cropped = self.center_crop_tensor(yb.squeeze(1), (xb.shape[0],)+self.crop_shape[1:])

                predb_acc = self._apply_acc_funcs(predb, yb_cropped)

                if return_classes:
                    return classes_predb, predb_acc
                else:
                    return predb, predb_acc

    def test(self, test_dl):
        '''Calculate accuracies for a dataset. Calculated accuracies are given by the functions
        in self.acc_funcs.

        Parameters
        ----------
        test_dl : torch.Dataset
            The input dataset. Usually a dataset used for the testing phase.

        Returns
        -------
        list
            List of calculated accuracies in the same order as the functions stored in self.acc_funcs
        '''

        test_acc = [0.]*len(self.acc_funcs)
        with torch.no_grad():
            for xb,yb in test_dl:
                _, pred_acc = self.pred(xb, yb)
                for idx, v in enumerate(pred_acc):
                    test_acc[idx] += v

        return [v/len(test_dl) for v in test_acc]

    def get_output_shape(self):
        '''Calculate the output shape of the model from one of the items in the dataset.

        Returns
        -------
        tuple
            Shape of the output from the model
        '''

        xb, *_ = next(iter(self.train_dl))

        self.model.to(self.device)
        xb = xb.to(self.device)
        with torch.no_grad():
            pred = self.model(xb)
        self.model.to('cpu')

        return pred.shape

    @staticmethod
    def center_crop_tensor(tensor, out_shape):
        '''Center crop a tensor without copying its contents.

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor to be cropped
        out_shape : tuple
            Desired shape

        Returns
        -------
        tensor : torch.Tensor
            A new view of the tensor with shape out_shape
        '''

        out_shape = torch.tensor(out_shape)
        tensor_shape = torch.tensor(tensor.shape)
        shape_diff = (tensor_shape - out_shape)//2

        for dim_idx, sd in enumerate(shape_diff):
            tensor = tensor.narrow(dim_idx, sd, out_shape[dim_idx])

        return tensor

    @staticmethod
    def center_expand_tensor(self, tensor, out_shape):
        '''Center expand a tensor. Assumes `tensor` is not larger than `out_shape`

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor to be expanded
        out_shape : tuple
            Desired shape

        Returns
        -------
        torch.Tensor
            A new tensor with shape out_shape
        '''

        out_shape = torch.tensor(out_shape)
        tensor_shape = torch.tensor(tensor.shape)
        shape_diff = (out_shape - tensor_shape)

        pad = []
        for dim_idx, sd in enumerate(shape_diff.flip(0)):
            if sd%2==0:
                pad += [sd//2, sd//2]
            else:
                pad += [sd//2, sd//2+1]

        return F.pad(tensor, pad)

    def get_history(self):
        '''Return the recorded history for some parameters and evaluations of this learner.
        Returned values are:
        train_loss : the training loss
        valid_loss : the validation loss
        acc : accuracies calculated for functions stored in self.acc_funcs
        lr : learning rate

        Returns
        -------
        history : dict
            Dictionary keyed by the name of the property.
        '''

        history = {
                    'train_loss':self.train_loss_history,
                    'valid_loss':self.valid_loss_history,
                    'acc':self.acc_funcs_history,
                    'lr':self.lr_history
        }
        return history

    def reset_history(self):
        '''Reset the history for this learner
        '''

        self.train_loss_history = []
        self.valid_loss_history = []
        acc_funcs_history = {}
        for k, v in self.acc_funcs.items():
            acc_funcs_history[k] = []
        self.acc_funcs_history = acc_funcs_history

    def reset_training(self, optm, scheduler):
        '''Reset parameters and history for this learner. Notice that this does not reset optimizer and scheduler
        parameters!
        '''

        # TODO: verify if optimizer.state=collections.defaultdict(dict) would work for resetting optimizer
        self.model.reset_parameters()     # Will probably not work for fastai model
        self.reset_history()

    def set_optimizer(self, optm):
        '''Set or update optimizer.

        Parameters
        ----------
        optm : torch.optim.Optimizer
            New optimizer
        '''

        self.optm = optm

    def set_scheduler(self, scheduler):
        '''Set or update the learning rate scheduler.

        Parameters
        ----------
        optm : torch.optim.lr_scheduler._LRScheduler
            New scheduler
        '''

        self.scheduler = scheduler
