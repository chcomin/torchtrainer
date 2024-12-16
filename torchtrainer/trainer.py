from pathlib import Path
import shutil
import argparse
from datetime import datetime
import operator
import yaml
from tqdm.auto import tqdm
import torch
from torch import nn
import torch.utils
from torch.utils.data import DataLoader
from torchtrainer.util.train_util import Logger, LoggerPlotter, WrapDict, dict_to_argv
from torchtrainer.util.train_util import seed_all, seed_worker, predict_and_save_val_img
from torchtrainer.util.train_util import ParseKwargs, ParseText
from torchtrainer.metrics import ConfusionMatrixMetrics


# TODO: wandb 
# TODO: profiling

class DefaultModuleRunner:
    """ Class to train, validate and test PyTorch models."""

    def add_dataset_elements(
            self, 
            ds_train, 
            ds_valid, 
            num_classes,
            num_channels,
            collate_fn,
            loss_func, 
            perf_funcs,
            logger,
            logger_plotter):
        self.ds_train = ds_train
        self.ds_valid = ds_valid
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.collate_fn = collate_fn
        self.loss_func = loss_func
        self.perf_funcs = perf_funcs
        self.logger = logger
        self.logger_plotter = logger_plotter

    def add_model_elements(self, model):
         self.model = model

    def add_training_elements(self, dl_train, dl_valid, optim, scheduler,
                              scaler, device):
        self.dl_train = dl_train
        self.dl_valid = dl_valid
        self.optim = optim
        self.scheduler = scheduler
        self.scaler = scaler
        self.device = device
     
    def train_one_epoch(self, epoch: int):

        self.model.train()
        scaler = self.scaler

        pbar = tqdm(
            self.dl_train,
            desc='Training',
            leave=False,
            unit='batchs',
            dynamic_ncols=True,
            colour='blue',
        )
        for batch_idx, (imgs, targets) in enumerate(pbar):
            imgs = imgs.to(self.device)
            targets = targets.to(self.device)
            self.optim.zero_grad()
            with torch.autocast(device_type=self.device, enabled=scaler.is_enabled()):    # Forward pass in mixed precision
                scores = self.model(imgs)
                loss = self.loss_func(scores, targets)
            
            scaler.scale(loss).backward()
            scaler.step(self.optim)
            scaler.update()

            self.logger.log(epoch, batch_idx, 'Train loss', loss.detach(), imgs.shape[0])

        # Log learning rate
        self.logger.log_epoch(epoch, 'lr', self.optim.param_groups[0]['lr'])
        self.scheduler.step()
    
    @torch.no_grad()
    def validate_one_epoch(self, epoch: int):

        self.model.eval()

        pbar = tqdm(
            self.dl_valid,
            desc='Validating',
            leave=False,
            unit='batchs',
            dynamic_ncols=True,
            colour='green',
        )    
        for batch_idx, (imgs, targets) in enumerate(pbar):
            imgs = imgs.to(self.device)
            targets = targets.to(self.device)
            scores = self.model(imgs)
            loss = self.loss_func(scores, targets)

            self.logger.log(epoch, batch_idx, 'Validation loss', loss, imgs.shape[0])
            for perf_func in self.perf_funcs:
                # Apply performance metric function
                results = perf_func(scores, targets)
                # Iterate over the results and log them
                for name, value in results.items():
                    self.logger.log(epoch, batch_idx, name, value, imgs.shape[0])
    
    @torch.no_grad()
    def predict(self, batch):
        """
        Method to apply after training the model to predict a single batch of data.
        """
        
        model = self.model
        training = model.training
        batch_device = batch.device

        model.eval()
        output = model(batch.to(self.device)).to(batch_device)

        model.train(training)

        return output
    
    def state_dict(self):

        output = {
            'model':self.model.state_dict(),
            'optim':self.optim.state_dict(),
            'sched':self.scheduler.state_dict(),
            'scaler':self.scaler.state_dict(),
            'logger':self.logger.epoch_data,
        }

        return output

class DefaultTrainer:
    """Class for setting up all components of a training experiment."""

    def __init__(self, param_dict: dict | None = None):
        """
        This class can be initialized from the command line as 
        
        python trainer.py --param1 value1 --param2 value2...

        or from a python script or notebook using a dictionary:

        params = {param1:value1, param2:value2, ...}
        DefaultTrainer(params)

        When using a dictionary:
        
        1. Do not include "--" before parameter names. 
        2. All values except single int or float numbers should be strings.
        3. The value of boolean parameters should be an empty string, e.g. 'bool-par': ''.

        Parameters
        ----------
        param_dict
            If None, uses the command line arguments.
        """

        args = self.get_args(param_dict)

        seed_all(args.seed)     

        self.args = args
        self.module_runner = DefaultModuleRunner()
        print("Setting up the experiment...")
        self.setup_experiment()
        self.setup_dataset()
        self.setup_model()
        self.setup_training()
        print("Done setting up.")

    def setup_experiment(self):
        """Setup elements related to persistent data storation on the disk."""

        args = self.args
        experiments_path = args.experiments_path
        experiment_name = args.experiment_name
        run_name = args.run_name

        # Create experiment and run directories
        experiment_path = Path(experiments_path)/experiment_name
        run_path = experiment_path/run_name
        if Path.exists(run_path):
            run_name_new = input('Run path already exists. Press enter to overwrite or write the name of the run: ')
            if run_name_new.strip():
                run_path = experiment_path/run_name_new
                args.run_name = run_name_new
            else:
                shutil.rmtree(run_path)
        Path.mkdir(run_path, parents=True, exist_ok=True)
        self.run_path = run_path

        if args.save_val_img:
            Path.mkdir(run_path/'images', exist_ok=True)
        if args.copy_model_every:
            Path.mkdir(run_path/'models', exist_ok=True)

        # Register training start time
        config_dict = vars(args)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        config_dict['timestamp_start'] = timestamp

        # Save the config file
        args_yaml = yaml.safe_dump(config_dict, default_flow_style=False)
        open(run_path/'config.yaml', 'w').write(args_yaml)

    def setup_dataset(self):
        """ 
        Setup the dataset and related elements.
        Each dataset must have a respective get_dataset function. This section of the
        training script can be adapted to send the relevant parameters to get_dataset

        All datasets must return:
        - Train and validation datasets
        - The class weights
        - If a value should be ignored in the target (ignore_index)
        - A collate function indicating how to batch the data
        ignore_index and collate_fn can be None.
        """

        args = self.args
        # Command line arguments that can be used here:
        dataset_class = args.dataset_class
        dataset_path = args.dataset_path
        split_strategy = args.split_strategy
        augmentation_strategy = args.augmentation_strategy
        resize_size = args.resize_size
        # A dictionary with additional parameters that can be passed to get_dataset
        dataset_params = args.dataset_params

        seed_all(args.seed)

        if dataset_class=='oxford_pets':
            from torchtrainer.datasets.oxford_pets import get_dataset

            split = float(split_strategy)
            ds_train, ds_valid, *dataset_props = get_dataset(dataset_path, split, resize_size)
            class_weights, ignore_index, collate_fn = dataset_props
        else:
            raise ValueError(f'Dataset {dataset_class} not recognized')
        
        # Can be used to test the code with a smaller dataset
        #ds_train = [ds_train[idx] for idx in range(2*args.bs_train)]

        if args.ignore_class_weights:
            class_weights = (1.,)*len(class_weights)
        if ignore_index is None: 
            ignore_index = -100    

        # How the dataset should be evaluated
        loss_function = args.loss_function
        if loss_function=='cross_entropy':
            loss_func = nn.CrossEntropyLoss(torch.tensor(class_weights, device=args.device), 
                                            ignore_index=ignore_index)
        else:
            raise ValueError(f'Loss function {loss_function} not recognized')

        perf_funcs = [
            WrapDict(ConfusionMatrixMetrics(ignore_index), ['Accuracy', 'IoU', 'Precision', 'Recall', 'Dice'])
        ]

        logger = Logger()
        # How to group the data when plotting, each group becomes an individual plot
        logger_plotter = LoggerPlotter([
            {'names': ['Train loss', 'Validation loss'], 'y_max': 1.},
            {'names': ['Accuracy', 'IoU', 'Precision', 'Recall', 'Dice'], 'y_max': 1.}
        ])

        num_classes = len(class_weights)
        # Infer number o channels from the dataset. Maybe unsafe?
        num_channels = ds_train[0][0].shape[0]
        self.module_runner.add_dataset_elements(
            ds_train, ds_valid, num_classes, num_channels, 
            collate_fn, loss_func, perf_funcs, logger, logger_plotter)

    def setup_model(self):
        """Model creation. Each model must have a respective get_model function."""

        args = self.args
        model_name = args.model_name
        # String representing how to load the model weights
        weights_strategy = args.weights_strategy
        # Dictionary with additional parameters to pass to the model creation function
        model_params = args.model_params
        num_classes = self.module_runner.num_classes
        num_channels = self.module_runner.num_channels

        seed_all(args.seed)

        if model_name=='encoder_decoder':
            from torchtrainer.models.simple_encoder_decoder import get_model

            model = get_model(**model_params, num_classes=num_classes, 
                              weights_strategy=weights_strategy)
        elif model_name=='test_model':
            from torchtrainer.models.testing import TestSegmentation

            model = TestSegmentation(num_channels=num_channels, num_classes=num_classes)
        else:
            raise ValueError(f'Model {model_name} not recognized')
                
        self.module_runner.add_model_elements(model)
        
    def setup_training(self):
        """Setup the training elements: dataloaders, optimizer, scheduler and logger"""

        args = self.args
        num_epochs = args.num_epochs
        optimizer = args.optimizer
        momentum = args.momentum
        module_runner = self.module_runner

        torch.backends.cudnn.deterministic = args.deterministic
        torch.backends.cudnn.benchmark = args.benchmark      
        torch.set_float32_matmul_precision('high') 

        # Create dataloaders        
        num_workers = args.num_workers
        device = args.device

        dl_train = DataLoader(
            module_runner.ds_train, 
            batch_size=args.bs_train, 
            shuffle=True, 
            collate_fn=module_runner.collate_fn,
            num_workers=num_workers,
            persistent_workers=num_workers>0,
            worker_init_fn=seed_worker,
            pin_memory='cuda' in device,
        )
        dl_valid = DataLoader(
            module_runner.ds_valid,
            batch_size=args.bs_valid, 
            shuffle=False, 
            collate_fn=module_runner.collate_fn,
            num_workers=num_workers, 
            persistent_workers=num_workers>0,
            pin_memory='cuda' in device,
        )

        device = args.device
        model = module_runner.model
        model.to(device)

        # Do a sanity check on the validation dataloader
        try:
           batch = next(iter(dl_valid))
        except Exception as e:
            print('The following problem was detected on the validation dataloader:')
            raise e

        # Do a sanity check on the model
        try:
            model(batch[0].to(device))
        except Exception as e:
            print("The following error happened when applying the model to the validation batch:")
            raise e
        
        if optimizer=='sgd':
            optim = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                    momentum=momentum)
        elif optimizer=='adam':
            optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                    betas=(momentum, 0.999))
        elif optimizer=='adamw':
            optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                    betas=(momentum, 0.999))
        
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optim, num_epochs, args.lr_decay)

        scaler = torch.GradScaler(device=device, enabled=args.use_amp)
            
        self.module_runner.add_training_elements(dl_train, dl_valid, optim, scheduler, 
                                                 scaler, args.device)

    def fit(self):
        """Start the training loop."""

        args = self.args
        runner = self.module_runner
        logger = runner.logger
        logger_plotter = runner.logger_plotter
        run_path = self.run_path

        seed_all(args.seed)

        # Validation metric for early stopping
        val_metric_name = args.validation_metric
        maximize = args.maximize_validation_metric
        # Set gt or lt operator depending on maximization or minimization problem
        compare = operator.gt if maximize else operator.lt
        best_val_metric = -torch.inf if maximize else torch.inf

        checkpoint = runner.state_dict()
        epochs_without_improvement = 0
        print("Training has started")
        pbar = tqdm(
            range(args.num_epochs),
            desc='Epochs',
            leave=True,
            unit='epochs',
            dynamic_ncols=True,
            colour='blue',
        )
        try:
            for epoch in pbar:
                runner.train_one_epoch(epoch)
                validate = epoch==0 or epoch==args.num_epochs-1 or epoch%args.validate_every==0

                if validate:
                    runner.validate_one_epoch(epoch)

                # Aggregate batch metrics into epoch metrics and get the data
                logger.end_epoch()
                logger_data = logger.get_data()
                last_metrics = logger_data.iloc[-1]

                # Set the metrics to be displayed in the progress bar
                tqdm_metrics = ['Train loss']
                if validate:
                    tqdm_metrics += ['Validation loss', val_metric_name]
                pbar.set_postfix(last_metrics[tqdm_metrics].to_dict()) 

                # Save logged data
                logger_data.to_csv(run_path/'log.csv', index=False)
                # Save plot of logged data
                logger_plotter.get_plot(logger).savefig(run_path/'plots.png')
                
                checkpoint = runner.state_dict()

                # Save the checkpoint and a copy of it if required
                torch.save(checkpoint, run_path/'checkpoint.pt')
                if args.copy_model_every and epoch%args.copy_model_every==0:
                    torch.save(checkpoint, run_path/'models'/f'checkpoint_{epoch}.pt')

                if validate:
                    if args.save_val_img:
                        predict_and_save_val_img(runner, epoch, args.val_img_idx, run_path)                  

                    # Check for model improvement
                    val_metric = last_metrics[val_metric_name]
                    if compare(val_metric, best_val_metric):
                        torch.save(checkpoint, run_path/'best_model.pt')
                        best_val_metric = val_metric
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                        # No improvement for `patience`` epochs
                        if epochs_without_improvement>args.patience:
                            break

        except KeyboardInterrupt:
            # This exception allows interrupting the training loop with Ctrl+C,
            # but sometimes it does not work due to the multiprocessing DataLoader
            pass

        print("Training has finished")

        # Include training end time in the config file
        config_dict = vars(args)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        config_dict['timestamp_end'] = timestamp
        args_yaml = yaml.safe_dump(config_dict, default_flow_style=False)
        open(run_path/'config.yaml', 'w').write(args_yaml)

    def get_args(self, param_dict: dict | None = None) -> argparse.Namespace:
        """Parse command line arguments or arguments from a string.

        Parameters
        ----------
        param_dict
            A dictionary containing the command line arguments to parse.

        Returns
        -------
        args
            An argparse namespace object containing the parsed arguments
        """

        if param_dict is None:
            sys_argv = None
        else:
            positional_args = ['dataset-path', 'dataset-class', 'model-name']
            sys_argv = dict_to_argv(param_dict, positional_args)

        parser, config_parser = self.get_parser()

        # If option --config was used, load the respective yaml file and set
        # the values for the main parser
        args_config, remaining = config_parser.parse_known_args(sys_argv)
        if args_config.config:
            with open(args_config.config, 'r') as f:
                cfg = yaml.safe_load(f)
                parser.set_defaults(**cfg)

        # The main arg parser parses the rest of the args, the usual
        # defaults will have been overridden if config file specified.
        args = parser.parse_args(remaining)

        return args

    def get_parser(self) -> argparse.ArgumentParser:

        # The config_parser parses only the --config argument, this argument is used to
        # load a yaml file containing key-values that override the defaults for the main parser below
        config_parser = argparse.ArgumentParser(description='Training Config', add_help=False)
        config_parser.add_argument('--config', default='', metavar='FILE',
                            help='Path to YAML config file specifying default arguments')

        parser = argparse.ArgumentParser(description='Below, N represents integer values and V represents float values')

        # Logging parameters
        group = parser.add_argument_group('Logging parameters')
        group.add_argument('-p', '--experiments-path', default='experiments', metavar='PATH', help='Path to save experiments data')
        group.add_argument('-e', '--experiment-name', default='no_name_experiment', metavar='NAME', help='Name of the experiment')
        group.add_argument('-n', '--run-name', default='no_name_run', metavar='NAME', help='Name of the run for a given experiment')
        group.add_argument('--validate-every', type=int, default=1, metavar='N', help='Run a validation step every N epochs')
        group.add_argument('--save-val-img', action='store_true', help='Save a validation image when validating')
        group.add_argument('--val-img-idx', type=int, default=0, metavar='N', help='Index of the validation image to save')
        group.add_argument('--copy-model-every', type=int, default=0, metavar='N', 
                           help='Save a copy of the model every N epochs. If 0 (default) no copies are saved')
        parser.add_argument('--meta', default='', nargs='*', action=ParseText, help='Additional metadata to save in the config.json file '
                            'describing the experiment. Whitespaces do not need to be escaped.')

        # Dataset parameters
        group = parser.add_argument_group('Dataset parameters')
        group.add_argument('dataset_path', help='Path to the dataset root directory')        
        group.add_argument('dataset_class', help='Name of the dataset class to use')
        group.add_argument('--split-strategy', default='0.2',  metavar='STRING',
                        help='How to split the data into train/val. This parameter can be any string that is then passed to the dataset creation function')
        group.add_argument('--augmentation-strategy', default=None, metavar='STRING', 
                           help='Data augmentation procedure. Can be any string and is passed to the dataset creation function')
        group.add_argument('--resize-size', default=(384,384), nargs=2, type=int, metavar=('N', 'N'), help='Size to resize the images. E.g. --resize-size 128 128')
        group.add_argument('--dataset-params', nargs='*', default={}, action=ParseKwargs, metavar='par1=v1 par2=v2 par3=v3', 
                        help='Additional parameters to pass to the dataset creation function. E.g. --dataset-params par1=v1 par2=v2 par3=v3. '
                        'The additional parameters are evaluated as Python code and cannot contain spaces.')
        group.add_argument('--loss-function', default='cross_entropy', metavar='LOSS', help='Loss function to use during training')
        group.add_argument('--ignore-class-weights', action='store_true', help='If provided, ignore class weights for the loss function')

        # Model parameters
        group = parser.add_argument_group('Model parameters')
        group.add_argument('model_name', help='Name of the model to train')
        group.add_argument('--weights-strategy', default=None, metavar='STRING', 
            help='This argument is sent to the model creation function and can be used to define how to load the weights')
        group.add_argument('--model-params', nargs='*', default={}, action=ParseKwargs, metavar='par1=v1 par2=v2 par3=v3', 
                        help='Additional parameters to pass to the model creation function. E.g. --model-params par1=v1 par2=v2 par3=v3')

        # Training parameters
        group = parser.add_argument_group('Training parameters')
        group.add_argument('--num-epochs', type=int, default=2, metavar='N', help='Number of training epochs')
        group.add_argument('--validation-metric', default='Validation loss', nargs='*', metavar='METRIC', action=ParseText, help='Which metric to use for early stopping')
        group.add_argument('--patience', type=int, default=50, metavar='N', help='Finish training if validation metric does not improve for N epochs')
        group.add_argument('--maximize-validation-metric', action='store_true', 
                           help='If set, early stopping will maximize the validation metric instead of minimizing')
        group.add_argument('--lr', type=float, default=0.01, metavar='V', help='Initial learning rate')
        group.add_argument('--lr-decay', type=float, default=1., metavar='V', help='Learning rate decay')
        group.add_argument('--bs-train', type=int, default=32, metavar='N', help='Batch size used durig training')
        group.add_argument('--bs-valid', type=int, default=8, metavar='N', help='Batch size used durig validation')
        group.add_argument('--weight-decay', type=float, default=1e-4, metavar='V', help='Weight decay for the optimizer')
        group.add_argument('--optimizer', default='sgd', help='Optimizer to use')
        group.add_argument('--momentum', type=float, default=0.9, metavar='V', help='Momentum/beta1 of the optimizer')
        group.add_argument('--seed', type=int, default=0, metavar='N', help='Seed for the random number generator')

        # Device and efficiency parameters
        group = parser.add_argument_group('Device and efficiency parameters')
        group.add_argument('--device', default='cuda:0', help='where to run the training code (e.g. "cpu" or "cuda:0")')
        group.add_argument('--num-workers', type=int, default=5, metavar='N', help='Number of workers for the DataLoader')
        group.add_argument('--use-amp', action='store_true', help='If automatic mixed precision should be used')
        group.add_argument('--deterministic', action='store_true', help='If deterministic algorithms should be used')
        group.add_argument('--benchmark', action='store_true', help='If cuda benchmark should be used')

        return parser, config_parser


if __name__ == '__main__':
    DefaultTrainer().fit()