from pathlib import Path
import argparse
from datetime import datetime
import operator
import yaml
import torch
from torch import nn
import torch.utils
from torch.utils.data import DataLoader
from torchtrainer.util.train_util import Logger, WrapDict, seed_all, seed_worker, show_log
from torchtrainer.metrics import ConfusionMatrixMetrics
from torchtrainer.util.train_util import ParseKwargs

class ModuleRunner:
    """ Class to train, validate and test PyTorch models."""

    def add_dataset_elements(self, dl_train, dl_valid, loss_func, perf_funcs):
        self.dl_train = dl_train
        self.dl_valid = dl_valid
        self.loss_func = loss_func
        self.perf_funcs = perf_funcs

    def add_model_elements(self, model):
         self.model = model

    def add_training_elements(self, optim, scheduler, logger, device):
        self.optim = optim
        self.scheduler = scheduler
        self.logger = logger
        self.device = device
     
    def train_one_epoch(self, epoch: int):

        self.model.train()
        for batch_idx, (imgs, targets) in enumerate(self.dl_train):
            imgs = imgs.to(self.device)
            targets = targets.to(self.device)
            scores = self.model(imgs)
            loss = self.loss_func(scores, targets)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            self.logger.log(epoch, batch_idx, 'train/loss', loss.detach(), imgs.shape[0])

            #if (batch_idx+1)%self.print_every == 0:
            #    pass

        self.scheduler.step()
    
    @torch.no_grad()
    def validate_one_epoch(self, epoch: int):

        dl_valid = self.dl_valid
     
        self.model.eval()
        for batch_idx, (imgs, targets) in enumerate(dl_valid):
            imgs = imgs.to(self.device)
            targets = targets.to(self.device)
            scores = self.model(imgs)
            loss = self.loss_func(scores, targets)

            self.logger.log(epoch, batch_idx, 'valid/loss', loss, imgs.shape[0])
            for perf_func in self.perf_funcs:
                # Apply performance metric function
                results = perf_func(scores, targets)
                # Iterate over the results and log them
                for name, value in results.items():
                    self.logger.log(epoch, batch_idx, f'valid/{name}', value, imgs.shape[0])

            #if (batch_idx+1)%self.print_every == 0:
            #    pass
    
    @torch.no_grad()
    def predict(self, batch):
        """
        Method to apply after training the model to predict a single batch of data.
        """
        
        model = self.model
        batch_device = batch.device

        model.eval()
        output = model(batch.to(model.device)).to(batch_device)

        return output
    
    def state_dict(self):

        output = {
            'model':self.model.state_dict(),
            'optim':self.optim.state_dict(),
            'sched':self.scheduler.state_dict(),
            'logger':self.logger.epoch_data,
        }

        return output

class Trainer:
    """Class for setting up all components of a training experiment."""

    def __init__(self, commandline_string: str | None = None):
        """
        This class can be initialized from the command line as 
        
        python train.py --param1 value1 --param2 value2...

        or from another python script or notebook as

        Trainer('--param1 value1 --param2 value2...')

        Parameters
        ----------
        commandline_string
            If None, uses the command line arguments.
        """

        args = self.get_args(commandline_string)

        seed_all(args.seed)     
        torch.set_float32_matmul_precision('high')

        self.args = args
        self.module_runner = ModuleRunner()

    def setup_experiment(self):
        """Setup elements related to persistent data storation on the disk."""

        args = self.args
        experiment_path = args.experiment_path
        run_name = args.run_name

        # Create experiment and run directories
        experiment_path = Path(experiment_path)
        run_path = experiment_path/run_name
        if Path.exists(run_path):
            run_name_new = input('Run path already exists. Press enter to overwrite or write the name of the run: ')
            if run_name_new.strip():
                run_path = experiment_path/run_name_new
                args.run_name = run_name_new
        Path.mkdir(run_path, parents=True, exist_ok=True)
        self.run_path = run_path

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
        dataset_name = args.dataset_name
        dataset_path = args.dataset_path
        split_strategy = args.split_strategy
        augmentation_strategy = args.augmentation_strategy
        resize_size = args.resize_size
        # A dictionary with additional parameters that can be passed to get_dataset
        dataset_params = args.dataset_params

        if dataset_name=='oxford_pets':
            from torchtrainer.datasets.oxford_pets import get_dataset

            split = float(split_strategy)
            ds_train, ds_valid, *dataset_props = get_dataset(dataset_path, split, resize_size)
            class_weights, ignore_index, collate_fn = dataset_props
        else:
            raise ValueError(f'Dataset {dataset_name} not recognized')
        
        # Can be used to test the code with a smaller dataset
        #ds_train = [ds_train[idx] for idx in range(2*args.bs_train)]

        if args.ignore_class_weights:
            class_weights = (1.,)*len(class_weights)

        if ignore_index is None: ignore_index = -100    

        # Infer number of classes and channels from the dataset. Maybe unsafe?
        num_classes = len(class_weights)
        num_channels = ds_train[0][0].shape[0]

        # How the dataset should be evaluated
        loss_function = args.loss_function
        if loss_function=='cross_entropy':
            loss_func = nn.CrossEntropyLoss(torch.tensor(class_weights, device=args.device), 
                                            ignore_index=ignore_index or -100)
        else:
            raise ValueError(f'Loss function {loss_function} not recognized')

        perf_funcs = [
            WrapDict(ConfusionMatrixMetrics(ignore_index), ['Accuracy', 'IoU', 'Precision', 'Recall', 'Dice'])
        ]

        # Create dataloaders        
        num_workers = args.num_workers
        device = args.device

        dl_train = DataLoader(
            ds_train, 
            batch_size=args.bs_train, 
            shuffle=True, 
            num_workers=num_workers,
            persistent_workers=num_workers>0,
            worker_init_fn=seed_worker,
            pin_memory='cuda' in device,
        )
        dl_valid = DataLoader(
            ds_valid,
            batch_size=args.bs_valid, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=num_workers, 
            persistent_workers=num_workers>0,
            pin_memory='cuda' in device,
        )

        # Do a sanity check on the validation dataloader
        try:
            next(iter(dl_valid))
        except Exception as e:
            print('The following problem was detected on the validation dataloader:')
            raise e
        
        # We need to create these attributes to use them in the model setup
        dl_valid.num_classes = num_classes
        dl_valid.num_channels = num_channels
        self.module_runner.add_dataset_elements(dl_train, dl_valid, loss_func, perf_funcs)

    def setup_model(self):
        """Model creation. Each model must have a respective get_model function."""

        args = self.args
        model_name = args.model_name
        # String representing how to load the model weights
        weights_strategy = args.weights_strategy
        # Dictionary with additional parameters to pass to the model creation function
        model_params = args.model_params
        dl_valid = self.module_runner.dl_valid
        num_classes = dl_valid.num_classes
        num_channels = dl_valid.num_channels

        if model_name=='encoder_decoder':
            from torchtrainer.models.simple_encoder_decoder import get_model

            model = get_model(model_params['decoder_channels'], 
                              num_classes, weights_strategy)
        else:
            raise ValueError(f'Model {model} not recognized')
        
        device = args.device
        model.to(device)

        # Do a sanity check on the validation dataloader
        valid_batch = next(iter(dl_valid))[0]
        try:
            model(valid_batch.to(device))
        except Exception as e:
            print("The following error happened when applying the model to the validation batch:")
            raise e
        
        self.module_runner.add_model_elements(model)
        
    def setup_training(self):
        """Setup the training elements: optimizer, scheduler, logger, device"""

        args = self.args

        num_epochs = args.num_epochs
        optimizer = args.optimizer
        momentum = args.momentum

        model = self.module_runner.model
        
        if optimizer=='sgd':
            optim = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                    momentum=momentum)
        elif optimizer=='adam':
            optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                    betas=(momentum, 0.999))
        elif optimizer=='adamw':
            optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                    betas=(momentum, 0.999))
        
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optim, num_epochs)
            
        logger = Logger()
        self.module_runner.add_training_elements(optim, scheduler, logger, args.device)

    def train(self):
        """Setup averything and run the training loop."""

        self.setup_experiment()
        self.setup_dataset()
        self.setup_model()
        self.setup_training()

        args = self.args
        runner = self.module_runner
        logger = runner.logger
        run_path = self.run_path

        # Validation metric for early stopping
        val_metric_name = f'valid/{args.validation_metric}'
        maximize = args.maximize_validation_metric
        # Set gt or lt operator depending on maximization or minimization problem
        compare = operator.gt if maximize else operator.lt
        best_val_metric = -torch.inf if maximize else torch.inf

        epochs_without_improvement = 0
        for epoch in range(0, args.num_epochs):
            runner.train_one_epoch(epoch)
            runner.validate_one_epoch(epoch)
            # Aggregate batch metrics into epoch metrics
            logger.end_epoch()

            show_log(logger)

            logger.get_data().to_csv(run_path/'log.csv', index=False)
            
            checkpoint = runner.state_dict()

            if (epoch+1)%args.save_every==0:
                torch.save(checkpoint, run_path/'checkpoint.pt')

            # Check for model improvement
            val_metric = logger.get_data()[val_metric_name].iloc[-1]
            if compare(val_metric, best_val_metric):
                torch.save(checkpoint, run_path/'best_model.pt')
                best_val_metric = val_metric
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                # No improvement for `patience`` epochs
                if epochs_without_improvement>args.patience:
                    break
            
        # Save the last checkpoint in case save_every is not multiple of num_epochs
        torch.save(checkpoint, run_path/'checkpoint.pt')

        # Include training end time in the config file
        config_dict = vars(args)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        config_dict['timestamp_end'] = timestamp
        args_yaml = yaml.safe_dump(config_dict, default_flow_style=False)
        open(run_path/'config.yaml', 'w').write(args_yaml)

    def get_args(self, commandline_string: str | None = None) -> argparse.Namespace:
        """Parse command line arguments or arguments from a string. Adapted from
        the timm training script

        Parameters
        ----------
        commandline_string
            A string containing the command line arguments to parse.

        Returns
        -------
        args
            An argparse namespace object containing the parsed arguments
        """

        if commandline_string is not None:
            commandline_string = commandline_string.split()

        parser, config_parser = self.get_parser()

        # If option --config was used, load the respective yaml file and set
        # the values for the main parser
        args_config, remaining = config_parser.parse_known_args(commandline_string)
        if args_config.config:
            with open(args_config.config, 'r') as f:
                cfg = yaml.safe_load(f)
                parser.set_defaults(**cfg)

        # The main arg parser parses the rest of the args, the usual
        # defaults will have been overridden if config file specified.
        args = parser.parse_args(remaining)

        return args

    def get_parser(self) -> argparse.ArgumentParser:

        # TODO: do not log on every batch

        # The config_parser parses only the --config argument, this argument is used to
        # load a yaml file containing key-values that override the defaults for the main parser below
        config_parser = argparse.ArgumentParser(description='Training Config', add_help=False)
        config_parser.add_argument('--config', default='', metavar='FILE',
                            help='Path to YAML config file specifying default arguments')

        parser = argparse.ArgumentParser()

        # Logging parameters
        group = parser.add_argument_group('Logging parameters')
        group.add_argument('--experiment-path', default='experiments/no_name_experiment', help='Path to save experiment data')
        group.add_argument('--run-name', default='no_name_run', help='Name of the run for a given experiment')
        group.add_argument('--save-every', type=int, default=1, help='Save a model checkpoint every n epochs')
        group.add_argument('--meta', default='', help='Additional metadata to save in the config.json file describing the experiment')
        
        # Dataset parameters
        group = parser.add_argument_group('Dataset parameters')
        group.add_argument('--dataset-name', help='Name of the dataset')
        group.add_argument('--dataset-path', help='Path to the dataset files. By default, ./dataset-name is used.')        
        group.add_argument('--split-strategy', default='0.2', 
                        help='How to split the data into train/val. This parameter can be any string that is then passed to the dataset creation function')
        group.add_argument('--augmentation-strategy', help='Data augmentation procedure. Can be any string and is passed to the dataset creation function')
        group.add_argument('--resize-size', default=(384,384), nargs=2, type=int, help='Size to resize the images')
        group.add_argument('--dataset-params', nargs='*', default={}, action=ParseKwargs,
                        help='Additional parameters to pass to the dataset creation function. E.g. --dataset-params a=1 b=2 c=3'
                        'The additional parameters are evaluated as Python code and cannot contain spaces.')

        group.add_argument('--ignore-class-weights', action='store_true', help='If provided, ignore class weights on the loss function')
        
        # Model parameters
        group = parser.add_argument_group('Model parameters')
        group.add_argument('--model-name', help='Name of the model to train')
        group.add_argument('--weights-strategy', 
            help='This argument is sent to the model creation function and can be used to define how to load the weights')
        group.add_argument('--model-params', nargs='*', default={}, action=ParseKwargs,
                        help='Additional parameters to pass to the model creation function. E.g. --model-params a=1 b=2 c=3')
        
        # Training parameters
        group = parser.add_argument_group('Training parameters')
        group.add_argument('--num-epochs', type=int, default=10, help='Number of training epochs')
        group.add_argument('--patience', type=int, default=10, help='Early stopping. Finish training if validation metric does not improve for `patience` epochs')
        group.add_argument('--validation-metric', default='loss', help='Which metric to use for early stopping')
        group.add_argument('--maximize-validation-metric', action='store_true', help='If the validation metric should be maximized or minimized')
        group.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
        group.add_argument('--bs-train', type=int, default=32, help='Batch size used durig training')
        group.add_argument('--bs-valid', type=int, default=8, help='Batch size used durig validation')
        group.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for the optimizer')
        group.add_argument('--loss-function', default='cross_entropy', help='Loss function to use during training')
        group.add_argument('--optimizer', default='sgd', help='Optimizer to use')
        group.add_argument('--momentum', type=float, default=0.9, help='Momentum of the optimizer')
        group.add_argument('--seed', type=int, default=0, help='Seed for the random number generator')
        
        # Device and efficiency parameters
        group = parser.add_argument_group('Device and efficiency parameters')
        group.add_argument('--device', default='cuda:0', help='where to run the training code (e.g. "cpu" or "cuda:0")')
        group.add_argument('--num-workers', type=int, default=0, help='Number of workers for the DataLoader')
        group.add_argument('--use-amp', action='store_true', help='If automatic mixed precision should be used')

        return parser, config_parser


if __name__ == '__main__':
    Trainer().train()