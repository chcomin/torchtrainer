from pathlib import Path
import inspect
import argparse
from datetime import datetime
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

from torchtrainer.datasets.oxford_pets import get_dataset, collate_fn
from torchtrainer.util.train_util import Logger, MultipleMetrics, seed_all, seed_worker, show_log
from torchtrainer.metrics import confusion_matrix_metrics
from torchtrainer.models import SimpleEncoderDecoder

class ModuleRunner:
    """
    Class to train, validate and test PyTorch models.
    """

    def __init__(
            self, 
            model, 
            dl_train, 
            dl_valid, 
            loss_func, 
            optim, 
            scheduler,
            perf_funcs,  # List of performance functions to apply during validation
            logger,
            device,
        ): 

        # Save all arguments as class attributes
        frame = inspect.currentframe()
        args, varargs, varkw, values = inspect.getargvalues(frame)
        for arg in args:
            setattr(self, arg, values[arg])
     
    def train_one_epoch(self, epoch):

        self.model.train()
        loss_log = 0.
        for imgs, targets in self.dl_train:
            imgs = imgs.to(self.device)
            targets = targets.to(self.device)
            self.optim.zero_grad()
            scores = self.model(imgs)
            loss = self.loss_func(scores, targets)
            loss.backward()
            self.optim.step()

            loss_log += loss.detach()*imgs.shape[0]

        loss_log /= len(self.dl_train.dataset)
        self.logger.log(epoch, 'train/loss', loss_log.item())
        self.scheduler.step()
    
    @torch.no_grad()
    def validate_one_epoch(self, epoch):

        dl_valid = self.dl_valid
     
        self.model.eval()
        loss_log = 0.
        perf_log = {}
        for batch_idx, (imgs, targets) in enumerate(dl_valid):
            imgs = imgs.to(self.device)
            targets = targets.to(self.device)
            scores = self.model(imgs)
            loss = self.loss_func(scores, targets)

            loss_log += loss*imgs.shape[0]
            # Calculate performance metrics and save them on perf_log
            self._performance_metrics(scores, targets, perf_log, imgs.shape[0])

        loss_log /= len(dl_valid.dataset)
        self.logger.log(epoch, 'valid/loss', loss_log.item())
        for name, value in perf_log.items():
            value = value/len(dl_valid.dataset)
            self.logger.log(epoch, f'valid/{name}', value.item())

    @torch.no_grad()
    def test(self):
        """
        This method should implement the test procedure after training the model.
        It may do, for instante, test-time augmentation
        """
        pass
    
    def predict(self, batch):
        """
        Method to apply after training the model to predict a single batch of data.
        """
        
        model = self.model
        batch_device = batch.device

        model.eval()
        output = model(batch.to(model.device)).to(batch_device)

        return output

    def _performance_metrics(self, scores, targets, perf_log, n_items):
        """Calculate performance metrics for a batch of data."""
        for perf_func in self.perf_funcs:
            # Apply performance metric function
            results = perf_func(scores, targets)
            # Iterate over the results and save them on perf_log
            for name, value in results:
                weighted_value = value*n_items
                if name not in perf_log:
                    perf_log[name] = weighted_value
                else:
                    perf_log[name] += weighted_value

def train(arg_dict=None):

    args = get_args(arg_dict)
    dataset_name = args.dataset_name
    experiment_path = args.experiment_path
    run_name = args.run_name
    model_name = args.model_name
    bs_train = args.bs_train
    bs_valid = args.bs_valid
    num_epochs = args.num_epochs
    lr = args.lr
    weight_decay = args.weight_decay
    resize_size = args.resize_size
    seed = args.seed
    device = args.device
    num_workers = args.num_workers

    seed_all(seed)     

    # Model creation
    if model_name=='encoder_decoder':
        encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
        model = SimpleEncoderDecoder(encoder, decoder_channels=64, num_classes=2)  
    else:
        raise ValueError(f'Model {model} not recognized')
    #----

    # Persistent experiment data configuration
    experiment_path = Path(experiment_path)
    run_path = experiment_path/run_name
    if Path.exists(run_path):
        name = input('Run path already exists. Press enter to overwrite or write the name of the run: ')
        if name.strip():
            run_path = experiment_path/name
    Path.mkdir(run_path, parents=True, exist_ok=True)

    config_dict = vars(args)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    config_dict['timestamp'] = timestamp
    json.dump(config_dict, open(run_path/'config.json', 'w'), indent=2)
    #----

    # Dataset and DataLoaders creation
    if dataset_name=='oxford_pets':
        ds_train, ds_valid, class_weights = get_dataset(
            "/home/chcomin/Dropbox/ufscar/Visao Computacional/01-2024/Aulas/data/oxford_pets"
            , resize_size=resize_size
            )
    else:
        raise ValueError(f'Dataset {dataset_name} not recognized')
    #ds_train.indices = ds_train.indices[:5*bs_train]

    dl_train = DataLoader(
        ds_train, 
        batch_size=bs_train, 
        shuffle=True, 
        num_workers=num_workers,
        persistent_workers=num_workers>0,
        worker_init_fn=seed_worker,
        pin_memory='cuda' in device,
    )
    dl_valid = DataLoader(
        ds_valid,
        batch_size=bs_valid, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=num_workers, 
        persistent_workers=num_workers>0,
        pin_memory='cuda' in device,
    )
    #----

    # Training elements
    loss_func = nn.CrossEntropyLoss(torch.tensor(class_weights, device=device), ignore_index=2)
    optim = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,
                            momentum=0.9)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optim, num_epochs)
    #---

    perf_funcs = [
        MultipleMetrics(['acc', 'iou', 'prec', 'rec', 'dice'], confusion_matrix_metrics)
    ]
    
    
    logger = Logger()
    runner = ModuleRunner(
        model, 
        dl_train, 
        dl_valid, 
        loss_func, 
        optim, 
        scheduler,
        perf_funcs,
        logger,
        device
    )

    model.to(device)

    best_loss = torch.inf
    for epoch in range(0, num_epochs):
        runner.train_one_epoch(epoch)
        runner.validate_one_epoch(epoch)

        show_log(logger)

        checkpoint = {
            'model':model.state_dict(),
            'optim':optim.state_dict(),
            'sched':scheduler.state_dict(),
        }

        torch.save(checkpoint, run_path/'checkpoint.pt')
        loss_valid = logger.get_data()['valid/loss'].iloc[-1]
        if loss_valid<best_loss:
            torch.save(checkpoint, run_path/'best_model.pt')
            best_loss = loss_valid

    return runner

def get_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-name', help='Name of the dataset')
    parser.add_argument('--experiment-path', default='experiments/no_name_experiment', help='Path to save experiment data')
    parser.add_argument('--run-name', default='no_name_run', help='Name of the run for a given experiment')
    parser.add_argument('--model-name', default='encoder_decoder', help='Name of the model to train')
    parser.add_argument('--bs-train', type=int, default=32, help='Batch size used durig training')
    parser.add_argument('--bs-valid', type=int, default=8, help='Batch size used durig validation')
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0., help='Weight decay for the optimizer')
    parser.add_argument('--resize-size', type=int, default=256, help='Size to resize the images')
    parser.add_argument('--seed', type=int, default=0, help='Seed for the random number generator')
    parser.add_argument('--device', default='cuda:0', help='where to run the training code (e.g. "cpu" or "cuda:0")')
    parser.add_argument('--num-workers', type=int, default=5, help='Number of workers for the DataLoader')

    return parser

def get_args(arg_dict=None):
    """Parse command line arguments or arguments from a dictionary.

    Parameters
    ----------
    arg_dict, optional
        A dictionary containing the arguments to be parsed. 

    Returns
    -------
    args
        An argparse namespace object containing the parsed arguments
    """

    parser = get_parser()
    if arg_dict is None:
        args = parser.parse_args()
    else:
        vals = []
        for k, v in arg_dict.items():
            if k.startswith('--'):
                vals.extend([k, str(v)])
            else:
                vals.extend([str(v)])
        args = parser.parse_args(vals)

    return args

if __name__ == '__main__':
    train()