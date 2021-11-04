'''
Utilities for working with PIL, tensor, numpy and imgaug images
'''

from PIL import Image
import torch
import random
import matplotlib.pyplot as plt
from IPython.display import display
from ipywidgets import interact, fixed, IntSlider
from .transforms import pil_to_tensor

def pil_img_info(img, print_repr=False):
    """Return the following information about a PIL image:
    The color mode (RGB, L, F, etc)
    Width
    Height
    Number of channels
    Intensity range (min, max)
    Additional info such as compression...

    Parameters
    ----------
    img : PIL.Image
        PIL image
    print_repr : bool
        If False, only returns a string with the image information. If True,
        also prints the information

    Returns
    -------
    info_str : string
        Information about the image
    """

    if isinstance(img, Image.Image):
        info_str = f'''
        Image information:
        Mode:{img.mode}
        Width:{img.width}
        Height:{img.height}
        Num channels:{len(img.getbands())}
        Intensity range: {img.getextrema()}
        Additional info: {img.info}
        '''
    else:
        info_str = 'Not a PIL image'

    if print_repr:
        print(info_str)

    return info_str

def tensor_info(tensor, print_repr=False):
    """Return the following information about a torch tensor:
    Shape
    Type

    Parameters
    ----------
    tensor : torch.Tensor
        Torch tensor
    print_repr : bool
        If False, only returns a string with the tensor information. If True,
        also prints the information

    Returns
    -------
    info_str : string
        Information about the tensor
    """

    if isinstance(tensor, torch.Tensor):
        info_str = f'''
        Tensor information:
        Shape:{tensor.shape}
        Type:{tensor.dtype}
        '''
    else:
        info_str = 'Not a tensor'

    if print_repr:
        print(info_str)

    return info_str

def show(pil_img, binary=False):
    """Show PIL image in a Jupyter notebook

    Parameters
    ----------
    pil_img : PIL.Image
        PIL image
    binary : bool
        If True, the image should be treated as binary. That is, the range
        [0, 1] is shown as [0, 255]
    """

    if binary:
        palette = [  0,     0,   0,    # RGB value for color 0
                   255,   255, 255]    # RGB value for color 1
        pil_img = pil_img.copy()
        pil_img.putpalette(palette)

    display(pil_img)

def pil_img_opener(img_file_path, channel=None, convert_gray=False, is_label=False, return_tensor=False, print_info=False):
    """Open a PIL image

    Parameters
    ----------
    img_file_path : string
        Path to the image
    channel : int
        Image channel to return. If None, returns all channels
    convert_gray : bool
        If True, image is converted to grayscale with single channel
    is_label : bool
        If True, image is treated as binary and intensities are coded as class indices.
        For instance, if the image contains the intensity values {0, 255}, they will be onverted
        to {0, 1}.
    return_tensor : bool
        If True, the image is converted to a Pytorch tensor.
    print_info :  bool
        If True, image information is printed when opening the image.

    Returns
    -------
    img : PIL.Image
        The PIL image
    """

    img = Image.open(img_file_path)
    if print_info: print(pil_img_info(img))

    if channel is not None: img = img.getchannel(channel)
    if convert_gray: img = img.convert('L')
    if is_label:
        # Map intensity values to indices 0, 1, 2,...
        colors = [t[1] for t in img.getcolors()]
        lut = [0]*256
        for i, c in enumerate(colors):
            lut[c]=i
        img = img.point(lut)

    if return_tensor:
        img = pil_to_tensor(img)

    return img


def get_shape(img, warn=True):
    """Get the shape of a ndarray, PIL or tensor image. Works for 2D and 3D images. Warning, for 
    three-dimensional arrays, the function guesses if the image is 2D with colors or 3D grayscale.
    The following is assumed:

    -For tensors:
        - If img[-3]<=4, the image is 2D with colors
        - If img[-3]>4, the image is 3D
    -For numpy arrays:
        - If img[-1]<=4, the image is 2D with colors
        - If img[-1]>4, the image is 3D
    """

    if isinstance(img, Image.Image):
        img_shape = (img.height, img.width)
    elif isinstance(img, torch.Tensor):
        img_shape = img.shape
        if (img.ndim==3):
            if img_shape[-3]<=4:
                # Consider that third to last dimension is for color
                img_shape = img_shape[-2:]
            else:
                img_shape = img_shape[-3:]
        if (img.ndim==4):
            img_shape = img_shape[-3:]
    elif isinstance(img, np.ndarray):
        img_shape = img.shape
        if img.ndim==3:
            if img_shape[-1]<=4:
                # Consider that last dimension is for color
                img_shape = img_shape[-3:-1]
            else:
                img_shape = img_shape[-3:]
        elif img.ndim==4:
            img_shape = img_shape[-3:]
    else:
        raise AttributeError("Image is not a PIL, Tensor or ndarray. Cannot safely infer shape")

    if min(img_shape)<=4:
        print(f'Warning, inferred shape {img_shape} is probably incorrect. Sizes smaller than 5 are being discarded')
        img_shape = filter(lambda v:v>4, img_shape)

    return img_shape

class PerfVisualizer:
    """Class for visualizing classification results in increasing order of the values returned by `perf_func`"""

    def __init__(self, dataset, model, perf_func, model_pred_func=None, device=None):

        if model_pred_func is None:
            model_pred_func = self.pred

        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

        self.dataset = dataset
        self.model = model
        self.perf_func = perf_func
        self.device = device
        self.model_pred_func = model_pred_func

        self._performance_per_image()

    def _performance_per_image(self, label_thresh=10):

        self.model.eval()
        self.model.to(self.device)

        print_interv_perc = 5   # In percent

        num_samples = len(self.dataset)
        print_interv = max([round(print_interv_perc*num_samples/100), 1])

        perf_dict = {}
        #print("allocated, allocated_bytes, segment, reserved_bytes, active, active_bytes")
        for idx, (img, label) in enumerate(self.dataset):
            if label.ndim>1 and label.sum()>label_thresh:
                predb_acc = self.pred(img.unsqueeze(0), label.unsqueeze(0))
                perf_dict[self.dataset.img_file_paths[idx].stem] = {'idx':idx, 'perf':predb_acc.item()}

            perc = round(100*idx/num_samples)
            if idx%print_interv==0 or idx==num_samples-1:
                print(f'Evaluating images...{100*(idx+1)/num_samples:1.0f}%', end='\r')
                '''memstats = torch.cuda.memory_stats(device)
                print(memstats["allocation.all.peak"],
                memstats["allocated_bytes.all.peak"],
                memstats["segment.all.peak"],
                memstats["reserved_bytes.all.peak"],
                memstats["active.all.peak"],
                memstats["active_bytes.all.peak"])'''

        print(''*30, end='\r')

        perf_list = sorted(list(perf_dict.items()), key=lambda x:x[1]['perf'])
        self.perf_list = perf_list

        return perf_list

    def pred(self, xb, yb, return_classes=False):

        with torch.no_grad():
            xb = xb.to(self.device, torch.float32)
            predb = self.model(xb).to('cpu')

            predb = predb.cpu()
            predb_acc = self.perf_func(predb, yb)

            if return_classes:
                classes_predb = torch.argmax(predb, dim=1).to(torch.uint8)
                return predb_acc, classes_predb
            else:
                return predb_acc

    def plot_performance(self):

        perf_vals = [elem[1]['perf'] for elem in self.perf_list]

        plt.figure(figsize=[15,15])
        plt.plot(perf_vals, '-o')
        plt.ylim(0, 1)

    def plot_samples(self, num_samples=5, which='worst', show_original=True):
        """which must be {worst, top, random}."""
           
        perf_list = self.perf_list
        if which=='worst':
            samples_to_plot = perf_list[0:num_samples]
        elif which=='best':
            samples_to_plot = perf_list[-1:num_samples:-1]
        elif which=='random':
            samples_to_plot = random.sample(perf_list, num_samples)

        plt.figure(figsize=[15, num_samples*6])
        for idx in range(num_samples):
            file, perf = samples_to_plot[idx]
            img, label, *_ = self.dataset.get_item(perf['idx'])
            img_transf, label_transf, *_ = self.dataset[perf['idx']]
            _, bin_pred = self.pred(img_transf.unsqueeze(0), label_transf.unsqueeze(0), return_classes=True)
            bin_pred = bin_pred[0]

            if show_original:
                img_show = img
                label_show = label
            else:
                img_show = img_transf
                label_show = label_transf

            if img_transf.ndim==3:
                img_show = img_show.permute(1, 2, 0)

            plt.subplot(num_samples, 3, 3*idx+1)
            plt.imshow(img_show, 'gray')
            plt.title(file)

            plt.subplot(num_samples, 3, 3*idx+2)
            plt.imshow(label_show, 'gray')

            plt.subplot(num_samples, 3, 3*idx+3)
            plt.imshow(bin_pred, 'gray')
            plt.title(perf['perf'])

class InteractiveVisualizer:
    """Copied from cortex notebook"""
    
    def __init__(self, dataset, model, perf_list=None, device=None):
        
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')    
        
        model.to(device)
        model.eval()
        
        self.dataset = dataset
        self.model = model
        self.perf_list = perf_list
        self.device = device
        
        self.init_plot()
        
        interact(self.display_item, idx=IntSlider(min=0, max=len(perf_list)-1, step=1, continuous_update=False))
        
    def init_plot(self):
        
        plt.figure(figsize=[8,8])
        axs = []
        ims = []
        ax = plt.subplot(2, 2, 1)
        im = ax.imshow(torch.zeros(100, 100), 'gray', vmin=0, vmax=255)
        ax.axis('off')
        axs.append(ax)
        ims.append(im)
            
        ax = plt.subplot(2, 2, 2)
        im = ax.imshow(torch.zeros((100, 100)), cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        axs.append(ax)
        ims.append(im)
        
        ax = plt.subplot(2, 2, 3)
        im = ax.imshow(torch.zeros((100, 100, 3)))
        ax.axis('off')
        axs.append(ax)
        ims.append(im)
        
        ax = plt.subplot(2, 2, 4)
        im = ax.imshow(torch.zeros((100, 100, 3)), cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        axs.append(ax)
        ims.append(im)
            
        self.axs = axs
        self.ims = ims
    
    def display_item(self, idx):

        ims = self.ims
        axs = self.axs
        
        if self.perf_list is None:
            img_idx = idx
        else:
            item = self.perf_list[idx]
            img_name, data  = item
            img_idx, perf = data['idx'], data['perf']
            axs[0].set_title(img_name)
            axs[3].set_title(f'{perf:.2f}')
            
        xb_or, yb_or, *_ = self.dataset.get_item(img_idx)
        xb_aug, yb_aug, *_ = self.dataset[img_idx]
        predb = self.model(xb_aug.unsqueeze(0).to(self.device))
        bin_pred = torch.argmax(predb, dim=1)
        
        ims[0].set_data(xb_or)
        ims[1].set_data(yb_or)
        ims[2].set_data(xb_aug.unsqueeze(3).transpose(0, 3).squeeze())
        ims[3].set_data(bin_pred.squeeze().cpu())
        
        axs[1].set_title('Target')
        axs[2].set_title('Augmented image')
        axs[0].figure.canvas.draw()
