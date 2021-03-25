'''
Utilities for working with PIL, tensor, numpy and imgaug images
'''

from PIL import Image
import torch
from IPython.display import display
import matplotlib.pyplot as plt

def pil_img_info(img, print_repr=False):
    '''Returns the following information about a PIL image:
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
    '''

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
    '''Returns the following information about a torch tensor:
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
    '''

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
    '''Show PIL image in a Jupyter notebook

    Parameters
    ----------
    pil_img : PIL.Image
        PIL image
    binary : bool
        If True, the image should be treated as binary. That is, the range
        [0, 1] is shown as [0, 255]

    Returns
    -------
    None
    '''

    if binary:
        palette = [  0,     0,   0,    # RGB value for color 0
                   255,   255, 255]    # RGB value for color 1
        pil_img = pil_img.copy()
        pil_img.putpalette(palette)

    display(pil_img)

def pil_img_opener(img_file_path, channel=None, convert_gray=False, is_label=False, print_info=False):
    '''Opens a PIL image

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
    print_info :  bool
        If True, image information is printed when opening the image.

    Returns
    -------
    img : PIL.Image
        The PIL image
    '''

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

    return img

class PerfVisualizer:
    '''Class for visualizing classification results in increasing order of the values returned by `perf_func`'''

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

        self.performance_per_image()
        self.plot_samples()


    def performance_per_image(self, label_thresh=10):

        self.model.eval()
        self.model.to(self.device)

        num_samples = len(self.dataset)
        perf_dict = {}
        #print("allocated, allocated_bytes, segment, reserved_bytes, active, active_bytes")
        for idx, (img, label) in enumerate(self.dataset):
            if label.sum()>label_thresh:
                predb_acc = self.pred(img.unsqueeze(0), label.unsqueeze(0))
                perf_dict[self.dataset.img_file_paths[idx].stem] = {'idx':idx, 'perf':predb_acc}

            if idx%100==0:
                print(f'Evaluating images...{100*idx/num_samples:1.0f}%', end='\r')
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

    def pred(self, xb, yb):

        with torch.no_grad():
            xb = xb.to(self.device, torch.float32)
            predb = self.model(xb)

            predb = predb.cpu()
            predb_acc = self.perf_func(predb, yb)

        return predb_acc

    def plot_samples(self):

        perf_vals = [elem[1]['perf'] for elem in self.perf_list]

        plt.figure(figsize=[15,15])
        plt.plot(perf_vals, '-o')
        plt.ylim(0, 1)