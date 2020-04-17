'''
Image transformation routines
'''

import imgaug as ia
import numpy as np
import torch
from torchvision import transforms as torch_transforms
import cv2

def to_tensor(img, is_label=False):
    '''Transform PIL.Image or numpy array to a tensor. Most of the
    function was taken from torchvision.transforms.functional.to_tensor.

    Note 1: Intensities are reescaled to [0, 1] if is_label is False and `img`
    is of type uint8. Use is_label=True if you do not want to reescale image
    (even if it is not a label).

    Note 2: The image is expected to have size (height, width) or (height, width, channels).
    Returns image with size (channels, height, width).

    Parameters
    ----------
    img : PIL.Image or np.ndarray
        The input image
    is_label : bool
        If True, images is not reescaled

    Returns
    -------
    img : torch.Tensor
        The output image as a tensor
    '''

    if isinstance(img, np.ndarray):
        # handle numpy array
        if img.ndim == 2:
            img = img[:, :, None]

        img = torch.from_numpy(img.transpose((2, 0, 1)))
        # backward compatibility
        if isinstance(img, torch.ByteTensor) and not is_label:
            return img.float().div(255)
        else:
            return img

    # handle PIL Image
    if img.mode == 'I':
        img = torch.from_numpy(np.array(img, np.int32, copy=False))
    elif img.mode == 'I;16':
        img = torch.from_numpy(np.array(img, np.int16, copy=False))
    elif img.mode == 'F':
        img = torch.from_numpy(np.array(img, np.float32, copy=False))
    elif img.mode == '1':
        img = 255 * torch.from_numpy(np.array(img, np.uint8, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    # PIL image mode: L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK
    if img.mode == 'YCbCr':
        nchannel = 3
    elif img.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(img.mode)
    img = img.view(img.size[1], img.size[0], nchannel)
    # put it from HWC to CHW format
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor) and not is_label:
        return img.float().div(255)
    else:
        return img

def tensor_2_pil(tensor):
    '''Transform torch.Tensor to a PIL image.

    Note: Intensities are reescaled to [0, 255] if tensor is of type float. Values
    are not changed if the tensor is of type int.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor

    Returns
    -------
    PIL.Image
        The output PIL image
    '''

    return torch_transforms.ToPILImage()(tensor)


def pil_to_imgaug(img, label=None, weight=None):
    '''Transform PIL images to corresponding objects of the imgaug (image augmentation)
    module.

    Parameters
    ----------
    img : PIL.Image
        PIL image, only tested for uint8 type
    label : PIL.Image
        Label, must contain integer values. Converted to a imgaug.SegmentationMapsOnImage object
    weight : PIL.Image
        Weight, converted to a imgaug.HeatmapsOnImage object

    Returns
    -------
    ret_vals : list
        List containing imgaug images corresponding to the `img`, `label` and `weight` inputs
    '''

    img_shape = img.size[::-1]
    ret_vals = [np.array(img)]
    if label is not None:
        segmap = ia.SegmentationMapsOnImage(np.array(label), img_shape)
        ret_vals.append(segmap)
    if weight is not None:
        heatmap = ia.HeatmapsOnImage(np.array(weight), img_shape)
        ret_vals.append(heatmap)

    return ret_vals

def imgaug_to_tensor(img=None, label=None, weight=None):
    '''Transform imgaug (image augmentation) images to torch tensors.

    Parameters
    ----------
    img : np.ndarray
        Image to be transformed
    label : imageaug.SegmentationMapsOnImage
        Label image
    weight : imageaug.HeatmapsOnImage
        Weight image

    Returns
    -------
    ret_vals : list
        torch tensors corresponding to the `img`, `label` and `weight` inputs
    '''

    ret_vals = []
    if img is not None:
        ret_vals.append(to_tensor(img))
    if label is not None:
        ret_vals.append(to_tensor(label.get_arr(), True))
    if weight is not None:
        ret_vals.append(to_tensor(weight.get_arr()))

    return ret_vals

def translate_imagaug_seq(imgaug_seq):
    '''Closure for translating arguments 'image', 'segmentation_maps' and 'heatmaps' of
    imgaug functions to 'img', 'label' and 'weight'

    Parameters
    ----------
    imgaug_seq : function or class
        imgaug function or class to be translated. Usually, it is an imgaug.Sequential
        class

    Returns
    -------
    transf_imgaug_seq : function
        New function with translated arguments
    '''

    def transf_imgaug_seq(img, label=None, weight=None, **kwargs):
        return imgaug_seq(image=img, segmentation_maps=label, heatmaps=weight, **kwargs)

    return transf_imgaug_seq

def pil_to_imgaug_to_tensor(imgaug_seq):
    '''Utility function for generating a typical transfromation pipeline:
    pil image -> imgaug -> tensor

    Parameters
    ----------
    imgaug_seq : function or class
        imgaug function or class to be translated. Usually, it is an imgaug.Sequential
        class

    Returns
    -------
    transform_funcs : list
        List of transformations
    '''

    transf_pil_to_imgaug = pil_to_imgaug
    transf_imgaug = translate_imagaug_seq(imgaug_seq)
    transf_imgaug_to_tensor = imgaug_to_tensor
    transform_funcs = [transf_pil_to_imgaug, transf_imgaug, transf_imgaug_to_tensor]

    return transform_funcs

def clahe(pil_img, clip_limit=2.0, tile_shape=(8, 8)):
    '''Contrast Limited Adaptive Histogram Equalization.

    Parameters
    ----------
    pil_img : PIL.Image
        Image to be transformed
    clip_limit : float
        Threshold value for contrast limiting
    tile_shape : 2-tuple
        Tuple setting the tile size for the method

    Returns
    -------
    out_pil_img : PIL.Image
        Output image
    '''

    np_img = np.array(pil_img).astype(np.uint8)
    enhancer = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_shape)
    if np_img.ndim == 2:
        np_img = enhancer.apply(np_img)
    elif np_img.ndim == 3:
        np_img[:, :, 0] = enhancer.apply(np_img[:, :, 0])
        np_img[:, :, 1] = enhancer.apply(np_img[:, :, 1])
        np_img[:, :, 2] = enhancer.apply(np_img[:, :, 2])

    out_pil_img = Image.fromarray(np_img)
    return out_pil_img