import torch.nn as nn
import numpy as np
import torch
import os


def float_to_floatq(tensor: torch.Tensor, exponent: int, name: str, bit_width: int) -> torch.Tensor:
    """
    convert a tensor from a float32 dtype into a signed int type of width bit_width
    convert it back to float32 for simplicity
    tensor: the tensor to quantize

    exponent: the tensor will be multiplied by 2**exponent
    bit_width: number of bit for signed intger type
    """
    min_int = -2**(bit_width-1)
    max_int = 2**(bit_width-1)-1
    tensor_int = torch.clip(torch.round(
        tensor * 2**exponent), min_int, max_int)
    tensor_floatq = tensor_int / (2**exponent)

    error = torch.mean(tensor - tensor_floatq).numpy()
    print(f"{error} on layer {name} ")
    return tensor_floatq


def find_exponent1(layer: nn.Module, weight: torch.Tensor, bit_width: int) -> int:
    """
    simple function to get a good exponent to represent the weights on 2**bit_width values

    Args:
        layer (torch.Module): pytorch layer's module
        weight (torch.Tensor): weight tensor
        bit_width (int): bit width for the signed integer representation

    Returns:
        int: exponent that minimize weight distance
    """
    return 4


def find_exponent2(layer: nn.Module, weight: torch.Tensor, bit_width: int) -> int:
    """
    find the exponent that minimize the distance(weights, weight_floatq)

    Args:
        layer (torch.Module): pytorch layer's module
        weight (torch.Tensor): weight tensor
        bit_width (int): bit width for the signed integer representation

    Returns:
        int: exponent that minimize weight distance
    """
    raise NotImplementedError()


def find_exponent3(layer: nn.Module, weight: torch.Tensor, bit_width: int) -> int:
    """
    Find exponent using layer's bottom and top data
    Args:
        layer (torch.Module): pytorch layer's module
        weight (torch.Tensor): weight tensor
        bit_width (int): bit width for the signed integer representation

    Returns:
        int: exponent
    """
    save_dir = './forward_data_float/'
    input_ref = torch.tensor(
        np.load(os.path.join(save_dir, layer.name + "_in.npy")))
    output_ref = torch.tensor(
        np.load(os.path.join(save_dir, layer.name + "_out.npy")))
    output_run = layer(input_ref)

    raise NotImplementedError()


def compress_model(net: nn.Module):
    """compress models by quantizing convolution's weight

    Args:
        net (nn.Module): network
    """

    bit_width = 8

    for name, layer in net.named_modules():
        if isinstance(layer, nn.Conv2d):
            layer.name = name

            exponent = find_exponent1(layer, layer.weight, bit_width)
            new_weights = float_to_floatq(
                layer.weight, exponent, layer.name, bit_width)
            layer.weight = nn.Parameter(new_weights)
