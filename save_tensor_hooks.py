import os
import torch.nn as nn
import numpy as np


save_dir = './forward_data_float/'


def save_data_forward_hook(module, input, output):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path_in = os.path.join(save_dir, module.name+'_in.npy')
    path_out = os.path.join(save_dir, module.name+'_out.npy')

    if not os.path.exists(path_in):
        assert len(input) == 1
        print(f'Saving {path_in}')
        np.save(path_in, input[0].numpy())

        print(f'Saving {path_out}')
        np.save(path_out, output.numpy())


def save_tensors_hooks(net):
    for name, layer in net.named_modules():
        if isinstance(layer, nn.Conv2d):

            layer.name = name
            layer.register_forward_hook(save_data_forward_hook)

    return net
