import copy
import random
import numpy as np
from functools import wraps, partial

import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms as T


def default(val, def_val):
    return def_val if val is None else val


def flatten(t):
    return t.reshape(t.shape[0], -1)


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance

        return wrapper

    return inner_fn


def get_module_device(module):
    return next(module.parameters()).device


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


# loss fn


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


# augmentation utils


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size=4096):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_size),
                                 nn.BatchNorm1d(hidden_size),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_size, projection_size))

    def forward(self, x):
        return self.net(x)


class ConvMLP(nn.Module):
    def __init__(self, chan, chan_out=256, inner_dim=2048):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(chan, inner_dim, 1),
                                 nn.BatchNorm2d(inner_dim), nn.ReLU(),
                                 nn.Conv2d(inner_dim, chan_out, 1))

    def forward(self, x):
        return self.net(x)


class NetWrapper(nn.Module):
    def __init__(self,
                 net,
                 projection_size,
                 projection_hidden_size,
                 layer_pixel=-2,
                 layer_instance=-2):
        super().__init__()
        self.net = net
        self.layer_pixel = layer_pixel
        self.layer_instance = layer_instance

        self.pixel_projector = None
        self.instance_projector = None

        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.hidden_pixel = None
        self.hidden_instance = None
        self.hook_registered = False

    def _find_layer(self, layer_id):
        if type(layer_id) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(layer_id, None)
        elif type(layer_id) == int:
            children = [*self.net.children()]
            return children[layer_id]
        return None

    def _hook(self, attr_name, _, __, output):
        setattr(self, attr_name, output)

    def _register_hook(self):
        pixel_layer = self._find_layer(self.layer_pixel)
        instance_layer = self._find_layer(self.layer_instance)

        assert pixel_layer is not None, f'hidden layer ({self.layer_pixel}) not found'
        assert instance_layer is not None, f'hidden layer ({self.layer_instance}) not found'

        pixel_layer.register_forward_hook(partial(self._hook, 'hidden_pixel'))
        instance_layer.register_forward_hook(
            partial(self._hook, 'hidden_instance'))
        self.hook_registered = True

    @singleton('pixel_projector')
    def _get_pixel_projector(self, hidden):
        _, dim, *_ = hidden.shape
        projector = ConvMLP(dim, self.projection_size,
                            self.projection_hidden_size)
        return projector.to(hidden)

    @singleton('instance_projector')
    def _get_instance_projector(self, hidden):
        _, dim = hidden.shape
        projector = MLP(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    def get_representation(self, x):
        if not self.hook_registered:
            self._register_hook()

        _ = self.net(x)
        hidden_pixel = self.hidden_pixel
        hidden_instance = self.hidden_instance
        self.hidden_pixel = None
        self.hidden_instance = None
        assert hidden_pixel is not None, f'hidden pixel layer {self.layer_pixel} never emitted an output'
        assert hidden_instance is not None, f'hidden instance layer {self.layer_instance} never emitted an output'
        return hidden_pixel, hidden_instance

    def forward(self, x, return_embedding=False):
        pixel_representation, instance_representation = self.get_representation(
            x)
        if return_embedding:
            return pixel_representation, instance_representation
        instance_representation = instance_representation.flatten(1)

        pixel_projector = self._get_pixel_projector(pixel_representation)
        instance_projector = self._get_instance_projector(
            instance_representation)

        pixel_projection = pixel_projector(pixel_representation)
        instance_projection = instance_projector(instance_representation)
        return pixel_projection, instance_projection


class GL(nn.Module):
    def __init__(self,
                 net,
                 image_size,
                 projection_size=256,
                 projection_hidden_size=4096,
                 augment_fn=None,
                 augment_fn2=None,
                 temperature=0.1):
        super().__init__()
        self.net = net
        self.temperature = temperature

        # default SimCLR augmentation

        DEFAULT_AUG = torch.nn.Sequential(
            RandomApply(T.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.3),
            T.RandomGrayscale(p=0.2),
            RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)), p=0.2),
            T.RandomResizedCrop((image_size, image_size)),
        )

        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, self.augment1)

        self.online_encoder = NetWrapper(net,
                                         projection_size,
                                         projection_hidden_size,
                                         layer_pixel=-3)

        device = get_module_device(net)
        self.to(device)
        self.temperature = 0.1
        # s end a mock image tensor to instantiate singleton parameters
        # self.forward(torch.randn(32, 3, image_size, image_size, device=device))

    # x : 4 patients and 8 partions; each partion provides a slice
    def forward(self, x, return_embedding=False):
        cos_sim = nn.CosineSimilarity()

        n_vols, n_parts = 4, 8
        assert n_vols * n_parts == x.size(0)
        batch_size = x.size(0)

        if return_embedding:
            return self.online_encoder(x, True)

        image_one, image_two = self.augment1(x), self.augment2(x)

        proj_pixel_one, proj_instance_one = self.online_encoder(image_one)
        proj_pixel_two, proj_instance_two = self.online_encoder(image_two)
        b, c, w, h = proj_pixel_one.size()

        # global
        proj_instance_one = F.normalize(proj_instance_one, dim=-1, p=2)
        proj_instance_two = F.normalize(proj_instance_two, dim=-1, p=2)

        sims = torch.cat([
            cos_sim(emb.unsqueeze(0), proj_instance_two).unsqueeze(0)
            for emb in proj_instance_one
        ]) / self.temperature

        loss = 0
        for i in range(0, batch_size):
            pos_index = [i]
            pos_index = list(
                set(pos_index + [(i % n_parts) + volume * n_parts
                                 for volume in range(n_vols)]))
            neg_index = np.delete(np.arange(batch_size), pos_index)
            _loss = 0
            for pos_n in pos_index:
                _loss -= torch.log(
                    torch.exp(sims[i, pos_n]) /
                    (torch.exp(sims[i, pos_n]) +
                     torch.sum(torch.exp(sims[i][neg_index]))))
                _loss -= torch.log(
                    torch.exp(sims[pos_n, i]) /
                    (torch.exp(sims[pos_n, i]) +
                     torch.sum(torch.exp(sims[:, i][neg_index]))))
            _loss /= len(pos_index)
            loss += _loss
        loss /= batch_size * 2

        # local
        # each image provides five pixels
        mask = torch.zeros((w, h))
        mask[0, 0] = 1
        mask[0, -1] = 1
        mask[-1, 0] = 1
        mask[-1, -1] = 1
        mask[w // 2, h // 2] = 1

        proj_pixel_one = proj_pixel_one.masked_select(mask[None, None, ...].to(
            torch.bool)).view(b, c, -1).permute(0, 2, 1).reshape(b * 5, -1)
        proj_pixel_two = proj_pixel_two.masked_select(mask[None, None, ...].to(
            torch.bool)).view(b, c, -1).permute(0, 2, 1).reshape(b * 5, -1)

        sims = torch.cat([
            cos_sim(emb.unsqueeze(0), proj_pixel_two).unsqueeze(0)
            for emb in proj_pixel_one
        ]) / self.temperature

        loss1 = 0
        for i in range(0, batch_size * 5):
            local_pos = i % 5
            part = (i % (5 * n_parts) - local_pos) // 5
            pos_index = [i]
            pos_index = list(
                set(pos_index + [
                    local_pos + part * 5 + volume * n_parts * 5
                    for volume in range(n_vols)
                ]))
            neg_index = np.delete(np.arange(batch_size * 5), pos_index)
            pos_index = [i]
            _loss = 0
            for pos_n in pos_index:
                _loss -= torch.log(
                    torch.exp(sims[i, pos_n]) /
                    (torch.exp(sims[i, pos_n]) +
                     torch.sum(torch.exp(sims[i][neg_index]))))
                _loss -= torch.log(
                    torch.exp(sims[pos_n, i]) /
                    (torch.exp(sims[pos_n, i]) +
                     torch.sum(torch.exp(sims[:, i][neg_index]))))
            _loss /= len(pos_index)
            loss1 += _loss
        loss1 /= batch_size * 2 * 5

        return loss + loss1


if __name__ == '__main__':
    import sys, os, torch
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append('/raid/hym/code/Superpixel_CL')
    from net.resnet import resnet50
    resnet = resnet50()
    learner = GL(resnet, image_size=128)

    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)
    x = torch.rand((32, 3, 128, 128))
    loss = learner(x)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(loss)