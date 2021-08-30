import copy
import random
import numpy as np
from functools import wraps

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


class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer=-2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = MLP(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x, return_embedding=False):
        representation = self.get_representation(x)

        if return_embedding:
            return representation

        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection, representation


class GL(nn.Module):
    def __init__(self,
                 net,
                 image_size,
                 hidden_layer=-2,
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
                                         layer=hidden_layer)

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

        online_proj_one, _ = self.online_encoder(image_one)
        online_proj_two, _ = self.online_encoder(image_two)

        online_proj_one = F.normalize(online_proj_one, dim=-1, p=2)
        online_proj_two = F.normalize(online_proj_two, dim=-1, p=2)

        # 2*b,demb_0, emb1 = embeddings[:bs], embeddings[bs:]
        sims = torch.cat([
            cos_sim(emb.unsqueeze(0), online_proj_one).unsqueeze(0)
            for emb in online_proj_one
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
        return loss


if __name__ == '__main__':
    import sys
    sys.path.append('/raid/hym/code/Superpixel_CL/')
    from net.resnet import resnet50
    resnet = resnet50()
    learner = GL(resnet, image_size=128, hidden_layer='avgpool')
    x = torch.rand((32, 3, 128, 128))
    loss = learner(x)
    print(loss)