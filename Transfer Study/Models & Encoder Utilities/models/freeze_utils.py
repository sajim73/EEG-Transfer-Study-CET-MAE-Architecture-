from typing import Iterable


def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False
    return module


def unfreeze_module(module):
    for param in module.parameters():
        param.requires_grad = True
    return module


def freeze_except(module, trainable_prefixes: Iterable[str]):
    prefixes = tuple(trainable_prefixes)
    for name, param in module.named_parameters():
        param.requires_grad = name.startswith(prefixes)
    return module


def count_trainable_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def count_all_parameters(module):
    return sum(p.numel() for p in module.parameters())
