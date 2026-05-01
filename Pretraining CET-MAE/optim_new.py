import torch


def build_optimizer(args, model, mode="cet-mae"):
    lr = args.get("cet_mae_lr", 1e-4)
    weight_decay = args.get("weight_decay", 5e-7)
    betas = tuple(args.get("betas", (0.9, 0.999)))

    if mode == "cet-mae":
        betas = tuple(args.get("cet_mae_betas", (0.95, 0.999)))
        lr = args.get("cet_mae_lr", lr)
        weight_decay = args.get("cet_mae_weight_decay", weight_decay)

    params = [p for p in model.parameters() if p.requires_grad]

    if len(params) == 0:
        raise RuntimeError("No trainable parameters found for optimizer.")

    optimizer_name = str(args.get("optimizer", "AdamW")).lower()

    if optimizer_name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        momentum = args.get("momentum", 0.9)
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
