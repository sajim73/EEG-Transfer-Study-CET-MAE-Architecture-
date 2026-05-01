import torch
import torch.nn as nn


class cscl_model_cet_mae(nn.Module):
    """
    Compatibility stub.

    The current pre_train_eval_cet_mae_later_project_7575.py imports this symbol,
    but in the provided script path it is not actually used during training.
    This class is included to satisfy imports and preserve compatibility with
    earlier CET-MAE / EEG-To-Text code organization.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dummy = nn.Identity()

    def forward(self, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            return self.dummy(args[0])
        return None
