import json
import torch
import torch.nn as nn

from config.model_config import ModelArgs
from .normalization import RMSNorm
from .mamba_block import MambaBlock


class Mamba(nn.Module):
    def __init__(self, args: ModelArgs ,hid):
        """Full Mamba model."""
        super().__init__()
        self.args = args
        self.nl =args.n_layer

        self.embedding = nn.Linear(args.vocab_size, args.d_model)
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.layers2 = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])

        self.lin = nn.ModuleList([
             nn.Sequential(
                 nn.LayerNorm(args.seq_in),
                 nn.Linear(args.seq_in ,hid),
                 nn.ReLU(),
                 nn.Linear(hid ,args.seq_in))
         ] + [
             nn.Sequential(
                 RMSNorm(args.seq_in),
                 nn.Linear(args.seq_in ,hid),
                 nn.ReLU(),
                 nn.Linear(hid ,args.seq_in)
             ) for _ in range(args.n_layer -2)
         ] + [
             nn.Sequential(
                 RMSNorm(args.seq_in),
                 nn.Linear(args.seq_in ,hid),
                 nn.ReLU(),
                 nn.Linear(hid ,args.seq_in)
             )
         ])

        self.norm_f = nn.LayerNorm(args.d_model)
        self.lm_head = nn.Linear(args.d_model, args.vocab_size)
        self.proj = nn.Sequential(
            nn.Linear(args.seq_in ,hid),
            nn.ReLU(),
            nn.Linear(hid ,args.seq_in)
        )
        self.nnl =nn.LayerNorm(args.vocab_size)

    def forward(self, input_ids):
        """
        Args:
            input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)
        Returns:
            logits: shape (b, l, vocab_size)
        """

        x = self.embedding(input_ids)
        x1 =x
        x2 =x

        for i in range(self.nl):
            x1 = self.layers[i](x1)
            x2 = self.layers2[i](x2.flip([1]))
            x = x1 + x2.flip([1]) + x
            x = self.lin[i](x.permute(0,2,1)).permute(0,2,1) + x
            x1 = x
            x2 = x

        x = self.norm_f(x)
        logits = self.lm_head(x)

        return logits

    @staticmethod
    def from_pretrained(pretrained_model_name: str):
        """Load pretrained weights from HuggingFace into model."""
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file

        def load_config_hf(model_name):
            resolved_archive_file = cached_file(model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
            return json.load(open(resolved_archive_file))


        def load_state_dict_hf(model_name, device=None, dtype=None):
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
            return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)

        config_data = load_config_hf(pretrained_model_name)
        args = ModelArgs(
            d_model=config_data['d_model'],
            n_layer=config_data['n_layer'],
            vocab_size=config_data['vocab_size'],
            seq_in=config_data['seq_in'],
            seq_out=config_data['seq_out'],
        )
        model = Mamba(args)

        state_dict = load_state_dict_hf(pretrained_model_name)
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = state_dict[key]
        model.load_state_dict(new_state_dict)

        return model


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args)
        self.norm = nn.LayerNorm(args.d_model)


    def forward(self, x):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
        Returns:
            output: shape (b, l, d)
        """
        output = self.mixer(self.norm(x))
        return output
