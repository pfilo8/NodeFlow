# Neural Oblivious Decision Ensembles
# Author: Sergey Popov, Julian Qian
# https://github.com/Qwicen/node
# For license information, see https://github.com/Qwicen/node/blob/master/LICENSE.md
"""Dense ODST Block"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .odst import ODST
from .activations import sparsemax, sparsemoid


class DenseODSTBlock(nn.Sequential):
    def __init__(
            self,
            input_dim,
            num_trees,
            depth=6,
            tree_output_dim=1,
            choice_function=sparsemax,
            bin_function=sparsemoid,
            initialize_response_=nn.init.normal_,
            initialize_selection_logits_=nn.init.uniform_,
            threshold_init_beta=1.0,
            threshold_init_cutoff=1.0,
            num_layers=6,
            max_features=None,
            input_dropout=0.0,
            flatten_output=False,
    ):
        layers = []
        for i in range(num_layers):
            oddt = ODST(
                input_dim,
                num_trees,
                depth=depth,
                tree_output_dim=tree_output_dim,
                flatten_output=True,
                choice_function=choice_function,
                bin_function=bin_function,
                initialize_response_=initialize_response_,
                initialize_selection_logits_=initialize_selection_logits_,
                threshold_init_beta=threshold_init_beta,
                threshold_init_cutoff=threshold_init_cutoff,
            )

            input_dim = min(input_dim + num_trees * tree_output_dim, max_features or float("inf"))
            layers.append(oddt)

        super().__init__(*layers)
        self.num_layers, self.layer_dim, self.tree_dim = (
            num_layers,
            num_trees,
            tree_output_dim,
        )
        self.input_dim = input_dim
        self.max_features, self.flatten_output = max_features, flatten_output
        self.input_dropout = input_dropout

    def forward(self, x):
        initial_features = x.shape[-1]
        for layer in self:
            layer_inp = x
            if self.max_features is not None:
                tail_features = min(self.max_features, layer_inp.shape[-1]) - initial_features
                if tail_features != 0:
                    layer_inp = torch.cat(
                        [
                            layer_inp[..., :initial_features],
                            layer_inp[..., -tail_features:],
                        ],
                        dim=-1,
                    )
            if self.training and self.input_dropout:
                layer_inp = F.dropout(layer_inp, self.input_dropout)
            h = layer(layer_inp)
            x = torch.cat([x, h], dim=-1)

        outputs = x[..., initial_features:]
        # outputs = x
        if not self.flatten_output:
            outputs = outputs.view(*outputs.shape[:-1], self.num_layers * self.layer_dim, self.tree_dim) # self.input_dim + 
        return outputs
