# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention, MultimodelMultiheadAttention
from torch import Tensor
import math


class HighWayNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.dropout = args.attention_dropout

        for i in range(2):
            setattr(self, 'highway_linear{}'.format(i),
                    nn.Sequential(nn.Linear(args.encoder_embed_dim * 2, args.encoder_embed_dim * 2),
                                  nn.ReLU()))
            setattr(self, 'highway_gate{}'.format(i),
                    nn.Sequential(nn.Linear(args.encoder_embed_dim * 2, args.encoder_embed_dim * 2),
                                  nn.Sigmoid()))
        self.highway_linear = nn.Linear(args.encoder_embed_dim * 2, args.encoder_embed_dim)

    def forward(self, x, x1):

        x = torch.cat([x, x1], dim=-1)

        for i in range(2):
            h = getattr(self, 'highway_linear{}'.format(i))(x)
            g = getattr(self, 'highway_gate{}'.format(i))(x)
            x = g * h + (1 - g) * x
        x = self.highway_linear(x)
        x = nn.functional.dropout(x, self.dropout, self.training)
        return x


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.pre_mix = args.pre_mix

        self.self_attn = MultiheadAttention(
            self.embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )

        self.norm_layer = Norm_Layer(args)

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape (T_tgt, T_src), where
            T_tgt is the length of query, while T_src is the length of key,
            though here both query and key is x here,
            attn_mask[t_tgt, t_src] = 1 means when calculating embedding
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """

        residual = x

        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        ############ txt features attention  ##################
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )

        x = self.norm_layer(x, residual)

        return x


class TransformerEncoderLayer_grid(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.dropout = args.dropout
        self.pre_mix = args.pre_mix

        self.self_attn = MultiheadAttention(
            self.embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )

        self.norm_layer = Norm_Layer(args)

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, grid_img_features, grid_img_mask, attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape (T_tgt, T_src), where
            T_tgt is the length of query, while T_src is the length of key,
            though here both query and key is x here,
            attn_mask[t_tgt, t_src] = 1 means when calculating embedding
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """

        residual = grid_img_features

        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        ############ txt features attention  ##################
        grid_img_features, _ = self.self_attn(
            query=grid_img_features,
            key=grid_img_features,
            value=grid_img_features,
            key_padding_mask=grid_img_mask,
            attn_mask=attn_mask,
        )

        # grid_img_features = self.norm_layer(grid_img_features,residual)

        return grid_img_features


class TransformerEncoderLayer_region(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.dropout = args.dropout
        self.pre_mix = args.pre_mix

        self.self_attn = MultiheadAttention(
            self.embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )

        self.norm_layer = Norm_Layer(args)

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, region_img_features, region_img_mask, attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape (T_tgt, T_src), where
            T_tgt is the length of query, while T_src is the length of key,
            though here both query and key is x here,
            attn_mask[t_tgt, t_src] = 1 means when calculating embedding
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """

        residual = region_img_features

        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        ############ txt features attention  ##################
        region_img_features, _ = self.self_attn(
            query=region_img_features,
            key=region_img_features,
            value=region_img_features,
            key_padding_mask=region_img_mask,
            attn_mask=attn_mask,
        )

        # region_img_features = self.norm_layer(region_img_features,residual)

        return region_img_features


class TransformerEncoderLayer_text_region(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.dropout = args.dropout
        self.pre_mix = args.pre_mix

        self.self_attn = MultiheadAttention(
            self.embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )

        self.norm_layer = Norm_Layer(args)

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self,text_region_img_features, text_region_img_mask, attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape (T_tgt, T_src), where
            T_tgt is the length of query, while T_src is the length of key,
            though here both query and key is x here,
            attn_mask[t_tgt, t_src] = 1 means when calculating embedding
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """

        residual = text_region_img_features

        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        ############ txt features attention  ##################
        text_region_img_features, _ = self.self_attn(
            query=text_region_img_features,
            key=text_region_img_features,
            value=text_region_img_features,
            key_padding_mask=text_region_img_mask,
            attn_mask=attn_mask,
        )

        text_region_img_features = self.norm_layer(text_region_img_features, residual)

        return text_region_img_features


class TransformerEncoderLayer_text_grid(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.dropout = args.dropout
        self.pre_mix = args.pre_mix

        self.self_attn = MultiheadAttention(
            self.embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )

        self.norm_layer = Norm_Layer(args)

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]


    def forward(self,x, text_grid_img_features,text_grid_mask,attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape (T_tgt, T_src), where
            T_tgt is the length of query, while T_src is the length of key,
            though here both query and key is x here,
            attn_mask[t_tgt, t_src] = 1 means when calculating embedding
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x

        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        text_grid_img_features, _ = self.self_attn(
            query=x,
            key=text_grid_img_features,
            value=text_grid_img_features,
            key_padding_mask=text_grid_mask,
            attn_mask=attn_mask,
        )

        text_grid_img_features = self.norm_layer(text_grid_img_features, residual)



        return text_grid_img_features


class TransformerEncoderLayer_text_grid_region(nn.Module):
    """Encoder layer block.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.dropout = args.attention_dropout
        self.embed_dim = args.encoder_embed_dim  # 512

        self.final_liner = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.fc_img = Linear(args.gating_dim * 2, 1)
        self.mix_up = Linear(self.embed_dim, self.embed_dim)

        self.self_attn_x = MultiheadAttention(
            self.embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )

        self.self_attn_x_region = MultiheadAttention(
            self.embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )

        self.self_attn_img_x = MultiheadAttention(
            self.embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )
        self.Threshold = args.Threshold

        self.norm_layer = Norm_Layer(args)

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def gating_multimodel_mix(self,x, text_grid):

        # ### rt
        # x = torch.mean(x, dim=0, keepdim=True)
        # grid_x_features = torch.cat([text_grid, x.repeat(text_grid.size(0), 1, 1)], dim=-1)
        # grid_linear_x = self.fc_img(grid_x_features)
        # rt = torch.sigmoid(grid_linear_x)
        # # max_len * batch * 1

        ### zt
        x = torch.mean(x, dim=0, keepdim=True)
        grid_x_features = torch.cat([text_grid, x.repeat(text_grid.size(0), 1, 1)], dim=-1)
        grid_linear_x = self.fc_img(grid_x_features)
        zt = torch.sigmoid(grid_linear_x)

        ### nt
        x = torch.mean(x, dim=0, keepdim=True)
        grid_x_features = torch.cat([zt*text_grid, (x.repeat(text_grid.size(0), 1, 1))], dim=-1)
        grid_linear_x = self.fc_img(grid_x_features)
        nt = torch.torch.tanh((grid_linear_x))

        ### ht
        ht = (1-zt)*nt + zt*x           #41.87
        # ht = (1-zt)*nt + zt*text_grid #
        # ht = (1-zt)*text_grid + nt*x

        return ht


    # def gating_multimodel_mix(self,x, text_grid, text_region):
    #     ####  stra visual to textual  #########
    #     # pseudo_features = img[torch.LongTensor(np.random.randint(0, img.size(0), x.size(0)))]
    #     # alpha = torch.tensor([random.betavariate(1, 1) for _ in range(x.size(1))]).unsqueeze(0).unsqueeze(-1).type_as(x)
    #     # mixed_x = alpha * x + (1 - alpha) * pseudo_features
    #     # mixed_x = self.mix_up(mixed_x)
    #     # grid_img_features = torch.cat([x, img, mixed_x], dim=0)
    #
    #     ####text_grid==>ht
    #     ####text_grid==>ct
    #     ####resudial==>x
    #
    #     ### it
    #     grid_img_features = torch.mean(text_grid, dim=0, keepdim=True)
    #     grid_x_features = torch.cat([x, grid_img_features.repeat(x.size(0), 1, 1)], dim=-1)
    #     grid_linear_x = self.fc_img(grid_x_features)
    #     it = torch.sigmoid(grid_linear_x)
    #
    #     #### ft
    #     grid_img_features = torch.mean(text_grid, dim=0, keepdim=True)
    #     grid_x_features = torch.cat([x, grid_img_features.repeat(x.size(0), 1, 1)], dim=-1)
    #     grid_linear_x = self.fc_img(grid_x_features)
    #     ft = torch.sigmoid(grid_linear_x)
    #
    #     #### gt
    #     grid_img_features = torch.mean(text_grid, dim=0, keepdim=True)
    #     grid_x_features = torch.cat([x, grid_img_features.repeat(x.size(0), 1, 1)], dim=-1)
    #     grid_linear_x = self.fc_img(grid_x_features)
    #     gt = torch.tanh(grid_linear_x)
    #
    #     ### ot
    #     grid_img_features = torch.mean(text_grid, dim=0, keepdim=True)
    #     grid_x_features = torch.cat([x, grid_img_features.repeat(x.size(0), 1, 1)], dim=-1)
    #     grid_linear_x = self.fc_img(grid_x_features)
    #     ot = torch.sigmoid(grid_linear_x)
    #
    #     ### ct
    #     # ct = (ft* x) + (x * gt)
    #     ct = (ft*text_region) + (it*gt)
    #
    #     ### ht
    #     # ht = ct
    #     ht = ot*torch.tanh(ct) +x
    #
    #     return ht

    def gating(self, x, region_img_features):

        x_grid = region_img_features
        region_img_features = torch.mean(region_img_features, dim=0, keepdim=True)
        region_x_features = torch.cat([x, region_img_features.repeat(x.size(0), 1, 1)], dim=-1)
        region_linear_x = self.fc_img(region_x_features)

        region_sigmoid_x = torch.sigmoid(region_linear_x)  # max_len * batch * 1
        region_img_x = torch.mul(region_sigmoid_x, region_img_features)

        region_img_features = region_img_x + x
        return region_img_features




    def forward(self,x, region_img_features, text_region_mask, encoder_padding_mask, id_x, src_taokens,attn_mask: Optional[Tensor] = None):


        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        orgian = x
        x, _ = self.self_attn_x(
            query= x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )

        region_img_features, _ = self.self_attn_img_x(
            query=region_img_features,
            key=region_img_features,
            value=region_img_features,
            key_padding_mask=text_region_mask,
            attn_mask=attn_mask,
        )

        # region_img_features = region_img_features[:, torch.randperm(region_img_features.size(1)), :]
        # region_img_features[:,:,:] = 1e-20
        text_region_img_features, _ = self.self_attn_x_region(
            query= x,
            key=region_img_features,
            value=region_img_features,
            key_padding_mask=text_region_mask,
            attn_mask=attn_mask,
        )

        # region_img_features, _ = self.self_attn_img_x(
        #     query=region_img_features,
        #     key=region_img_features,
        #     value=region_img_features,
        #     key_padding_mask=text_region_mask,
        #     attn_mask=attn_mask,
        # )
        #
        # interaction_features = torch.cat([x,region_img_features], dim=0)
        #
        # text_region_img_features, _ = self.self_attn_x_region(
        #     query= x,
        #     key=interaction_features,
        #     value=interaction_features,
        #     key_padding_mask=interaction_mask,
        #     attn_mask=attn_mask,
        # )




        text_region_img_features = self.gating_multimodel_mix(x, text_region_img_features)
        ###wu 论文中的门控方法
        # text_region_img_features = self.gating(x, text_region_img_features)

        # if id_x>=3:
        #     img_text = self.norm_layer(text_region_img_features, orgian)
        #
        # else:
        # text_region_img_features = self.gating(x, region_img_features)

        # log_text_region_img_features = nn.LogSoftmax(dim=1)(text_region_img_features)
        # log_x = nn.LogSoftmax(dim=0)(x)

        # log_text_region_img_features = F.softmax(text_region_img_features, dim=0)
        # log_x = F.softmax(x, dim=0)


        log_text_region_img_features = text_region_img_features
        log_x = x

        ## filter_matrix
        # # kl_matrix_1 = torch.nn.functional.kl_div(log_text_region_img_features, log_x, reduction='none')
        # kl_matrix_2 = torch.mul(log_x,log_text_region_img_features)/log_x.size(-1)
        # kl_matrix_2 = torch.softmax(kl_matrix_2,dim=-1)
        # kl_matrix_2 = torch.sum(kl_matrix_2, dim=-1,keepdim=True)
        # kl_matrix_2 = torch.mul(kl_matrix_2,log_text_region_img_features)

        kl_matrix_2 = torch.nn.functional.kl_div(log_x, log_text_region_img_features, reduction='none')

        # kl_matrix_1 = torch.nn.functional.kl_div(log_text_region_img_features, log_x, reduction='sum')
        # kl_matrix_2 = torch.nn.functional.kl_div(log_x, log_text_region_img_features, reduction='sum')

        # kl_matrix_1 = torch.sum(kl_matrix_1,dim=-1, keepdim= True)/ log_x.size(-1)

        kl_matrix_2 = torch.sum(kl_matrix_2, dim=-1, keepdim=True)/log_text_region_img_features.size(-1)
        kl_matrix_all = - ( kl_matrix_2)

        # filter = text_region_img_features
        # filter_1 = x
        # filter_2 = text_region_img_features
        ### pad_mask
        ### 方法一
        # pad_mask = np.bool_(kl_matrix_all.cpu().detach().numpy())
        # pad_mask = torch.tensor(pad_mask).cuda()
        # torch.where()
        pad_mask = kl_matrix_all
        # for i in range(0, kl_matrix_all.size(0)):
        #     for j in range(0, kl_matrix_all.size(1)):
        #         for k in range(0, kl_matrix_all.size(-1)):
        #             if kl_matrix_all[i][j][k]>=0.2:
        #                 pad_mask[i][j][k]=False
        #             else:
        #                 pad_mask[i][j][k]=True
        ###方法一
        ones = torch.ones_like(kl_matrix_all)
        zeros = torch.zeros_like(kl_matrix_all)
        filter_matrix_text = torch.where(pad_mask > self.Threshold, zeros, ones)
        filter_matrix_multi = torch.where(pad_mask <= self.Threshold, zeros, ones)


        ### 方法二:
        # pad_mask = kl_matrix_all.eq(0)
        # pad_mask = pad_mask.eq(1)
        # op_pad_mask = ~ pad_mask
        # filter_1 = filter_1.masked_fill(pad_mask, 1.)  ## True 为1  Flase 保持不变。
        # filter_matrix_text = filter_1.masked_fill(op_pad_mask, 0)
        #
        # filter_2 = filter_2.masked_fill(pad_mask, 0.)  ## True 为0  Flase 保持不变。
        # filter_matrix_multi = filter_2.masked_fill(op_pad_mask, 1.)

        filter_text_region_img_features = text_region_img_features * filter_matrix_multi
        filter_x = x * filter_matrix_text

        img_text = filter_text_region_img_features + filter_x           #

        # if idx>3:
        # x_grid = self.gating_multimodel_mix(orgian, x)
        # x_grid = self.gating(x, x_grid)

        # img_text = self.norm_layer(img_text,orgian)
        img_text = self.norm_layer(img_text, orgian)


        return img_text


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
            self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.cross_self_attention = getattr(args, "cross_self_attention", False)
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not self.cross_self_attention,
        )
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        self.activation_dropout = getattr(args, "activation_dropout", 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, "relu_dropout", 0)
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = MultiheadAttention(
                self.embed_dim,
                args.decoder_attention_heads,
                kdim=getattr(args, "encoder_embed_dim", None),
                vdim=getattr(args, "encoder_embed_dim", None),
                dropout=args.attention_dropout,
                encoder_decoder_attention=True,
            )
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
            self,
            x,
            encoder_out: Optional[torch.Tensor] = None,
            encoder_padding_mask: Optional[torch.Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            prev_self_attn_state: Optional[List[torch.Tensor]] = None,
            prev_attn_state: Optional[List[torch.Tensor]] = None,
            self_attn_mask: Optional[torch.Tensor] = None,
            self_attn_padding_mask: Optional[torch.Tensor] = None,
            need_attn: bool = False,
            need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
                incremental_state is not None
                and _self_attn_input_buffer is not None
                and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=float(self.activation_dropout), training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class Norm_Layer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.embed_dim = args.encoder_embed_dim
        self.dropout = args.dropout
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")

        )
        self.activation_dropout = getattr(args, "activation_dropout", 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, "relu_dropout", 0)

        self.normalize_before = args.encoder_normalize_before

        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)

        # self.fc_con = Linear(2*self.embed_dim, self.embed_dim)
        # self.fc_con_layer_norm = LayerNorm(self.embed_dim)
        # self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, x, residual):

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + residual
        if not self.normalize_before:  # self.normalize_before = False
            x = self.self_attn_layer_norm(x)
        # frist add & Norm

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=float(self.activation_dropout), training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        if not self.normalize_before:  # self.normalize_before = False
            x = self.self_attn_layer_norm(x)

        return x


class MLP(nn.Module):
    def __init__(self, n_i, n_h, n_o):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(n_i, n_h)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(n_h, n_o)

    def forward(self, input):
        return self.linear2(self.relu(self.linear1(input)))

