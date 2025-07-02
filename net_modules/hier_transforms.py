import torch
from torch import nn, einsum
import torch.nn.functional as F
import torchvision
#
from torch import Tensor
from torch.nn.parameter import Parameter
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, cast

def getPositionEncoding(specified_ks, d=128, n=10000):
    '''
	seq_len should be the relative time in this interval
	'''
    seq_len = len(specified_ks)
    P = torch.zeros((seq_len, d)).cuda()
    for j, k in enumerate(specified_ks):
        for i in range(int(d / 2)):
            denominator = torch.pow(torch.tensor(n).cuda(), 2 * i / d)
            P[j, 2 * i] = torch.sin(torch.tensor(k).cuda() / denominator)
            P[j, 2 * i + 1] = torch.cos(torch.tensor(k).cuda() / denominator)
    return P


def getPositionEncoding_naive(seq_len, d=128, n=10000):
    '''
	seq_len should be the relative time in this interval
	'''

    P = torch.zeros((seq_len, d)).cuda()
    for k in range(seq_len):
        for i in range(int(d / 2)):
            denominator = torch.pow(torch.tensor(n).cuda(), 2 * i / d)
            P[k, 2 * i] = torch.sin(torch.tensor(k).cuda() / denominator)
            P[k, 2 * i + 1] = torch.cos(torch.tensor(k).cuda() / denominator)
    return P
class LinearEmbeddings(nn.Module):
    """Linear embeddings for continuous features.

    **Shape**

    - Input: `(*, n_features)`
    - Output: `(*, n_features, d_embedding)`

    **Examples**

    >>> batch_size = 2
    >>> n_cont_features = 3
    >>> x = torch.randn(batch_size, n_cont_features)
    >>> d_embedding = 4
    >>> m = LinearEmbeddings(n_cont_features, d_embedding)
    >>> m(x).shape
    torch.Size([2, 3, 4])
    """

    def __init__(self, n_features: int, d_embedding: int) -> None:
        """
        Args:
            n_features: the number of continous features.
            d_embedding: the embedding size.
        """
        if n_features <= 0:
            raise ValueError(f'n_features must be positive, however: {n_features=}')
        if d_embedding <= 0:
            raise ValueError(f'd_embedding must be positive, however: {d_embedding=}')

        super().__init__()
        self.weight = Parameter(torch.empty(n_features, d_embedding))
        self.bias = Parameter(torch.empty(n_features, d_embedding))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d_rqsrt = self.weight.shape[1] ** -0.5
        nn.init.uniform_(self.weight, -d_rqsrt, d_rqsrt)
        nn.init.uniform_(self.bias, -d_rqsrt, d_rqsrt)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 2:
            raise ValueError(
                f'The input must have at least two dimensions, however: {x.ndim=}'
            )

        x = x[..., None] * self.weight
        x = x + self.bias[None]
        return x


class CategoricalEmbeddings(nn.Module):
    """Embeddings for categorical features.

    **Examples**

    >>> cardinalities = [3, 10]
    >>> x = torch.tensor([
    ...     [0, 5],
    ...     [1, 7],
    ...     [0, 2],
    ...     [2, 4]
    ... ])
    >>> x.shape  # (batch_size, n_cat_features)
    torch.Size([4, 2])
    >>> m = CategoricalEmbeddings(cardinalities, d_embedding=5)
    >>> m(x).shape  # (batch_size, n_cat_features, d_embedding)
    torch.Size([4, 2, 5])
    """

    def __init__(
        self, cardinalities: List[int], d_embedding: int, bias: bool = True
    ) -> None:
        """
        Args:
            cardinalities: the number of distinct values for each feature.
            d_embedding: the embedding size.
            bias: if `True`, for each feature, a trainable vector is added to the
                embedding regardless of a feature value. For each feature, a separate
                non-shared bias vector is allocated.
                In the paper, FT-Transformer uses `bias=True`.
        """
        super().__init__()
        if not cardinalities:
            raise ValueError('cardinalities must not be empty')
        if any(x <= 0 for x in cardinalities):
            i, value = next((i, x) for i, x in enumerate(cardinalities) if x <= 0)
            raise ValueError(
                'cardinalities must contain only positive values,'
                f' however: cardinalities[{i}]={value}'
            )
        if d_embedding <= 0:
            raise ValueError(f'd_embedding must be positive, however: {d_embedding=}')

        self.embeddings = nn.ModuleList(
            [nn.Embedding(x, d_embedding) for x in cardinalities]
        )
        self.bias = (
            Parameter(torch.empty(len(cardinalities), d_embedding)) if bias else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d_rsqrt = self.embeddings[0].embedding_dim ** -0.5
        for m in self.embeddings:
            nn.init.uniform_(m.weight, -d_rsqrt, d_rsqrt)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -d_rsqrt, d_rsqrt)

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        if x.ndim < 2:
            raise ValueError(
                f'The input must have at least two dimensions, however: {x.ndim=}'
            )
        n_features = len(self.embeddings)
        if x.shape[-1] != n_features:
            raise ValueError(
                'The last input dimension (the number of categorical features) must be'
                ' equal to the number of cardinalities passed to the constructor.'
                f' However: {x.shape[-1]=}, len(cardinalities)={n_features}'
            )

        x = torch.stack(
            [self.embeddings[i](x[..., i]) for i in range(n_features)], dim=-2
        )
        if self.bias is not None:
            x = x + self.bias
        return x

class _CLSEmbedding(nn.Module):
    def __init__(self, d_embedding: int) -> None:
        super().__init__()
        self.weight = Parameter(torch.empty(d_embedding))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d_rsqrt = self.weight.shape[-1] ** -0.5
        nn.init.uniform_(self.weight, -d_rsqrt, d_rsqrt)

    def forward(self, batch_dims: Tuple[int]) -> Tensor:
        if not batch_dims:
            raise ValueError('The input must be non-empty')

        return self.weight.expand(*batch_dims, 1, -1)

class VSTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_encoder_layers=2, max_seq_length=20, n_cont_features=8, cat_cardinalities=[27,8,5,16],pos_type='learn_naive'):
        super(VSTransformer, self).__init__()
        # Embedding 层：将输入嵌入到 d_model 维度
        self.cls_embedding = _CLSEmbedding(d_model)
        self.cont_embeddings = (
            LinearEmbeddings(n_cont_features, d_model) if n_cont_features > 0 else None
        )
        self.cat_embeddings = (
            CategoricalEmbeddings(cat_cardinalities, d_model, True)
            if cat_cardinalities
            else None
        )
        # 位置编码
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, max_seq_length, d_model)  # [1, seq_len, d_model]
        )
        self.pos_type = pos_type
        if pos_type == 'sin_rtime_mod':

            self.vs_encoding = nn.Parameter(torch.zeros(1, d_model))
        if pos_type == 'learn_rtime_mod':
            self.pos_mlp = nn.Sequential(
                nn.Linear(2, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model)
            )
        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
    def forward(self, x_cont, x_cat, mask_cont, mask_cat, agg=True, time_stamps=None):
        # mask shape B,N,D to B,N+1,D with new as False
        mask = torch.cat([mask_cont,mask_cat], dim=1)
        mask = F.pad(mask, (1, 0), value=False) # 最后一维左侧填充
        x_any = x_cont
        x_embeddings = [self.cls_embedding(x_any.shape[:-1])]
        x_cat = torch.tensor(x_cat, dtype=torch.long)

        for x, module in [
            (x_cont, self.cont_embeddings),
            (x_cat, self.cat_embeddings),
        ]:
            x_embeddings.append(module(x))

        x = torch.cat(x_embeddings, dim=1)
        # x: [batch_size, seq_len]
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :]

        # Transformer Encoder
        x = self.transformer_encoder(x, src_key_padding_mask=mask)  # [seq_len, batch_size, d_model]
        if self.pos_type == 'sin_rtime_mod':
            x[:, 0, :] = x[:, 0, :] + self.vs_encoding
        if self.pos_type == 'learn_rtime_mod':
            # mod = 2 vs

            mlp_input = torch.cat((time_stamps.unsqueeze(-1).float(), torch.ones_like(time_stamps.unsqueeze(-1))+1.0),
                                  dim=-1).cuda()

            learned_pos = self.pos_mlp(mlp_input)
            x[:, 0, :] = x[:, 0, :] + learned_pos
        if self.pos_type == 'sin_naive':
            sin_pos = getPositionEncoding_naive(x.shape[0], d=x.shape[-1])
            x[:, 0, :] = x[:, 0, :] + sin_pos
        if self.pos_type.startswith('sin_rtime'):
            sin_pos = getPositionEncoding(time_stamps, d=x.shape[-1])
            x[:, 0, :] = x[:, 0, :] + sin_pos

        if agg:
            return x[:, 0, :]
        else:
            return x[:, 1:, :]


class LabTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_encoder_layers=2, max_seq_length=50, n_cont_features=45, pos_type='learn_naive'):
        super(LabTransformer, self).__init__()
        # Embedding 层：将输入嵌入到 d_model 维度
        self.cls_embedding = _CLSEmbedding(d_model)
        self.cont_embeddings = (
            LinearEmbeddings(n_cont_features, d_model) if n_cont_features > 0 else None
        )
        # 位置编码
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, max_seq_length, d_model)  # [1, seq_len, d_model]
        )
        self.pos_type = pos_type
        if pos_type == 'sin_rtime_mod':
            self.lab_encoding = nn.Parameter(torch.zeros(1, d_model))
        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        if pos_type == 'learn_rtime_mod':
            self.pos_mlp = nn.Sequential(
                nn.Linear(2, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model)
            )
    def forward(self, x_cont, mask, agg=True, time_stamps=None):
        # mask shape B,N to B,N+1with new as False
        mask = F.pad(mask, (1, 0), value=False) # 最后一维左侧填充
        x_embeddings = [self.cls_embedding(x_cont.shape[:-1]),self.cont_embeddings(x_cont)]
        x = torch.cat(x_embeddings, dim=1)
        # x: [batch_size, seq_len]
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :]
        # Transformer Encoder
        x = self.transformer_encoder(x, src_key_padding_mask=mask)  # [seq_len, batch_size, d_model]
        if self.pos_type == 'sin_rtime_mod':
            x[:, 0, :] = x[:, 0, :] + self.lab_encoding
        if self.pos_type == 'learn_rtime_mod':
            # mod = 1 lab

            mlp_input = torch.cat((time_stamps.unsqueeze(-1).float(), torch.ones_like(time_stamps.unsqueeze(-1))),
                                  dim=-1).cuda()

            learned_pos = self.pos_mlp(mlp_input)
            x[:, 0, :] = x[:, 0, :] + learned_pos
        if self.pos_type == 'sin_naive':
            sin_pos = getPositionEncoding_naive(x.shape[0], d=x.shape[-1])
            x[:, 0, :] = x[:, 0, :] + sin_pos
        if self.pos_type.startswith('sin_rtime'):
            sin_pos = getPositionEncoding(time_stamps, d=x.shape[-1])
            x[:, 0, :] = x[:, 0, :] + sin_pos
        if agg:
            return x[:, 0, :]
        else:
            return x[:, 1:, :]


class CXRCNN(nn.Module):

    def __init__(self, d_model, model_type, pos_type='learn_naive'):

        super(CXRCNN, self).__init__()
        # self.vision_backbone = torchvision.models.resnet50(pretrained=True)
        self.vision_backbone = getattr(torchvision.models,model_type)(pretrained=True)
        self.vision_backbone.fc = nn.Linear(in_features=self.vision_backbone.fc.in_features, out_features=d_model)
        self.pos_type = pos_type
        if pos_type == 'sin_rtime_mod':
            self.cxr_encoding = nn.Parameter(torch.zeros(1, d_model))
        if pos_type == 'learn_rtime_mod':
            self.pos_mlp = nn.Sequential(
                nn.Linear(2, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model)
            )



    def forward(self, x, time_stamps=None):
        x = self.vision_backbone(x)
        if self.pos_type == 'sin_rtime_mod':
            x = x + self.cxr_encoding
        if self.pos_type == 'learn_rtime_mod':
            # mod = 3 cxr

            mlp_input = torch.cat((time_stamps.unsqueeze(-1).float(), torch.ones_like(time_stamps.unsqueeze(-1))+3.0),
                                  dim=-1).cuda()

            learned_pos = self.pos_mlp(mlp_input)
            x = x + learned_pos
        if self.pos_type == 'sin_naive':
            sin_pos = getPositionEncoding_naive(x.shape[0], d=x.shape[-1])
            x = x + sin_pos
        if self.pos_type.startswith('sin_rtime'):
            sin_pos = getPositionEncoding(time_stamps, d=x.shape[-1])
            x = x + sin_pos
        return x

class TimeTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_encoder_layers=2, max_seq_length=2000,output_dim=1, use_learnable_pos=True):
        super(TimeTransformer, self).__init__()
        # Embedding 层：将输入嵌入到 d_model 维度
        self.cls_embedding = _CLSEmbedding(d_model)
        self.use_learnable_pos = use_learnable_pos
        if self.use_learnable_pos:
            self.positional_encoding = nn.Parameter(
                torch.zeros(1, max_seq_length, d_model)  # [1, seq_len, d_model]
            )
        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x, mask=None,output_logits=False):
        # mask shape B,N to B,N+1 with new as False
        if mask is not None:
            mask = F.pad(mask, (1, 0), value=False) # 最后一维左侧填充
        x_embeddings = [self.cls_embedding(x.shape[:1]),x]
        x = torch.cat(x_embeddings, dim=1)
        # x: [batch_size, seq_len]
        if self.use_learnable_pos:
            seq_len = x.size(1)
            x = x + self.positional_encoding[:, :seq_len, :]
        # Transformer Encoder
        if mask is not None:
            x = self.transformer_encoder(x, src_key_padding_mask=mask)  # [seq_len, batch_size, d_model]
        else:
            x = self.transformer_encoder(x)
        if output_logits:
            return self.fc(x[:, 0, :])
        else:
            return x[:, 0, :]


class MMFusionTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_encoder_layers=2, output_dim=3):
        super(MMFusionTransformer, self).__init__()
        # Embedding 层：将输入嵌入到 d_model 维度
        self.cls_embedding = _CLSEmbedding(d_model)
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, 4, d_model)  # [1, seq_len, d_model]
        )
        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, output_dim)
    def forward(self, x, mask=None,output_logits=False):
        # mask shape B,N,D to B,N+1,D with new as False
        x = x.permute(1,0,2)
        if mask is not None:
            mask = mask.permute(1,0)
            mask = F.pad(mask, (1, 0), value=False) # 最后一维左侧填充
        x_embeddings = [self.cls_embedding(x.shape[:1]),x]
        x = torch.cat(x_embeddings, dim=1)
        # x: [batch_size, seq_len]
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :]
        # Transformer Encoder
        if mask is not None:
            x = self.transformer_encoder(x, src_key_padding_mask=mask)  # [seq_len, batch_size, d_model]
        else:
            x = self.transformer_encoder(x)
        if output_logits:
            return self.fc(x[:, 0, :])
        else:
            return x[:, 0, :]

class PredTimeTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_encoder_layers=2, max_seq_length=200, output_dim=1):
        super(PredTimeTransformer, self).__init__()
        # Embedding 层：将输入嵌入到 d_model 维度
        self.cls_embedding = _CLSEmbedding(d_model)
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, max_seq_length, d_model)  # [1, seq_len, d_model]
        )
        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, output_dim)
    def forward(self, x, mask=None):
        # mask shape B,N to B,N+1 with new as False
        if mask is not None:
            mask = F.pad(mask, (1, 0), value=False) # 最后一维左侧填充
        x_embeddings = [self.cls_embedding(x.shape[:1]),x]
        x = torch.cat(x_embeddings, dim=1)
        # x: [batch_size, seq_len]
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :]
        # Transformer Encoder
        if mask is not None:
            x = self.transformer_encoder(x, src_key_padding_mask=mask)  # [seq_len, batch_size, d_model]
        else:
            x = self.transformer_encoder(x)
        logits = self.fc(x[:, 0, :])
        return logits
