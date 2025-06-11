import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from geoopt.utils import clamp_abs
from geoopt.manifolds.stereographic import math as gmath


class GeometricLinearizedAttention(nn.Module):
    def __init__(self, c, num_heads, head_dim, drop_attn):
        
        super().__init__()
        
        self.c = c
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.eps = 1e-5
        
    def forward(self, Q, K, V, mask):
        
        if len(self.c) == 1:
            curv = self.c.repeat(self.num_heads)[:,None,None]
        else:
            curv = self.c[:,None,None]

        x = V
        
        v1 = gmath.parallel_transport0back(x, Q, k=curv)  # [B, H, N, D]
        v2 = gmath.parallel_transport0back(x, K, k=curv)  # [B, H, N, D]
        
        gamma = gmath.lambda_x(x=x, k=curv, keepdim=True, dim=-1)
        denom = clamp_abs((gamma - 1), 1e-10)
        
        x = ((gamma / denom) * x) * mask[None, :, None]
        
        v1 = (nn.functional.elu(v1) + 1)
        v2 = (denom * (nn.functional.elu(v2) + 1)) * mask[None, :, None]

        # Linearized approximation
        v2_cumsum = v2.sum(dim=-2) # [B, H, D]
        D = torch.einsum('...nd,...d->...n', v1, v2_cumsum.type_as(v1)) # normalization terms
        D_inv = 1./D.masked_fill_(D == 0, self.eps)
        context = torch.einsum('...nd,...ne->...de', v2, x)
        X = torch.einsum('...de,...nd,...n->...ne', context, v1, D_inv)
        
        
        X = gmath.project(X, k=curv)
        X = gmath.mobius_scalar_mul(torch.tensor(0.5, dtype=X.dtype, device=X.device),
                                    X,
                                    k=curv, 
                                    dim=-1)
        X = gmath.project(X, k=curv)
        
        return X


class StereographicAttentionLayer(nn.Module):
    def __init__(self, 
                 manifold,
                 c, 
                 dim, 
                 head_dim, 
                 num_heads, 
                 attn_type="vanilla", 
                 attn_drop=0.):

        super().__init__()
        
        self.manifold = manifold
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        self.W_q = nn.Linear(dim, num_heads * head_dim)
        self.W_k = nn.Linear(dim, num_heads * head_dim)
        self.W_v = StereographicLinear(manifold, dim, num_heads * head_dim, c, num_heads)
        
        self.attn = GeometricLinearizedAttention(c, num_heads, head_dim, attn_drop)
        self.ff = StereographicLinear(manifold, num_heads*head_dim, dim, c, num_heads)
        
    def forward(self, X, mask):
        Q = self.split_heads(self.W_q(X)) # [B, H, N, D]
        K = self.split_heads(self.W_k(X))
        V = self.split_heads(self.W_v(X))
        
        attn_out = self.attn(Q, K, V, mask)
        attn_out = self.combine_heads(attn_out)
        
        out = self.ff(attn_out)
        
        return out
    
    def combine_heads(self, X):
        X = X.transpose(0, 1) # [N, H, D]
        X = X.reshape(X.size(0), self.num_heads * self.head_dim) # [N, H*D]
        return X
    
    def split_heads(self, X):
        X = X.reshape(X.size(0), self.num_heads, self.head_dim) # [N, H, D]
        X = X.transpose(0, 1) # [H, N, D]
        return X


class StereographicLayerNorm(nn.Module):

    def __init__(self, manifold, c, dim, num_heads):
        
        super(StereographicLayerNorm, self).__init__()
        
        self.manifold = manifold
        self.c = c
        self.norm = nn.LayerNorm(dim)
        self.num_heads = num_heads

    def forward(self, x):
        
        if len(self.c) == 1:
            curv = self.c.repeat(self.num_heads)
        else:
            curv = self.c
        
        xt = self.norm(self.manifold.logmap0(x, curv))
        output = self.manifold.expmap0(xt, curv)
        output = self.manifold.proj(output, curv)
        return output
        
      
class StereographicAct(nn.Module):
    """
    Stereographic activation layer.
    """

    def __init__(self, manifold, act, num_heads):
        
        super(StereographicAct, self).__init__()
        
        self.manifold = manifold
        self.act = act
        self.num_heads = num_heads

    def forward(self, x, curv):
        
        if len(curv) == 1:
            curv = curv.repeat(self.num_heads)
        else:
            curv = curv
        
        xt = torch.clamp(self.act(self.manifold.logmap0(x, curv)), max=self.manifold.max_norm)
        output = self.manifold.expmap0(xt, curv)
        output = self.manifold.proj(output, curv)
        return output

    def extra_repr(self):
        return 'c={}'.format(
            self.c
        )


class StereographicLinear(nn.Module):
    """
    Stereographic linear layer.
    """
    
    def __init__(self, manifold, 
                 in_features, out_features, 
                 num_heads,
                 dropout=0., use_bias=True):
        
        super(StereographicLinear, self).__init__()
        
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = nn.Parameter(torch.Tensor([0.]*num_heads))
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=1/math.sqrt(2))
        init.constant_(self.bias, 0)
        
        
    # x: [N HD]
    # m: [HD, HD']
    # out: [N HD']
    def mobius_matvec(self, m, x):
        
        if len(self.c) == 1:
            curv = self.c.repeat(self.num_heads)
        else:
            curv = self.c
        
        x, c = self.manifold.split(x, curv)
        u = self.manifold.merge(gmath.logmap0(x, k=c))
        mu, c = self.manifold.split(u @ m.transpose(-1, -2), curv)
        out = self.manifold.merge(gmath.expmap0(mu, k=c))
        return out
        
    def forward(self, x):
        #self.c = curv
        if len(self.c) == 1:
            curv = self.c.repeat(self.num_heads)
        else:
            curv = self.c
        
        assert not torch.isnan(x).any()
        
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        res = self.mobius_matvec(drop_weight, x)
        res = self.manifold.proj(res, curv)
        
        assert not torch.isnan(res).any()
        if self.use_bias:
            bias = self.bias.view(1,-1)
            str_bias = self.manifold.expmap0(bias, curv)
            str_bias = self.manifold.proj(str_bias, curv)
            res = self.manifold.mobius_add(res, str_bias, curv)
            res = self.manifold.proj(res, curv)
        assert not torch.isnan(res).any()
        return res
        
    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )
        

class StereographicTransformer(nn.Module):
    def __init__(self, manifold, c, args):
        
        super(StereographicTransformer, self).__init__()
        assert args.num_layers > 0
        
        self.act = getattr(nn, args.act)()
        
        self.manifold = manifold
        self.c = c
        self.num_heads = args.num_heads
        
        
        self.mha = StereographicAttentionLayer(self.manifold,
                                               self.c,
                                               args.dim,
                                               args.head_dim, 
                                               args.num_heads,
                                               attn_type=args.attn_type, 
                                               attn_drop=args.attn_dropout)
        self.dropout1 = nn.Dropout(p=args.dropout)
        
        if args.layer_norm:
            self.norm1 = StereographicLayerNorm(self.manifold, self.c, args.dim, args.num_heads)
            self.norm2 = StereographicLayerNorm(self.manifold, self.c, args.dim, args.num_heads)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        
        self.mlpblock = nn.Sequential(
            StereographicLinear(manifold, args.dim, args.dim, c, args.num_heads, args.dropout, True),
            StereographicAct(manifold, c, self.act, args.num_heads),
            StereographicLinear(manifold, args.dim, args.dim, c, args.num_heads, args.dropout, True),
        )
        
    def forward(self, X, mask):
        
        if len(self.c) == 1:
            curv = self.c.repeat(self.num_heads)
        else:
            curv = self.c
        
        X = self.manifold.mobius_add(self.dropout1(self.mha(self.norm1(X), mask)), X, curv)
        X = self.manifold.proj(X, curv)
        X = self.manifold.mobius_add(self.mlpblock(self.norm2(X)), X, curv)
        X = self.manifold.proj(X, curv)
        
        return X
    

class StereographicLogits(nn.Module):
    def __init__(self, manifold, input_dim, n_classes, num_heads):
        super().__init__()

        self.manifold = manifold
        self.n_classes = n_classes
        self.num_heads = num_heads

        self.a = nn.parameter.Parameter(torch.randn(1, n_classes, input_dim) * 0.01)
        self.p = nn.parameter.Parameter(torch.zeros(1, n_classes, input_dim))

    def forward(self, x, curv):
        if curv == 1:
            curv = curv.repeat(self.num_heads)
        else:
            curv = curv

        a, c = self.manifold.split(self.a, curv)
        p, _ = self.manifold.split(self.p, curv)
        x, _ = self.manifold.split(x, curv)

        p = gmath.expmap0(p, k=c)
        p = gmath.project(p, k=c)
        a = gmath.parallel_transport0(p, a, k=c)

        z = gmath.mobius_add(-p, x[:, None], k=c)
        z = gmath.project(z, k=c)
        a_norm = a.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-15).sqrt()
        za = (z * a).sum(dim=-1, keepdim=True)

        h_dist = 2 * za / ((1 + c * z.pow(2).sum(dim=-1, keepdim=True)) * a_norm)
        h_dist = gmath.arsin_k(h_dist, k=c * c.abs()) # / c_sqrt
        h_dist = h_dist.pow(2).sum(dim=-2, keepdim=True).clamp_min(1e-15).sqrt()

        logits = (gmath.lambda_x(p, k=c, keepdim=True) * a_norm).pow(2).sum(dim=-2, keepdim=True).clamp_min(1e-15).sqrt() * h_dist * torch.sign(za.sum(dim=-2, keepdim=True))
        logits = logits[..., 0, 0]

        return logits

