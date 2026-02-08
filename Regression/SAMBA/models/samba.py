import torch
import torch.nn as nn
import torch.nn.functional as F
from .mamba import Mamba
from config.model_config import ModelArgs

class SAMBA(nn.Module):
    def __init__(self, args: ModelArgs, hidden, inp, out, embed, cheb_k):
        super().__init__()
        self.args = args
        self.mam1 = Mamba(args, hidden)
        self.cheb_k = cheb_k
        self.gamma = nn.Parameter(torch.tensor(1.))
        self.adj = nn.Parameter(torch.randn(args.vocab_size, embed), requires_grad=True)
        self.embed_w = nn.Parameter(torch.randn(embed, embed), requires_grad=True)
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed, cheb_k, inp, out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed, out))
        self.proj = nn.Linear(args.vocab_size, 1)
        self.proj_seq = nn.Linear(args.seq_in, 1)

    def gaussian_kernel_graph(self, E_A, x, gamma=1.0):
        # Compute pairwise squared Euclidean distance
        x_mean = torch.mean(x, dim=0)
        x_time = torch.mm(x_mean.permute(1, 0), x_mean)

        N = E_A.size(0)
        # Expanding the dimensions to compute pairwise differences
        E_A_expanded = E_A.unsqueeze(0).expand(N, N, -1)
        E_A_T_expanded = E_A.unsqueeze(1).expand(N, N, -1)
        # Pairwise squared Euclidean distances
        distance_matrix = torch.sum((E_A_expanded - E_A_T_expanded) ** 2, dim=2)

        # Apply Gaussian kernel
        A = torch.exp(-gamma * distance_matrix)

        dr = nn.Dropout(0.35)

        A = F.softmax(A, dim=1)
        return dr(A)

    def forward(self, input_ids):
        xx = self.mam1(input_ids)

        ADJ = self.gaussian_kernel_graph(self.adj, xx, gamma=self.gamma)

        I = torch.eye(input_ids.size(2)).cuda()

        support_set = [I, ADJ]

        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * ADJ, support_set[-1]) - support_set[-2])

        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', self.adj, self.weights_pool)  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(self.adj, self.bias_pool)  # N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, xx.permute(0, 2, 1))  # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        out = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  # B,N,D_OUT
        return self.proj(out.permute(0, 2, 1))
