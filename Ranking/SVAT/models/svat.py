import torch_geometric
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, dimensions, num_stocks, attention_type='general'):
        # dimensions: hidden size of the feature, e.g., 32
        # num_stocks: number of stocks, e.g., 1026
        super(Attention, self).__init__()
        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.num_stocks = num_stocks
        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)
        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        # self.ae = nn.Parameter(torch.FloatTensor(num_stocks,1,1))
        # self.ab = nn.Parameter(torch.FloatTensor(num_stocks,1,1))
        self.ae = nn.Parameter(torch.zeros(num_stocks, 1, 1))
        self.ab = nn.Parameter(torch.zeros(num_stocks,1,1))

    def forward(self, query, context, stock_idx=None):
        # query: (stock_num, 1, hid_size)
        # context: (stock_num, seq_len, hid_size)
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        mix = attention_weights*(context.permute(0,2,1))

        delta_t = torch.flip(torch.arange(0, query_len), [0]).type(torch.float32).to(query.device)
        delta_len = len(stock_idx) if stock_idx is not None else self.num_stocks
        delta_t = delta_t.repeat(delta_len, 1).reshape(delta_len, 1, query_len)
        ab = self.ab[stock_idx] if stock_idx is not None else self.ab
        bt = torch.exp(-1 * ab * delta_t)
        ae = self.ae[stock_idx] if stock_idx is not None else self.ae
        term_2 = F.relu(ae * mix * bt)
        mix = torch.sum(term_2+mix, -1).unsqueeze(1)
        combined = torch.cat((mix, query), dim=2)

        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights

class HGAttn(nn.Module):
    def __init__(self, hid_size=32, drop_rate=0.1):
        super(HGAttn, self).__init__()
        self.hid_bn = nn.BatchNorm1d(hid_size)
        self.hatt1 = torch_geometric.nn.HypergraphConv(hid_size, hid_size, use_attention=True, heads=4, concat=False, negative_slope=0.2, dropout=drop_rate, bias=True)
        self.hatt2 = torch_geometric.nn.HypergraphConv(hid_size, hid_size, use_attention=True, heads=1, concat=False, negative_slope=0.2, dropout=drop_rate, bias=True)

    def gen_edge_features(self, node_features, hyperedge_index):
        # node_features: (stock_num, fea_dim)
        # hyperedge_index: (2, num_edge)
        edge_idxs = [int(i) for i in hyperedge_index[1, :]]
        edge_dict = {}
        for v_idx, e_idx in enumerate(edge_idxs):
            if e_idx not in edge_dict.keys():
                edge_dict[e_idx] = []
            edge_dict[e_idx].append(int(hyperedge_index[0, v_idx]))

        edge_features = None
        for e_idx in sorted(edge_dict):
            temp_feature = torch.sum(node_features[edge_dict[e_idx]], dim=0).unsqueeze(0)
            if edge_features is not None:
                edge_features = torch.cat([edge_features, temp_feature], dim=0)
            else:
                edge_features = temp_feature
        
        return edge_features

    def forward(self, node_fea, e):
        # node_fea: (stock_num, hid_size)
        # e: (2, num_edge)
        x = self.hid_bn(node_fea)
        edge_features = self.gen_edge_features(x, e)
        x = F.leaky_relu(self.hatt1(x, e, hyperedge_attr=edge_features), 0.2)
        edge_features = self.gen_edge_features(x, e)
        output = F.leaky_relu(self.hatt2(x, e, hyperedge_attr=edge_features), 0.2)

        return output

class SVAT(nn.Module):
    def __init__(self, num_stocks, fea_dim=5, hid_size=32, drop_rate=0.1, seq_len=20):
        super(SVAT, self).__init__()
        self.fea_dim = fea_dim
        self.seq_len = seq_len
        self.hid_size = hid_size
        self.in_bn = nn.BatchNorm1d(seq_len * fea_dim)
        self.gru = nn.GRU(input_size = fea_dim, hidden_size=hid_size, batch_first=True)
        self.attention = Attention(hid_size, num_stocks)
        self.hg_attn = HGAttn(hid_size, drop_rate)
        self.output_layer = nn.Sequential(
            nn.Linear(hid_size, 1),
            nn.LeakyReLU()
        )

    def adv(self, adv_input, e=None):
        x = adv_input
        if e is not None:
            x = self.hg_attn(x, e)
        output = self.output_layer(x)

        return output

    def forward(self, price_input, e, stock_idx=None):
        # price_input: (stock_num, seq_len, 5)
        # e: [2, num_edge]
        in_x = self.in_bn(price_input.view(-1, self.seq_len * self.fea_dim))
        in_x = in_x.view(-1, self.seq_len, self.fea_dim)
        context, query  = self.gru(in_x)
        query = query.reshape(query.size(1), 1, self.hid_size)
        node_features, weights = self.attention(query, context, stock_idx)
        node_features = node_features.reshape((node_features.size(0), self.hid_size))

        x = self.hg_attn(node_features, e)
        output = self.output_layer(x)

        return node_features, x, output

class AdvSampler(nn.Module):
    def __init__(self, input_size, output_size, hid_size=128):
        super(AdvSampler, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hid_size),
            nn.Tanh()
        )
        self.mu_layer = nn.Linear(hid_size, output_size)
        self.sig_layer = nn.Sequential(
            nn.Linear(hid_size, output_size),
            nn.Softplus()
        )

    def forward(self, xdelta_or_x):
        # xdelta_or_x: (stock_num, input_size)
        h_z = self.encoder(xdelta_or_x)
        z_mu = self.mu_layer(h_z)
        z_std = self.sig_layer(h_z)

        return z_mu, z_std

class AdvGenerator(nn.Module):
    def __init__(self, input_size, output_size):
        super(AdvGenerator, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.Tanh()
        )

    def forward(self, xz):
        # xz: (stock_num, input_size)
        return self.generator(xz)
