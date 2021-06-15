

import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder
import torch_geometric as pyg
import torch_scatter as pyscatter
import math
import transformers
import bert_mogai as bert_mogai


class EmbeddingSkipNegative(nn.Module):
    def __init__(self, num_emb, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_emb, emb_dim)
        pass

    def forward(self, idx):
        mask = idx < 0
        idx_ = idx.clone()
        idx_[mask] = 0

        emb = self.embedding(idx_)
        emb[mask] = 0

        return emb

    
class ScaledDotProductAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask < 0.5, -1e9)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        return attention.matmul(value)

    
class CosineAttention(nn.Module):
    
    def forward(self, query, key, value, mask=None):
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        
        scores = query.matmul(key.transpose(-2, -1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1)
            pass
        scores = (scores + 1) * 0.5
        attention = torch.div(scores, scores.sum(dim=-1, keepdim=True))
        return attention.matmul(value)


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 in_features,
                 head_num,
                 bias=True,
                 activation=F.relu):
        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError(
                '`in_features`({}) should be divisible by `head_num`({})'\
                .format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        # self.linear_o = nn.Linear(in_features, in_features, bias)
        self.atten = ScaledDotProductAttention()

    def forward(self, q, k, v, mask=None):
        # q, k, v: [B, S, D]
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = torch.unsqueeze(mask, 1).repeat(
                1, self.head_num, 1).reshape(
                    q.shape[0], 1, q.shape[1])
        y = self.atten(q, k, v, mask)
        # y = CosineAttention()(q, k, v, mask)
        y = self._reshape_from_batches(y)

        return y.contiguous()

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)


class CateFeatureEncoder(nn.Module):
    def __init__(self, embed_dim, num_uniq_values):
        super().__init__()
        csum = torch.cumsum(torch.LongTensor(num_uniq_values), dim=0)
        num_emb = csum[-1]
        num_uniq_values = torch.LongTensor(num_uniq_values).reshape(1, 1, -1)
        self.register_buffer('num_uniq_values', num_uniq_values)
        
        starts = torch.cat(
            (torch.LongTensor([0]), csum[:-1])).reshape(1, -1)
        self.register_buffer('starts', starts)
        
        self.embeddings = EmbeddingSkipNegative(
            num_emb, embed_dim)
            
        pass

    def forward(self, x):
        if torch.any(x < 0):
            raise RuntimeError(str(x))
        
        if torch.any(torch.ge(x, self.num_uniq_values)):
            raise RuntimeError(str(x))
            pass
        
        x = x + self.starts

        return self.embeddings(x).sum(dim=-2)
        pass
    pass


class MultiHeadLinear(nn.Module):

    def __init__(self, in_dim, num_heads):
        super().__init__()
        self.weight = nn.Parameter(nn.init.normal_(
            torch.zeros(in_dim)))
        self.bias = nn.Parameter(nn.init.normal_(
            torch.zeros(num_heads)))
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        pass

    def forward(self, x):
        if len(x.shape) < 2:
            x = x.reshape(1, -1)
            pass

        batchsize = x.shape[0]
        out = torch.mul(x, self.weight).reshape(
            (batchsize, self.num_heads, self.head_dim)).sum(dim=-1)

        if len(x.shape) < 2:
            out = out.reshape(self.num_heads)
        return out

    
class GCNLayer(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.linear_transform = nn.Linear(
            config['hidden_size'], config['hidden_size'])
        self.norm = nn.LayerNorm(config['hidden_size'])
        pass
    
    def forward(
            self, edge_index, node_emb: torch.Tensor, edge_emb, node_type=None,
            edge_type=None, *args):

        index_i = edge_index[0, :].view(-1)
        index_j = edge_index[1, :].view(-1)

        xi = node_emb[index_i] + edge_emb
        node_emb_new = pyscatter.scatter_mean(xi, index_j, dim=0)
        node_emb_new = self.linear_transform(node_emb_new)
        node_emb_new = nn.functional.relu(node_emb_new)
        node_emb_new = self.norm(node_emb_new)
        return node_emb_new, edge_emb
    pass


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.num_heads = config['num_attention_heads']

        self.attention = MultiHeadAttention(self.hidden_size, self.num_heads)
        self.linear_out = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_out = nn.GELU()
        self.dropout_out = nn.Dropout(config['dropout_layer_out'])
        self.norm_out = nn.LayerNorm(
            self.hidden_size)
        
        pass

    def forward(
            self, flat_index, flat_mask, node_emb, batch_size, max_seq_len):

        node_flat = torch.zeros(
            (batch_size, max_seq_len, self.hidden_size),
            device=node_emb.device)
        
        node_flat[flat_index] = node_emb
        # xquery = self.linear_query(node_flat)
        # xkey = self.linear_key(node_flat)
        # xvalue = self.linear_value(node_flat)
        node_flat_new = self.attention(
            node_flat,
            node_flat,
            node_flat,
            flat_mask)
        # att_mask = (mask_flat[:, None, None, :] - 1) * 1e9
        # node_flat_new = self.bert_layer(
        #     node_flat,
        #     attention_mask=att_mask)[0]
        
        node_emb_new = node_flat_new[flat_index]

        node_emb_new = self.linear_out(node_emb_new)
        node_emb_new = self.act_out(node_emb_new)
        node_emb_new = self.dropout_out(node_emb_new)
        node_emb_new += node_emb
        node_emb_new = self.norm_out(node_emb_new)

        return node_emb_new

    pass


class TransformerLayerFromBert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert_layer = bert_mogai.BertLayer(
            transformers.BertConfig.from_dict(config))
        # self.hidden_size = config['hidden_size']
        # self.num_heads = config['num_attention_heads']
        # 
        # self.attention = MultiHeadAttention(self.hidden_size, self.num_heads)
        # self.linear_out = nn.Linear(self.hidden_size, self.hidden_size)
        # self.act_out = nn.GELU()
        # self.dropout_out = nn.Dropout(config['dropout_layer_out'])
        # self.norm_out = nn.LayerNorm(
        #     self.hidden_size)

        self.hidden_size = config['hidden_size']
        pass

    def forward(
            self, flat_index, flat_mask, node_emb, batch_size, max_seq_len,
            edge_embedding=None):

        node_flat = torch.zeros(
            (batch_size, max_seq_len, self.hidden_size),
            device=node_emb.device)
        
        node_flat[flat_index] = node_emb
        
        att_mask = (flat_mask[:, None, None, :] - 1) * 1e9
        
        bert_result = self.bert_layer(
            node_flat,
            attention_mask=att_mask,
            edge_embedding=edge_embedding)
        
        node_flat_new = bert_result[0]
        node_emb_new = node_flat_new[flat_index]

        return (node_emb_new, ) + bert_result[1:]

    pass


class GNNLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config['hidden_size'] // config['num_attention_heads']
        
        self.layer_embed_path_length = nn.Sequential(
            nn.Embedding(
                config['max_path_length'], head_size, padding_idx=0),
            nn.LayerNorm(head_size)
        )
        
        self.layer_embed_same_ring_count = nn.Sequential(
            nn.Embedding(
                config['max_same_ring'], head_size, padding_idx=0),
            nn.LayerNorm(head_size)
        )

        self.layer_bond_embed_cate = nn.Sequential(
            CateFeatureEncoder(
                head_size, config['nums_bond_feat_cate']),
            nn.LayerNorm(head_size)
            # nn.Dropout(config['dropout_input'])
        )
        
        self.layer_bond_embed_float = nn.Sequential(
            nn.Linear(
                config['num_bond_feat_float'], head_size),
            nn.LayerNorm(head_size)
            # nn.Dropout(config['dropout_input'])
        )

        self.layer_bert_atom = bert_mogai.BertLayer(
            transformers.BertConfig.from_dict(config))

        self.layer_bert_bond = bert_mogai.BertLayer(
            transformers.BertConfig.from_dict(config))

        pass

    def batch_index_select(self, hidden, index):
        step = hidden.shape[1]

        batch_start = torch.arange(
            hidden.shape[0], device=hidden.device).reshape(
                -1, 1, 1, 1) * step

        index = index + batch_start

        return torch.mean(
            torch.reshape(hidden, (-1, hidden.shape[-1]))[index], dim=-2)
        pass
    
    def forward(
            self, hidden_node, node_mask, bond_feat_cate, bond_feat_float,
            edge_index_aa_flat, edge_aa_bond_flat, shortest_path_length,
            same_ring_count, paths2
    ):
        batch_size = hidden_node.shape[0]
        max_node = hidden_node.shape[1]
        
        hidden_bond = self.layer_bond_embed_cate(bond_feat_cate)
        hidden_bond = hidden_bond + \
            self.layer_bond_embed_float(bond_feat_float)

        edge_emb = torch.zeros(
            (batch_size, max_node, max_node, hidden_bond.shape[-1]),
            device=hidden_node.device)
        
        edge_emb[tuple(edge_index_aa_flat)] = hidden_bond[
            tuple(edge_aa_bond_flat)]

        # paths2_mask = torch.any(paths2 == -1, dim=-1)
        # paths2_emb = self.batch_index_select(hidden_bond, paths2)
        # paths2_emb[paths2_mask] = 0

        edge_emb[:, 1:, 1:] += self.layer_embed_path_length(
            shortest_path_length) + self.layer_embed_same_ring_count(
                same_ring_count)
                # + paths2_emb
        
        node_mask = (node_mask[:, None, None, :] - 1) * 1e9
        
        hidden_node = self.layer_bert_atom(
            hidden_node,
            attention_mask=node_mask,
            edge_embedding=edge_emb)[0]

        # max_bond = hidden_bond.shape[1]
        # edge_bb = torch.zeros(
        #     (batch_size, max_bond, max_bond, hidden_node.shape[-1]))
        # edge_bb[tuple(edge_index_bb_flat)] = hidden_node[
        #     tuple(edge_bb_atom_flat)]
        # 
        # hidden_bond = self.layer_bert_bond(
        #     hidden_bond,
        #     attention_mask=bond_mask,
        #     edge_embedding=edge_bb)[0]

        return hidden_node
    pass


class MoleculeNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        
        self.layer_atom_embed_cate = nn.Sequential(
            CateFeatureEncoder(
                config['hidden_size'], config['nums_atom_feat_cate']),
            nn.LayerNorm(config['hidden_size'])
            # nn.Dropout(config['dropout_input'])
        )
        
        self.layer_atom_embed_float = nn.Sequential(
            nn.Linear(
                config['num_atom_feat_float'], config['hidden_size']),
            nn.LayerNorm(config['hidden_size'])
            # nn.Dropout(config['dropout_input'])
        )

        self.layer_graph_embed_cate = nn.Sequential(
            CateFeatureEncoder(
                config['hidden_size'], config['nums_graph_feat_cate']),
            nn.LayerNorm(config['hidden_size'])
            # nn.Dropout(config['dropout_input'])
        )

        self.layer_graph_fingerprint = nn.Sequential(
            nn.Linear(
                config['num_graph_fingerprint'], config['hidden_size']),
            # nn.Dropout(config['dropout_input'])
            nn.LayerNorm(config['hidden_size'])
        )
        
        self.spread_layers = nn.ModuleList(
            [
                GNNLayer(config)
                for _ in range(config['num_hidden_layers'])
            ]
        )

        self.layer_edge_map_ab = EmbeddingSkipNegative(
            2, config['hidden_size'] // config['num_attention_heads'])

        self.layer_edge_map_aa = nn.Linear(
            config['hidden_size'],
            config['hidden_size'] // config['num_attention_heads'])

        pass

    def forward(
            self, atom_feat_cate, atom_feat_float, atom_mask,
            bond_feat_cate, bond_feat_float, bond_mask,
            edge_index_ab, edge_mask_ab, edge_index_aa, edge_mask_aa,
            edge_aa_bond,
            shortest_path_length, same_ring_count, paths2,
            # edge_index_bb, edge_mask_bb, edge_bb_atom,
            graph_feat_cate, graph_fingerprint):

        device = atom_feat_cate.device
        # atom_feat_cate = atom_feat_cate[:, :, [0, 1, 2, 12, 4, 5, 6, 7, 8]]
        hidden_atom = self.layer_atom_embed_cate(atom_feat_cate)
        hidden_atom = hidden_atom + \
            self.layer_atom_embed_float(atom_feat_float)
        
        hidden_graph = self.layer_graph_embed_cate(graph_feat_cate)
        hidden_graph = hidden_graph + \
            self.layer_graph_fingerprint(graph_fingerprint)

        max_atom_num = hidden_atom.shape[1]

        max_node = 1 + max_atom_num
        batch_size = hidden_atom.shape[0]

        hidden_node = torch.zeros(
            (batch_size, max_node, self.hidden_size),
            device=device)
        
        # hidden_node[:, 0, :] = hidden_graph
        hidden_node[:, 1: (1+max_atom_num), :] = hidden_atom

        node_mask = torch.cat(
            (torch.ones((batch_size, 1), device=device),
             atom_mask), dim=-1)

        max_edge_num = edge_index_ab.shape[-1]
        # max_edge_bond = edge_index_bb.shape[-1]
        
        batch_index = torch.arange(
            batch_size, device=device).unsqueeze(1).unsqueeze(1)
        
        batch_index_ab = batch_index.repeat(
            [1, 1, max_edge_num])
        # batch_index_bb = batch_index.repeat(
        #     [1, 1, max_edge_bond])

        edge_index_ab = torch.cat((batch_index_ab, edge_index_ab), dim=-2)
        edge_index_aa = torch.cat((batch_index_ab, edge_index_aa), dim=-2)
        # edge_index_bb = torch.cat((batch_index_bb, edge_index_bb), dim=-2)

        edge_index_ab = torch.transpose(edge_index_ab, 1, 2)
        edge_index_aa = torch.transpose(edge_index_aa, 1, 2)
        # edge_index_bb = torch.transpose(edge_index_bb, 1, 2)

        edge_index_ab_flat = edge_index_ab[edge_mask_ab > 0, :].T
        edge_index_aa_flat = edge_index_aa[edge_mask_aa > 0, :].T
        # edge_index_bb_flat = edge_index_bb[edge_mask_bb > 0, :].T

        edge_index_ab_flat += torch.tensor(
            [0, 1, 0], dtype=torch.int64,
            device=device).reshape(-1, 1)

        edge_index_aa_flat += torch.tensor(
            [0, 1, 1], dtype=torch.int64,
            device=device).reshape(-1, 1)

        # edge_ab_type = -torch.ones(
        #     (batch_size, max_node, max_node), dtype=torch.int64,
        #     device=device)
        # 
        # edge_ab_type[tuple(edge_index_ab_flat)] = 0
        # edge_ab_type[tuple(edge_index_ab_flat[[0, 2, 1]])] = 1
        # edge_emb = self.layer_edge_map_ab(edge_ab_type)

        edge_aa_bond = torch.stack(
            (batch_index_ab.squeeze(1), edge_aa_bond), dim=-1)
        edge_aa_bond_flat = edge_aa_bond[edge_mask_aa > 0].T

        # edge_bb_atom = torch.stack(
        #     (batch_index_bb.squeeze(1), edge_bb_atom+1), dim=-1)
        # edge_bb_atom_flat = edge_bb_atom[edge_mask_bb > 0].T

        for i in range(len(self.spread_layers)):
            hidden_node = self.spread_layers[i](
                hidden_node, node_mask, bond_feat_cate, bond_feat_float,
                edge_index_aa_flat, edge_aa_bond_flat,
                shortest_path_length,
                same_ring_count, paths2
            )
            pass

        return hidden_node
        pass
