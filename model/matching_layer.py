import torch
import torch.nn as nn
import torch.nn.functional as F

def mp_matching_func(v1, v2, w):
    """
    :param v1: (batch, seq_len, hidden_size)
    :param v2: (batch, seq_len, hidden_size) or (batch, hidden_size)
    :param w: (num_perspective, hidden_size)
    :return: (batch, num_perspective)
    """
    seq_len = v1.size(1)
    num_perspective = w.size()[0]

    # Trick for large memory requirement
    """
    if len(v2.size()) == 2:
        v2 = torch.stack([v2] * seq_len, dim=1)

    m = []
    for i in range(self.num_perspective):
        # v1: (batch, seq_len, hidden_size)
        # v2: (batch, seq_len, hidden_size)
        # w: (1, 1, hidden_size)
        # -> (batch, seq_len)
        m.append(F.cosine_similarity(w[i].view(1, 1, -1) * v1, w[i].view(1, 1, -1) * v2, dim=2))

    # list of (batch, seq_len) -> (batch, seq_len, num_perspective)
    m = torch.stack(m, dim=2)
    """

    # (1, 1, hidden_size, num_perspective)
    w = w.transpose(1, 0).unsqueeze(0).unsqueeze(0)
    # (batch, seq_len, hidden_size, num_perspective)
    v1 = w * torch.stack([v1] * num_perspective, dim=3)
    if len(v2.size()) == 3:
        v2 = w * torch.stack([v2] * num_perspective, dim=3)
    else:
        v2 = w * torch.stack([torch.stack([v2] * seq_len, dim=1)] * num_perspective, dim=3)

    m = F.cosine_similarity(v1, v2, dim=2)

    return m

def mp_matching_func_pairwise(v1, v2, w):
    """
    :param v1: (batch, seq_len1, hidden_size)
    :param v2: (batch, seq_len2, hidden_size)
    :param w: (num_perspective, hidden_size)
    :return: (batch, num_perspective, seq_len1, seq_len2)
    """

    # Trick for large memory requirement
    """
    m = []
    for i in range(self.num_perspective):
        # (1, 1, hidden_size)
        w_i = w[i].view(1, 1, -1)
        # (batch, seq_len1, hidden_size), (batch, seq_len2, hidden_size)
        v1, v2 = w_i * v1, w_i * v2
        # (batch, seq_len, hidden_size->1)
        v1_norm = v1.norm(p=2, dim=2, keepdim=True)
        v2_norm = v2.norm(p=2, dim=2, keepdim=True)

        # (batch, seq_len1, seq_len2)
        n = torch.matmul(v1, v2.permute(0, 2, 1))
        d = v1_norm * v2_norm.permute(0, 2, 1)

        m.append(div_with_small_value(n, d))

    # list of (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, num_perspective)
    m = torch.stack(m, dim=3)
    """
    
    num_perspective = w.size()[0]

    # (1, num_perspective, 1, hidden_size)
    w = w.unsqueeze(0).unsqueeze(2)
    # (batch, num_perspective, seq_len, hidden_size)
    v1, v2 = w * torch.stack([v1] * num_perspective, dim=1), w * torch.stack([v2] * num_perspective, dim=1)
    # (batch, num_perspective, seq_len, hidden_size->1)
    v1_norm = v1.norm(p=2, dim=3, keepdim=True)
    v2_norm = v2.norm(p=2, dim=3, keepdim=True)

    # (batch, num_perspective, seq_len1, seq_len2)
    n = torch.matmul(v1, v2.transpose(2, 3))
    d = v1_norm * v2_norm.transpose(2, 3)

    # (batch, seq_len1, seq_len2, num_perspective)
    m = div_with_small_value(n, d).permute(0, 2, 3, 1)

    return m

def attention(v1, v2):
    """
    :param v1: (batch, seq_len1, hidden_size)
    :param v2: (batch, seq_len2, hidden_size)
    :return: (batch, seq_len1, seq_len2)
    """

    # (batch, seq_len1, 1)
    v1_norm = v1.norm(p=2, dim=2, keepdim=True)
    # (batch, 1, seq_len2)
    v2_norm = v2.norm(p=2, dim=2, keepdim=True).permute(0, 2, 1)

    # (batch, seq_len1, seq_len2)
    a = torch.bmm(v1, v2.permute(0, 2, 1))
    d = v1_norm * v2_norm

    return div_with_small_value(a, d)

def div_with_small_value(n, d, eps=1e-8):
    # too small values are replaced by 1e-8 to prevent it from exploding.
    d = d * (d > eps).float() + eps * (d <= eps).float()
    return n / d
            
class MatchingLayer(nn.Module):
    def __init__(self, hidden_dim=100, num_perspective=20, dropout=0.1):
        super(MatchingLayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_perspective = num_perspective
        self.dropout = dropout
        
        for i in range(1, 9):
            setattr(self, f'mp_w{i}', nn.Parameter(torch.rand(num_perspective, hidden_dim)))
            w = getattr(self, f'mp_w{i}')
            nn.init.kaiming_normal_(w)
        
    def forward(self, con_p, con_h):
        # (batch, seq_len, hidden_dim)
        con_p_fw, con_p_bw = torch.split(con_p, self.hidden_dim, dim=-1)
        con_h_fw, con_h_bw = torch.split(con_h, self.hidden_dim, dim=-1)

        # 1. Full-Matching

        # (batch, seq_len, hidden_size), (batch, hidden_size)
        # -> (batch, seq_len, num_perspective)
        mv_p_full_fw = mp_matching_func(con_p_fw, con_h_fw[:, -1, :], self.mp_w1)
        mv_p_full_bw = mp_matching_func(con_p_bw, con_h_bw[:, 0, :], self.mp_w2)
        mv_h_full_fw = mp_matching_func(con_h_fw, con_p_fw[:, -1, :], self.mp_w1)
        mv_h_full_bw = mp_matching_func(con_h_bw, con_p_bw[:, 0, :], self.mp_w2)

        # 2. Maxpooling-Matching

        # (batch, seq_len1, seq_len2, num_perspective)
        mv_max_fw = mp_matching_func_pairwise(con_p_fw, con_h_fw, self.mp_w3)
        mv_max_bw = mp_matching_func_pairwise(con_p_bw, con_h_bw, self.mp_w4)

        # (batch, seq_len, num_perspective)
        mv_p_max_fw, _ = mv_max_fw.max(dim=2)
        mv_p_max_bw, _ = mv_max_bw.max(dim=2)
        mv_h_max_fw, _ = mv_max_fw.max(dim=1)
        mv_h_max_bw, _ = mv_max_bw.max(dim=1)

        # 3. Attentive-Matching

        # (batch, seq_len1, seq_len2)
        att_fw = attention(con_p_fw, con_h_fw)
        att_bw = attention(con_p_bw, con_h_bw)

        # (batch, seq_len2, hidden_size) -> (batch, 1, seq_len2, hidden_size)
        # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
        # -> (batch, seq_len1, seq_len2, hidden_size)
        att_h_fw = con_h_fw.unsqueeze(1) * att_fw.unsqueeze(3)
        att_h_bw = con_h_bw.unsqueeze(1) * att_bw.unsqueeze(3)
        # (batch, seq_len1, hidden_size) -> (batch, seq_len1, 1, hidden_size)
        # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
        # -> (batch, seq_len1, seq_len2, hidden_size)
        att_p_fw = con_p_fw.unsqueeze(2) * att_fw.unsqueeze(3)
        att_p_bw = con_p_bw.unsqueeze(2) * att_bw.unsqueeze(3)

        # (batch, seq_len1, hidden_size) / (batch, seq_len1, 1) -> 
        # (batch, seq_len1, hidden_size)
        att_mean_h_fw = div_with_small_value(att_h_fw.sum(dim=2), att_fw.sum(dim=2, keepdim=True))
        att_mean_h_bw = div_with_small_value(att_h_bw.sum(dim=2), att_bw.sum(dim=2, keepdim=True))

        # (batch, seq_len2, hidden_size) / (batch, seq_len2, 1) -> 
        # (batch, seq_len2, hidden_size)
        att_mean_p_fw = div_with_small_value(att_p_fw.sum(dim=1), att_fw.sum(dim=1, keepdim=True).permute(0, 2, 1))
        att_mean_p_bw = div_with_small_value(att_p_bw.sum(dim=1), att_bw.sum(dim=1, keepdim=True).permute(0, 2, 1))

        # (batch, seq_len, num_perspective)
        mv_p_att_mean_fw = mp_matching_func(con_p_fw, att_mean_h_fw, self.mp_w5)
        mv_p_att_mean_bw = mp_matching_func(con_p_bw, att_mean_h_bw, self.mp_w6)
        mv_h_att_mean_fw = mp_matching_func(con_h_fw, att_mean_p_fw, self.mp_w5)
        mv_h_att_mean_bw = mp_matching_func(con_h_bw, att_mean_p_bw, self.mp_w6)

        # 4. Max-Attentive-Matching

        # (batch, seq_len1, hidden_size)
        att_max_h_fw, _ = att_h_fw.max(dim=2)
        att_max_h_bw, _ = att_h_bw.max(dim=2)
        # (batch, seq_len2, hidden_size)
        att_max_p_fw, _ = att_p_fw.max(dim=1)
        att_max_p_bw, _ = att_p_bw.max(dim=1)

        # (batch, seq_len, num_perspective)
        mv_p_att_max_fw = mp_matching_func(con_p_fw, att_max_h_fw, self.mp_w7)
        mv_p_att_max_bw = mp_matching_func(con_p_bw, att_max_h_bw, self.mp_w8)
        mv_h_att_max_fw = mp_matching_func(con_h_fw, att_max_p_fw, self.mp_w7)
        mv_h_att_max_bw = mp_matching_func(con_h_bw, att_max_p_bw, self.mp_w8)

        # Lastly, concatenate the four matching results
        # (batch, seq_len, num_perspective * 8)
        mv_p = torch.cat(
            [mv_p_full_fw, mv_p_max_fw, mv_p_att_mean_fw, mv_p_att_max_fw,
             mv_p_full_bw, mv_p_max_bw, mv_p_att_mean_bw, mv_p_att_max_bw], dim=2)
        mv_h = torch.cat(
            [mv_h_full_fw, mv_h_max_fw, mv_h_att_mean_fw, mv_h_att_max_fw,
             mv_h_full_bw, mv_h_max_bw, mv_h_att_mean_bw, mv_h_att_max_bw], dim=2)

        mv_p = F.dropout(mv_p, p=self.dropout, training=self.training)
        mv_h = F.dropout(mv_h, p=self.dropout, training=self.training)
        
        return mv_p, mv_h

if __name__ == "__main__":
    torch.manual_seed(999)

    batch = 16
    seq_len = 50
    dim = 200
    test_input_p = torch.autograd.Variable(torch.randn(batch, seq_len, dim))
    test_input_h = torch.autograd.Variable(torch.randn(batch, seq_len, dim))
    l = 5

    ml = MatchingLayer(num_perspective=l)

    p_vecs, h_vecs = ml(test_input_p, test_input_h)

    assert p_vecs.size() == h_vecs.size() == torch.Size([batch, seq_len, 8*l])