import torch
import torch.nn as nn
import torch.nn.functional as F

from .matching_layer import MatchingLayer
eps = 1e-8

class BIMPM(nn.Module):
    def __init__(self, 
        class_size, word_vocab_size, char_vocab_size,
        pretrained_word_embedding=None, pretrained_char_embedding=None,
        num_perspective=20, word_dim=300, dropout=0.1, fix_word_vec=True,
        use_char_emb=True, char_dim=20, char_lstm_dim=50, fix_char_vec=False,
        context_lstm_dim=100, context_layer_num=2, 
        aggregation_lstm_dim=100, aggregation_layer_num=2, 
        wo_full_match=False, wo_maxpool_match=False, 
        wo_attentive_match=False, wo_max_attentive_match=False
    ):
        super(BIMPM, self).__init__()

        self.class_size = class_size
        self.use_char_emb = use_char_emb
        self.context_lstm_dim = context_lstm_dim
        self.context_layer_num = context_layer_num
        self.aggregation_lstm_dim = aggregation_lstm_dim
        self.aggregation_layer_num = aggregation_layer_num
        self.char_lstm_dim = char_lstm_dim
        self.dropout = dropout
        self.word_rep_dim = word_dim + int(use_char_emb) * char_lstm_dim
        self.num_perspective = num_perspective
        self.num_matching = 8 - 2 * (int(wo_full_match) + int(wo_maxpool_match) + int(wo_attentive_match) + int(wo_max_attentive_match))
        assert self.num_matching > 0

        # ----- Word Representation Layer -----
        assert pretrained_word_embedding is None or len(pretrained_word_embedding) == word_vocab_size
        self.word_emb = nn.Embedding(word_vocab_size, word_dim, padding_idx=1)
        if fix_word_vec:
            self.word_emb.weight.requires_grad = False

        if use_char_emb:
            assert pretrained_char_embedding is None or len(pretrained_char_embedding) == char_vocab_size
            self.char_emb = nn.Embedding(char_vocab_size, char_dim, padding_idx=1)
            if fix_char_vec:
                self.char_emb.weight.requires_grad = False

            self.char_LSTM = nn.LSTM(
                input_size=char_dim,
                hidden_size=char_lstm_dim,
                num_layers=1,
                bidirectional=False,
                batch_first=True)

        # ----- Context Representation Layer -----
        self.context_LSTM = nn.LSTM(
            input_size=self.word_rep_dim,
            hidden_size=context_lstm_dim,
            num_layers=self.context_layer_num,
            bidirectional=True,
            batch_first=True
        )

        # ----- Matching Layer -----
        self.matching_layer = MatchingLayer(
            hidden_dim=self.context_lstm_dim, 
            num_perspective=self.num_perspective, 
            dropout=self.dropout,
            wo_full_match=wo_full_match,
            wo_maxpool_match=wo_maxpool_match,
            wo_attentive_match=wo_attentive_match,
            wo_max_attentive_match=wo_max_attentive_match,
        )

        # ----- Aggregation Layer -----
        self.aggregation_LSTM = nn.LSTM(
            input_size=num_perspective * self.num_matching,
            hidden_size=aggregation_lstm_dim,
            num_layers=self.aggregation_layer_num,
            bidirectional=True,
            batch_first=True
        )

        # ----- Prediction Layer -----
        self.pred_fc1 = nn.Linear(
            self.aggregation_lstm_dim * 4 * self.aggregation_layer_num, 
            self.aggregation_lstm_dim * 2 * self.aggregation_layer_num
        )
        self.pred_fc2 = nn.Linear(
            self.aggregation_lstm_dim * 2 * self.aggregation_layer_num, 
            self.class_size
        )

        self.init_parameters(pretrained_word_embedding, pretrained_char_embedding)

    def init_parameters(self, pretrained_word_embedding, pretrained_char_embedding):
        # ----- Word Representation Layer -----

        # <unk> vectors is randomly initialized
        if pretrained_word_embedding is not None:
            self.word_emb.weight.data.copy_(pretrained_word_embedding)
            nn.init.uniform_(self.word_emb.weight.data[0], -0.005, 0.005)
            for data in self.word_emb.weight.data[2:]:
                if (data.abs() < eps).all():
                    nn.init.uniform_(data, -0.005, 0.005)
        else:
            nn.init.uniform_(self.word_emb.weight, -0.005, 0.005)
            
        self.word_emb.weight.data[1].fill_(0)

        if self.use_char_emb:
            # <unk> vectors is randomly initialized
            if pretrained_char_embedding is not None:
                self.char_emb.weight.data.copy_(pretrained_char_embedding)
                nn.init.uniform_(self.char_emb.weight.data[0], -0.005, 0.005)
                for data in self.char_emb.weight.data[2:]:
                    if (data.abs() < eps).all():
                        nn.init.uniform_(data, -0.005, 0.005)
            else:
                nn.init.uniform_(self.char_emb.weight, -0.005, 0.005)
                
            self.char_emb.weight.data[1].fill_(0)
            
            nn.init.kaiming_normal_(self.char_LSTM.weight_ih_l0)
            nn.init.constant_(self.char_LSTM.bias_ih_l0, val=0)
            nn.init.orthogonal_(self.char_LSTM.weight_hh_l0)
            nn.init.constant_(self.char_LSTM.bias_hh_l0, val=0)

        # ----- Context Representation Layer -----
        nn.init.kaiming_normal_(self.context_LSTM.weight_ih_l0)
        nn.init.constant_(self.context_LSTM.bias_ih_l0, val=0)
        nn.init.orthogonal_(self.context_LSTM.weight_hh_l0)
        nn.init.constant_(self.context_LSTM.bias_hh_l0, val=0)

        nn.init.kaiming_normal_(self.context_LSTM.weight_ih_l0_reverse)
        nn.init.constant_(self.context_LSTM.bias_ih_l0_reverse, val=0)
        nn.init.orthogonal_(self.context_LSTM.weight_hh_l0_reverse)
        nn.init.constant_(self.context_LSTM.bias_hh_l0_reverse, val=0)

        # ----- Aggregation Layer -----
        nn.init.kaiming_normal_(self.aggregation_LSTM.weight_ih_l0)
        nn.init.constant_(self.aggregation_LSTM.bias_ih_l0, val=0)
        nn.init.orthogonal_(self.aggregation_LSTM.weight_hh_l0)
        nn.init.constant_(self.aggregation_LSTM.bias_hh_l0, val=0)

        nn.init.kaiming_normal_(self.aggregation_LSTM.weight_ih_l0_reverse)
        nn.init.constant_(self.aggregation_LSTM.bias_ih_l0_reverse, val=0)
        nn.init.orthogonal_(self.aggregation_LSTM.weight_hh_l0_reverse)
        nn.init.constant_(self.aggregation_LSTM.bias_hh_l0_reverse, val=0)

        # ----- Prediction Layer ----
        nn.init.uniform_(self.pred_fc1.weight, -0.005, 0.005)
        nn.init.constant_(self.pred_fc1.bias, val=0)

        nn.init.uniform_(self.pred_fc2.weight, -0.005, 0.005)
        nn.init.constant_(self.pred_fc2.bias, val=0)
        
    def char_repr(self, char_data):
        # char_data: (batch, seq_len, max_word_len)
        seq_len, word_len = char_data.size(1), char_data.size(2)

        # (batch, seq_len, max_word_len) -> (batch * seq_len, max_word_len)
        char_data = char_data.view(-1, word_len)
        
        # (batch * seq_len, max_word_len) -> (batch * seq_len, max_word_len, char_dim)
        char_data = self.char_emb(char_data)

        # (batch * seq_len, max_word_len, char_dim)-> (1, batch * seq_len, char_lstm_dim)
        _, (char_data, _) = self.char_LSTM(char_data)

        # (batch, seq_len, char_lstm_dim)
        char_data = char_data.view(-1, seq_len, self.char_lstm_dim)
        
        return char_data
        
    def context_repr(self, word_data, char_data):
        # (batch, seq_len) -> (batch, seq_len, word_dim)
        context_data = self.word_emb(word_data)

        if self.use_char_emb:
            char_data = self.char_repr(char_data)
            
            # (batch, seq_len, word_dim + char_lstm_dim)
            context_data = torch.cat([context_data, char_data], dim=-1)

        context_data = F.dropout(context_data, p=self.dropout, training=self.training)

        # ----- Context Representation Layer -----
        # (batch, seq_len, context_lstm_dim * 2)
        context_data, _ = self.context_LSTM(context_data)

        context_data = F.dropout(context_data, p=self.dropout, training=self.training)
        
        return context_data

    def forward(self, **kwargs):

        # ----- Context Representation Layer -----
        p, h = kwargs['p'], kwargs['h']
        assert p.size(0) == h.size(0) 
        batch_size, seq_len_p, seq_len_h = p.size(0), p.size(1), h.size(1)
        
        if self.use_char_emb:
            char_p, char_h = kwargs['char_p'], kwargs['char_h']
            assert p.size(1) == char_p.size(1) and h.size(1) == char_h.size(1)
        else:
            char_p, char_h = None, None
        
        # (batch, seq_len) -> (batch, seq_len, context_lstm_dim * 2)
        con_p = self.context_repr(p, char_p)
        con_h = self.context_repr(h, char_h)
        assert con_p.size() == (batch_size, seq_len_p, self.context_lstm_dim * 2)
        assert con_h.size() == (batch_size, seq_len_h, self.context_lstm_dim * 2)

        # (batch, seq_len, context_lstm_dim * 2) -> (batch, seq_len, num_perspective * num_matching)
        mv_p, mv_h = self.matching_layer(con_p, con_h)
        assert mv_p.size() == (batch_size, seq_len_p, self.num_perspective * self.num_matching)
        assert mv_h.size() == (batch_size, seq_len_h, self.num_perspective * self.num_matching)

        # ----- Aggregation Layer -----
        # (batch, seq_len, num_perspective * num_matching) -> 
        # (2 * aggregation_layer_num, batch, aggregation_lstm_dim)
        _, (agg_p_last, _) = self.aggregation_LSTM(mv_p)
        _, (agg_h_last, _) = self.aggregation_LSTM(mv_h)
        assert agg_p_last.size() == agg_h_last.size() == (2 * self.aggregation_layer_num, batch_size, self.aggregation_lstm_dim)

        # 2 * (2 * aggregation_layer_num, batch, aggregation_lstm_dim) -> 
        # 2 * (batch, aggregation_lstm_dim * 2 * aggregation_layer_num) -> 
        # (batch, 2 * aggregation_lstm_dim * 2 * aggregation_layer_num)
        x = torch.cat(
            [agg_p_last.permute(1, 0, 2).contiguous().view(-1, self.aggregation_lstm_dim * 2 * self.aggregation_layer_num),
             agg_h_last.permute(1, 0, 2).contiguous().view(-1, self.aggregation_lstm_dim * 2 * self.aggregation_layer_num)], dim=1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        assert x.size() == (batch_size, 2 * self.aggregation_lstm_dim * 2 * self.aggregation_layer_num)

        # ----- Prediction Layer -----
        x = F.tanh(self.pred_fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        assert x.size() == (batch_size, self.aggregation_lstm_dim * 2 * self.aggregation_layer_num)
        
        x = self.pred_fc2(x)
        assert x.size() == (batch_size, self.class_size)

        return x
