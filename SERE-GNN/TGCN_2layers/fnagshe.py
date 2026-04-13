class GCNBert(nn.Module):
    def __init__(self, bert, opt, num_layers):
        super(GCNBert, self).__init__()
        self.bert = bert
        self.opt = opt
        self.layers = num_layers
        self.mem_dim = opt.bert_dim // 2
        self.attention_heads = opt.attention_heads
        self.bert_dim = opt.bert_dim
        self.bert_drop = nn.Dropout(opt.bert_dropout)
        self.pooled_drop = nn.Dropout(opt.bert_dropout)
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)
        self.layernorm = LayerNorm(opt.bert_dim)

        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.bert_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

        self.attn = MultiHeadAttention(self.opt, opt.attention_heads, self.bert_dim)
        self.wa = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.bert_dim if layer == 0 else self.mem_dim
            self.wa.append(nn.Linear(input_dim, self.mem_dim))

        self.ws = nn.ModuleList()
        for j in range(self.layers):
            input_dim = self.bert_dim if j == 0 else self.mem_dim
            self.ws.append(nn.Linear(input_dim, self.mem_dim))

        self.affine1 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))#affine1 被初始化为一个大小为 (self.mem_dim, self.mem_dim) 的张量，其中 self.mem_dim 是图卷积层的输出特征维度（即 opt.bert_dim // 2）。
        self.affine2 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, attention_mask, asp_start, asp_end, adj_reshape, src_mask, aspect_mask = inputs
        src_mask = src_mask.unsqueeze(-2) 
        batch = src_mask.size(0)
        len = src_mask.size()[2]
        
        sequence_output, pooled_output = self.bert(text_bert_indices, attention_mask=attention_mask, token_type_ids=bert_segments_ids).values()
        sequence_output = self.layernorm(sequence_output)
        gcn_inputs = self.bert_drop(sequence_output)
        pooled_output = self.pooled_drop(pooled_output)

        aspect_mask = aspect_mask.unsqueeze(-1).repeat(1, 1, self.bert_dim)
        aspect_outs = gcn_inputs*aspect_mask

        aspect_scores, s_attn = self.attn(gcn_inputs, gcn_inputs, src_mask, aspect_outs, aspect_mask)
        aspect_score_list = [attn_adj.squeeze(1) for attn_adj in torch.split(aspect_scores, 1, dim=1)]
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(s_attn, 1, dim=1)]
        adj_ag = None

        aspect_score_avg = None
        adj_s = None

        # Average Aspect-aware Attention scores
        for i in range(self.attention_heads):
            if aspect_score_avg is None:
                aspect_score_avg = aspect_score_list[i]
            else:
                aspect_score_avg += aspect_score_list[i]
        aspect_score_avg = aspect_score_avg / self.attention_heads

        # * Average Multi-head Attention matrices
        for i in range(self.attention_heads):
            if adj_s is None:
                adj_s = attn_adj_list[i]
            else:
                adj_s += attn_adj_list[i]
        adj_s = adj_s / self.attention_heads

        for j in range(adj_s.size(0)):
            adj_s[j] -= torch.diag(torch.diag(adj_s[j]))
            adj_s[j] += torch.eye(adj_s[j].size(0)).cuda()  # self-loop
        adj_s = src_mask.transpose(1, 2) * adj_s

        # distance based weighted matrix
        adj_reshape = torch.exp((-1.0) * self.opt.alpha * adj_reshape)

        # aspect-aware attention * distance based weighted matrix
        distance_mask = (
                    aspect_score_avg > torch.ones_like(aspect_score_avg) * self.opt.beta)
        adj_reshape = adj_reshape.masked_fill(distance_mask, 1).cuda()
        adj_ag = (adj_reshape * aspect_score_avg).type(torch.float32)

        # KL divergence
        kl_loss = F.kl_div(adj_ag.softmax(-1).log(), adj_s.softmax(-1), reduction='sum')
        kl_loss = torch.exp((-1.0) * kl_loss * self.opt.gama)

        # gcn layer
        denom_s = adj_s.sum(2).unsqueeze(2) + 1
        denom_ag = adj_ag.sum(2).unsqueeze(2) + 1
        outputs_s = gcn_inputs
        outputs_ag = gcn_inputs

        for l in range(self.layers):
            Ax_ag = adj_ag.bmm(outputs_ag)
            AxW_ag = self.wa[l](Ax_ag)
            AxW_ag = AxW_ag / denom_ag
            gAxW_ag = F.relu(AxW_ag)

            Ax_s = adj_s.bmm(outputs_s)
            AxW_s = self.ws[l](Ax_s)
            AxW_s = AxW_s / denom_s
            gAxW_s = F.relu(AxW_s)

            # * mutual Biaffine module
            A1 = F.softmax(torch.bmm(torch.matmul(gAxW_ag, self.affine1), torch.transpose(gAxW_s, 1, 2)), dim=-1)
            A2 = F.softmax(torch.bmm(torch.matmul(gAxW_s, self.affine2), torch.transpose(gAxW_ag, 1, 2)), dim=-1)
            gAxW_ag, gAxW_s = torch.bmm(A1, gAxW_s), torch.bmm(A2, gAxW_ag)
            outputs_ag = self.gcn_drop(gAxW_ag) if l < self.layers - 1 else gAxW_ag
            outputs_s = self.gcn_drop(gAxW_s) if l < self.layers - 1 else gAxW_s

        return outputs_ag, outputs_s, kl_loss, pooled_output

