import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F
import torch
from reformer_pytorch.reformer_pytorch import LSHSelfAttention


# N-GCN
class nestGCN(nn.Module):
    def __init__(self):
        super(nestGCN,self).__init__()
        emb_dim = 128
        hidden_dim = emb_dim
        self.emb_dim = emb_dim
        k = 8
        self.hidden_dim = hidden_dim
        self.conv1 = dglnn.GATConv(emb_dim, hidden_dim,num_heads=1,allow_zero_in_degree=True,bias=True).cuda()
        self.conv2 = dglnn.GATConv(hidden_dim, hidden_dim,num_heads=1, allow_zero_in_degree=True,bias=True).cuda()
        self.conv3 = dglnn.GATConv(k*hidden_dim, hidden_dim,num_heads=1, allow_zero_in_degree=True, bias=True).cuda()
        self.conv4 = dglnn.GATConv(hidden_dim, hidden_dim,num_heads=1, allow_zero_in_degree=True, bias=True).cuda()
        self.conv5 = dglnn.GATConv(hidden_dim, hidden_dim, num_heads=1, allow_zero_in_degree=True, bias=True).cuda()
        self.conv6 = dglnn.GATConv(hidden_dim, hidden_dim, num_heads=1, allow_zero_in_degree=True, bias=True).cuda()
        self.conv7 = dglnn.GATConv(hidden_dim, hidden_dim, num_heads=1, allow_zero_in_degree=True, bias=True).cuda()

        # torch.nn.init.xavier_normal_(self.conv2.weight, 0.1)
        # torch.nn.init.xavier_normal_(self.conv1.weight, 0.1)
        # torch.nn.init.xavier_normal_(self.conv3.weight, 0.1)
        # torch.nn.init.xavier_normal_(self.conv4.weight, 0.1)
        # self.gat = dglnn.GATConv(hidden_dim,2,num_heads=3)
        # self.conv1 = dglnn.GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True,bias=True).cuda()
        # self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True,bias=True).cuda()
        self.l = nn.Linear(k*hidden_dim, hidden_dim).cuda()
        torch.nn.init.xavier_normal_(self.l.weight, 10)
        self.linear = nn.Linear(hidden_dim, hidden_dim).cuda()
        torch.nn.init.xavier_normal_(self.linear.weight, 10)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim).cuda()
        torch.nn.init.xavier_normal_(self.linear1.weight, 10)
        self.clasifier = nn.Linear(hidden_dim,2).cuda()
        torch.nn.init.xavier_normal_(self.clasifier.weight, 10)
        self.token_emb = nn.Embedding(150, embedding_dim=emb_dim).cuda()
        torch.nn.init.xavier_normal_(self.token_emb.weight, 10)
        self.linear_forward = nn.Linear(hidden_dim, hidden_dim)
        self.linear_forward1 = nn.Linear(hidden_dim, hidden_dim)
        torch.nn.init.xavier_normal_(self.linear_forward.weight, 3)
        torch.nn.init.xavier_normal_(self.linear_forward1.weight, 3)
        self.sortpool = dglnn.SortPooling(k)
        # torch.nn.init.xavier_normal_(self.sortpool.weight, 10)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self,fg,g,h):
        # fg:file_graphs g:func_graphs h:func_attrs

        h = self.token_emb(h)
        h = F.relu(h)
        h = h.view(-1,self.emb_dim)
        h = self.conv1(g, h)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        # h = self.conv5(g, h)
        # h = F.relu(h)
        # h = self.conv6(g, h)
        # h = F.relu(h)
        # h = self.conv7(g, h)
        # h = F.relu(h)

        h = h.view(-1,self.hidden_dim)
        h = self.linear_forward(h)
        h = F.relu(h)
        h = self.sortpool(g,h)  # func_num * (k * hidden_dim)

        h = self.conv3(fg, h)
        h = F.relu(h)
        # h = self.conv4(fg, h)
        # h = F.relu(h)
        # h = self.conv5(fg, h)
        # h = F.relu(h)
        # h = self.l(h)
        # h = F.relu(h)
        h = self.linear(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.linear1(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.clasifier(h)
        return h.view(-1,2)


#MRN-GCN(w/o sa)
class nestEGCN(nn.Module):
    def __init__(self):
        super(nestEGCN,self).__init__()
        num_heads = 1
        emb_dim = 128
        self.emb_dim = emb_dim
        k = 32
        num_layers = 8
        hidden_dim = emb_dim
        self.emb_dim = emb_dim

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gconvs = []
        self.gconvs.append(dglnn.EGATConv(emb_dim, hidden_dim,num_heads=num_heads,allow_zero_in_degree=True,bias=True).cuda())
        for i in range(num_layers-1):
            self.gconvs.append(
                dglnn.EGATConv(hidden_dim*num_heads, hidden_dim, num_heads=num_heads, allow_zero_in_degree=True, bias=True).cuda())

        self.conv3 = dglnn.GATConv(k*hidden_dim, hidden_dim,num_heads=1, allow_zero_in_degree=True, bias=True).cuda()
        self.conv4 = dglnn.GATConv(hidden_dim, hidden_dim,num_heads=1, allow_zero_in_degree=True, bias=True).cuda()
        self.linear = nn.Linear(hidden_dim, hidden_dim).cuda()
        torch.nn.init.xavier_normal_(self.linear.weight, 10)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim).cuda()
        torch.nn.init.xavier_normal_(self.linear1.weight, 10)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim).cuda()
        torch.nn.init.xavier_normal_(self.linear1.weight, 10)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim).cuda()
        torch.nn.init.xavier_normal_(self.linear1.weight, 10)
        self.clasifier = nn.Linear(hidden_dim,2).cuda()
        torch.nn.init.xavier_normal_(self.clasifier.weight, 10)
        self.token_emb = nn.Embedding(150, embedding_dim=emb_dim).cuda()
        torch.nn.init.xavier_normal_(self.token_emb.weight, 10)
        self.e_token_emb = nn.Embedding(100, embedding_dim=emb_dim).cuda()
        torch.nn.init.xavier_normal_(self.token_emb.weight, 10)

        self.linear_forward = nn.Linear(hidden_dim, hidden_dim)
        self.linear_forward1 = nn.Linear(hidden_dim, hidden_dim)
        torch.nn.init.xavier_normal_(self.linear_forward.weight, 3)
        torch.nn.init.xavier_normal_(self.linear_forward1.weight, 3)

        self.sortpool = dglnn.SortPooling(k)
        # torch.nn.init.xavier_normal_(self.sortpool.weight, 10)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self,fg,g,h,e):
        # fg:file_graphs g:func_graphs h:func_attrs
        #embeddings
        h = self.token_emb(h)
        h = F.relu(h)
        h = h.view(-1,1,self.emb_dim)
        e = self.e_token_emb(e)
        e = e.view(-1,1,self.emb_dim)

        #gconvs
        for i in range(self.num_layers):
            h,e = self.gconvs[i](g, h, e)
            h = F.relu(h)
            h = torch.squeeze(h,1)
            e = torch.unsqueeze(e, 1)

        h = h.view(-1,self.hidden_dim)

        h = self.linear_forward(h)
        h = F.relu(h)
        # h = self.linear_forward1(h)
        # h = F.relu(h)

        h = self.sortpool(g,h)

        h = self.conv3(fg, h)
        h = F.relu(h)
        # h = self.conv4(fg, h)
        # h = F.relu(h)

        h = self.linear(h)
        h = F.relu(h)
        h = self.dropout(h)
        # h = self.linear1(h)
        # h = F.relu(h)
        # h = self.dropout(h)
        # h = self.linear2(h)
        # h = F.relu(h)
        # h = self.dropout(h)
        # h = self.linear3(h)
        # h = F.relu(h)
        # h = self.dropout(h)
        h = self.clasifier(h)
        return h.view(-1,2)


#MRN-GCN
class nestEGCNs(nn.Module):
    def __init__(self):
        super(nestEGCNs,self).__init__()
        num_heads = 1
        emb_dim = 64
        hidden_dim = 64
        self.emb_dim = emb_dim
        k = 16
        num_layers = 8
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # stack of gconv layers
        self.gconvs = []
        self.gconvs.append(dglnn.EGATConv(emb_dim, hidden_dim,num_heads=num_heads,allow_zero_in_degree=True,bias=True).cuda())
        for i in range(num_layers-1):
            self.gconvs.append(
                dglnn.EGATConv(hidden_dim*num_heads, hidden_dim, num_heads=num_heads, allow_zero_in_degree=True, bias=True).cuda())
        self.self_attention = LSHSelfAttention(self.num_layers*self.hidden_dim, heads=4, bucket_size = 16, n_hashes = 16, post_attn_dropout=0., dropout=0.,
                                               n_local_attn_heads=0, causal=True, dim_head=None, attn_chunks=1,
                                               random_rotations_per_head=False, attend_across_buckets=True,
                                               allow_duplicate_attention=True, num_mem_kv=0, one_value_head=False,
                                               use_full_attn=True, full_attn_thres=None, return_attn=False)
        # self.lsh_attention = LSHSelfAttention(num_layers, heads=4, bucket_size=16, n_hashes=16,
        #                                       post_attn_dropout=0., dropout=0.,
        #                                       n_local_attn_heads=0, causal=True, dim_head=None, attn_chunks=1,
        #                                       random_rotations_per_head=False, attend_across_buckets=True,
        #                                       allow_duplicate_attention=True, num_mem_kv=0, one_value_head=False,
        #                                       use_full_attn=False, full_attn_thres=None, return_attn=False)

        self.linear_forward = nn.Linear(hidden_dim * (num_layers), hidden_dim)
        self.linear_forward1 = nn.Linear(self.num_layers*self.hidden_dim, hidden_dim)
        self.linear_forward2 = nn.Linear(hidden_dim, hidden_dim)
        torch.nn.init.xavier_normal_(self.linear_forward.weight, 3)
        torch.nn.init.xavier_normal_(self.linear_forward1.weight, 3)
        torch.nn.init.xavier_normal_(self.linear_forward2.weight, 3)

        #sa
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=hidden_dim, kernel_size=(1, hidden_dim * (num_layers)),
                               bias=True)
        torch.nn.init.xavier_normal_(self.conv1.weight, 3)

        self.conv3 = dglnn.GATConv(k*hidden_dim, hidden_dim,num_heads=1, allow_zero_in_degree=True, bias=True).cuda()
        self.conv4 = dglnn.GATConv(hidden_dim, hidden_dim,num_heads=1, allow_zero_in_degree=True, bias=True).cuda()
        self.linear = nn.Linear(hidden_dim, hidden_dim).cuda()
        torch.nn.init.xavier_normal_(self.linear.weight, 3)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim).cuda()
        torch.nn.init.xavier_normal_(self.linear1.weight, 3)
        self.clasifier = nn.Linear(hidden_dim,2).cuda()
        torch.nn.init.xavier_normal_(self.clasifier.weight, 3)
        self.token_emb = nn.Embedding(150, embedding_dim=emb_dim).cuda()
        torch.nn.init.xavier_normal_(self.token_emb.weight, 3)
        self.e_token_emb = nn.Embedding(100, embedding_dim=emb_dim).cuda()
        torch.nn.init.xavier_normal_(self.token_emb.weight, 3)

        self.l = nn.Linear(hidden_dim,1).cuda()

        linear_inter = nn.Linear(self.num_layers*self.hidden_dim, hidden_dim, bias=True)
        torch.nn.init.xavier_normal_(linear_inter.weight, 5)

        linear_inter2 = nn.Linear(hidden_dim,self.num_layers*self.hidden_dim, bias=True)
        torch.nn.init.xavier_normal_(linear_inter.weight, 5)
        self.linear_inter = nn.Sequential(linear_inter, nn.GELU(), linear_inter2, nn.GELU())

        self.sortpool = dglnn.SortPooling(k)
        # torch.nn.init.xavier_normal_(self.sortpool.weight, 10)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self,fg,g,h,e):
        # fg:file_graphs g:func_graphs h:func_attrs
        #embeddings
        h = self.token_emb(h)
        h = F.relu(h)
        h = h.view(-1,1,self.emb_dim)
        e = self.e_token_emb(e)
        e = e.view(-1,1,self.emb_dim)

        #gconvs
        h,e = self.gconvs[0](g, h, e)
        h = F.relu(h)
        h = torch.squeeze(h, 1)
        e = torch.unsqueeze(e, 1)
        hs = h
        for i in range(1,self.num_layers):
            h,e = self.gconvs[i](g, h, e)
            h = F.relu(h)
            h = torch.squeeze(h,1)
            e = torch.unsqueeze(e, 1)
            hs = torch.cat((hs,h),-1)    # (-1,1,layer_num)

        #self-attentions
        # hs = hs.view(-1,self.hidden_dim*(self.num_layers))
        #hs = hs.view(-1, self.num_layers)       # (-1,layer_num)
        # hs = torch.unsqueeze(hs,1)
        # h = self.conv1(hs)
        # h = F.relu(h)
        #h = self.self_attention(hs)
        for i in range(1):
            h = self.self_attention(hs)
            h = h.view(-1, self.num_layers*self.hidden_dim)
            h = self.linear_inter(h)
            h = h.view(-1, 1, self.num_layers*self.hidden_dim)

        h = h.view(-1,self.num_layers*self.hidden_dim)

        # h = self.linear_forward(hs)
        # h = F.relu(h)
        h = self.linear_forward1(h)
        h = F.relu(h)
        h = self.sortpool(g,h)

        h = self.conv3(fg, h)
        h = F.relu(h)
        # h = self.conv4(fg, h)
        # h = F.relu(h)

        h = self.linear(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.linear1(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.clasifier(h)
        return h.view(-1,2)

