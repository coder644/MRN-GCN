import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F
import dgl
import torch
from reformer_pytorch.reformer_pytorch import LSHSelfAttention


class egat(nn.Module):
    def __init__(self):
        super(egat,self).__init__()
        num_heads = 1
        emb_dim = 32
        hidden_dim = 32
        num_layers = 8
        self.emb_dim = emb_dim
        k = 8
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        # stack of gconv layers
        self.gconvs = []
        self.gconvs.append(dglnn.EGATConv(emb_dim, hidden_dim,num_heads=num_heads,allow_zero_in_degree=True,bias=True).cuda())
        for i in range(num_layers-1):
            self.gconvs.append(
                dglnn.EGATConv(hidden_dim*num_heads, hidden_dim, num_heads=num_heads, allow_zero_in_degree=True, bias=True).cuda())

        self.linear = nn.Linear(k*hidden_dim, hidden_dim).cuda()
        torch.nn.init.xavier_normal_(self.linear.weight, 5)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim).cuda()
        torch.nn.init.xavier_normal_(self.linear1.weight, 5)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim).cuda()
        torch.nn.init.xavier_normal_(self.linear1.weight, 5)
        self.clasifier = nn.Linear(hidden_dim,2).cuda()
        torch.nn.init.xavier_normal_(self.clasifier.weight, 5)

        self.token_emb = nn.Embedding(100, embedding_dim=emb_dim).cuda()
        torch.nn.init.xavier_normal_(self.token_emb.weight, 5)
        self.sortpool = dglnn.SortPooling(k)
        self.dropout = nn.Dropout(p=0.2)

        self.e_token_emb = nn.Embedding(100, embedding_dim=emb_dim).cuda()
        torch.nn.init.xavier_normal_(self.token_emb.weight, 5)
    def forward(self,g,h,e):

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
            # h = h.view(-1,self.num_heads*self.hidden_dim)
            h = torch.squeeze(h,1)
            e = torch.unsqueeze(e, 1)
        #
        #
        # h,e = self.conv2(g, h, e)
        # h = F.relu(h)
        # h = torch.squeeze(h,1)
        # e = torch.unsqueeze(e, 1)
        # h,e = self.conv3(g, h, e)
        # h = F.relu(h)
        # h = torch.squeeze(h,1)
        # h = self.conv4(g, h, e)
        # h = F.relu(h)
        # h = torch.squeeze(h,1)
        # h = self.conv5(g, h, e)
        # h = F.relu(h)
        # h = torch.squeeze(h,1)
        # h = self.conv6(g, h, e)
        # h = F.relu(h)
        # h = torch.squeeze(h,1)
        # h = self.conv7(g, h, e)
        # h = F.relu(h)
        # h = torch.squeeze(h,1)
        # h = self.conv8(g, h, e)
        # h = F.relu(h)

        h = h.view(-1,self.hidden_dim*self.num_heads)

        h = self.sortpool(g,h)
        h = self.linear(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.linear1(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.linear2(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.clasifier(h)
        return h


class egats(nn.Module):
    def __init__(self):
        super(egats,self).__init__()
        num_heads = 1
        emb_dim = 64
        hidden_dim = 64
        num_layers = 16
        self.emb_dim = emb_dim
        k = 16
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # stack of gconv layers
        self.gconvs = []
        self.gconvs.append(dglnn.EGATConv(emb_dim, hidden_dim,num_heads=num_heads,allow_zero_in_degree=True,bias=True).cuda())
        for i in range(num_layers-1):
            self.gconvs.append(
                dglnn.EGATConv(hidden_dim*num_heads, hidden_dim, num_heads=num_heads, allow_zero_in_degree=True, bias=True).cuda())

        self.self_attention = LSHSelfAttention(self.num_layers * self.hidden_dim, heads=4, bucket_size=16, n_hashes=16,
                                               post_attn_dropout=0., dropout=0.,
                                               n_local_attn_heads=0, causal=True, dim_head=None, attn_chunks=1,
                                               random_rotations_per_head=False, attend_across_buckets=True,
                                               allow_duplicate_attention=True, num_mem_kv=0, one_value_head=False,
                                               use_full_attn=True, full_attn_thres=None, return_attn=False)

        self.linear_forward = nn.Linear(hidden_dim*(1+num_layers),hidden_dim)
        self.linear_forward1 = nn.Linear(self.num_layers * hidden_dim,hidden_dim)
        self.linear_forward2 = nn.Linear(hidden_dim,hidden_dim)
        torch.nn.init.xavier_normal_(self.linear_forward.weight, 5)
        torch.nn.init.xavier_normal_(self.linear_forward1.weight, 5)
        torch.nn.init.xavier_normal_(self.linear_forward2.weight, 5)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=hidden_dim, kernel_size=(1, hidden_dim*(num_layers)), bias=True)
        torch.nn.init.xavier_normal_(self.conv1.weight, 5)

        linear_inter = nn.Linear(self.num_layers * self.hidden_dim, hidden_dim, bias=True)
        torch.nn.init.xavier_normal_(linear_inter.weight, 5)

        linear_inter2 = nn.Linear(hidden_dim, self.num_layers * self.hidden_dim, bias=True)
        torch.nn.init.xavier_normal_(linear_inter.weight, 5)
        self.linear_inter = nn.Sequential(linear_inter, nn.GELU(), linear_inter2, nn.GELU())

        self.linear = nn.Linear(k*hidden_dim, hidden_dim).cuda()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim).cuda()
        self.clasifier = nn.Linear(hidden_dim,2).cuda()
        torch.nn.init.xavier_normal_(self.linear.weight, 5)
        torch.nn.init.xavier_normal_(self.linear1.weight, 5)
        torch.nn.init.xavier_normal_(self.clasifier.weight, 5)


        self.token_emb = nn.Embedding(100, embedding_dim=emb_dim).cuda()
        self.e_token_emb = nn.Embedding(50, embedding_dim=emb_dim).cuda()
        torch.nn.init.xavier_normal_(self.token_emb.weight, 5)
        torch.nn.init.xavier_normal_(self.token_emb.weight, 5)

        self.sortpool = dglnn.SortPooling(k)
        self.dropout = nn.Dropout(p=0.2)


    def forward(self,g,h,e):

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
        for i in range(1, self.num_layers):
            h, e = self.gconvs[i](g, h, e)
            h = F.relu(h)
            h = torch.squeeze(h, 1)
            e = torch.unsqueeze(e, 1)
            hs = torch.cat((hs, h), -1)

        #hs = hs.view(-1,self.hidden_dim*(1+self.num_layers))
        for i in range(1):
            h = self.self_attention(hs)
            h = h.view(-1, self.num_layers * self.hidden_dim)
            h = self.linear_inter(h)
            h = h.view(-1, 1, self.num_layers * self.hidden_dim)

        h = h.view(-1, self.num_layers * self.hidden_dim)
        h = self.linear_forward1(h)
        h = F.relu(h)
        h = self.sortpool(g,h)
        h = self.linear(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.linear1(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.clasifier(h)
        return h
