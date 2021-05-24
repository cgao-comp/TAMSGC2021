'''
This code is based on the paper in AAAI 2019. There are two main improvements in the TAMSGC model as follows:

First, we designed the Temporal Attention Mechanism to replace the original RNN in the paper of AAAI 2019;
Second, we used the Simple Graph Convolution (SGC) to replace the Graph Convolutional Network (GCN) in the paper of AAAI 2019.

@inproceedings{shang2019gamenet,
  title={Gamenet: Graph augmented memory networks for recommending medication combination},
  author={Shang, Junyuan and Xiao, Cao and Ma, Tengfei and Li, Hongyan and Sun, Jimeng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  number={01},
  pages={1126--1133},
  year={2019}
}
'''

import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params


class TAMSGC(nn.Module):
    def __init__(self, size, ehr_adj, ddi_adj, emb_dim=64, device=torch.device('cpu'), ddi_in_memory=True):
        super(TAMSGC, self).__init__()
        K = len(size)
        self.K = K
        self.vocab_size = size
        self.device = device
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.ddi_in_memory = ddi_in_memory
        self.embeddings = nn.ModuleList(
            [nn.Embedding(size[i], emb_dim) for i in range(K-1)])
        self.dropout = nn.Dropout(p=0.3)
        self.encoders = nn.ModuleList(
            [nn.GRU(emb_dim, emb_dim*2) for _ in range(K - 1)])
        self.alpha = nn.Linear(emb_dim*2, 1)
        self.beta = nn.Linear(emb_dim*2, emb_dim)
        self.Reain_output = nn.Linear(emb_dim, emb_dim*2)
        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim*4, emb_dim),
        )
        self.ehr_gnn = SGC(size=size[2], emb_dim=emb_dim, adj=ehr_adj,  device=device)  # GNN for EHR
        self.ddi_gnn = SGC(size=size[2], emb_dim=emb_dim, adj=ddi_adj,  device=device)  # GNN for DDI
        self.lambda_ = nn.Parameter(torch.FloatTensor(1))
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, size[2])
        )

        self.init_weights()

    def forward(self, patient):

        # the embeddings of diagnosis and procedure
        diagnosis_seq = []
        procedure_seq = []
        def mean_embedding(embedding):
            return embedding.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)
        for admission_time in patient:
            i1 = mean_embedding(self.dropout(self.embeddings[0](torch.LongTensor(admission_time[0]).unsqueeze(dim=0).to(self.device)))) # (1,1,dim)
            i2 = mean_embedding(self.dropout(self.embeddings[1](torch.LongTensor(admission_time[1]).unsqueeze(dim=0).to(self.device))))
            diagnosis_seq.append(i1)
            procedure_seq.append(i2)
        diagnosis_seq = torch.cat(diagnosis_seq, dim=1)  #(1,seq,dim)
        procedure_seq = torch.cat(procedure_seq, dim=1)  #(1,seq,dim)

        # Temporal Attention Mechanism
        diagnosis_output, diagnosis_hidden = self.encoders[0](
            diagnosis_seq
        )  # for diagnosis
        attn_o1_alpha = F.tanhshrink(self.alpha(diagnosis_output))  # (visit, 1)
        attn_o1_beta = F.tanh(self.beta(diagnosis_output))
        diagnosis_output = attn_o1_alpha * attn_o1_beta * diagnosis_seq  # (visit, emb)
        diagnosis_output = torch.sum(diagnosis_output, dim=0).unsqueeze(dim=0)  # (1, emb)
        diagnosis_output = self.Reain_output(diagnosis_output)

        procedure_output, procedure_hidden = self.encoders[1](
            procedure_seq
        )  # for procedure
        attn_o2_alpha = F.tanhshrink(self.alpha(procedure_output))
        attn_o2_beta = F.tanh(self.beta(procedure_output))
        procedure_output = attn_o2_alpha * attn_o2_beta * procedure_seq  # (visit, emb)
        procedure_output = torch.sum(procedure_output, dim=0).unsqueeze(dim=0)  # (1, emb)
        procedure_output = self.Reain_output(procedure_output)

        # RNN
        # diagnosis_output, diagnosis_hidden = self.encoders[0](
        #     diagnosis_seq
        # )  # for dia
        # procedure_output, procedure_hidden = self.encoders[1](
        #     procedure_seq
        # )  # for procedure

        patient_representations = torch.cat([diagnosis_output, procedure_output], dim=-1).squeeze(dim=0)  # (seq, dim*4)
        queries = self.query(patient_representations)  # (seq, dim)
        P = queries[-1:]  # (1,dim)

        #  medication representation
        if self.ddi_in_memory:
            medication_representation_K = self.ehr_gnn() - self.ddi_gnn() * self.lambda_  # (size, dim)
        else:
            medication_representation_K = self.ehr_gnn()

        if len(patient) > 1:
            history_P = queries[:(queries.size(0)-1)]  # (seq-1, dim)
            history_medication = np.zeros((len(patient)-1, self.vocab_size[2]))
            for idx, adm in enumerate(patient):
                if idx == len(patient)-1:
                    break
                history_medication[idx, adm[2]] = 1
            history_medication = torch.FloatTensor(history_medication).to(self.device)  # (seq-1, size)

        weights1 = F.softmax(torch.mm(P, medication_representation_K.t()), dim=-1)  # (1, size)
        R1 = torch.mm(weights1, medication_representation_K)  # (1, dim)

        if len(patient) > 1:
            weight2 = F.softmax(torch.mm(P, history_P.t()))  # (1, seq-1)
            weighted_values = weight2.mm(history_medication)  # (1, size)
            R2 = torch.mm(weighted_values, medication_representation_K)  # (1, dim)
        else:
            R2 = R1

        output = self.output(torch.cat([P, R1, R2], dim=-1))  # (1, dim)

        if self.training:
            neg_pred_prob = F.sigmoid(output)
            neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
            batch_neg = neg_pred_prob.mul(self.tensor_ddi_adj).mean()

            return output, batch_neg
        else:
            return output

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)

        self.lambda_.data.uniform_(-initrange, initrange)

class SGC(nn.Module):
    def __init__(self, size, emb_dim, adj, device=torch.device('cuda')):
        super(SGC, self).__init__()
        self.voc_size = size
        self.emb_dim = emb_dim
        self.device = device
        # self.x = torch.eye(size).to(device)
        adj = self.normalize(adj + np.eye(adj.shape[0]))
        self.x = torch.FloatTensor(adj).to(device)
        self.W = nn.Linear(size, emb_dim)

    def forward(self):
        #print(self.x)
        return self.W(self.x)

    def normalize(self, mx):
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj,  device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        adj = self.normalize(adj        + np.eye(adj.shape[0]))

        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(voc_size).to(device)
        # self.gcns = nn.ModuleList()
        # self.gcn = GraphConvolution(voc_size, emb_dim)
        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        # for i in range(layers-1):
        #     self.gcns.append(GraphConvolution(emb_dim, emb_dim))
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)

        # node_embedding = self.gcn(self.x, self.adj)
        # for indx in range(len(self.gcns)):
        #     node_embedding = F.relu(node_embedding)
        #     node_embedding = self.dropout(node_embedding)
        #     node_embedding = self.gcns[indx](node_embedding, self.adj)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features #145
        self.out_features = out_features #64
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))  #145 * 64
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))  #64
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

def eval(model, data_eval, voc_size, epoch):
    # evaluate
    model.eval()
    smm_record = []
    Jaccard, PRAUC, F1 = [[] for _ in range(3)]
    med_cnt = 0
    visit_cnt = 0
    for index1, patient in enumerate(data_eval):
        y_gt = []
        y_pred = []
        y_pred_prob = []
        y_pred_label = []
        for index2, admission_time in enumerate(patient):
            target_output1 = model(patient[:index2+1])
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[admission_time[2]] = 1
            y_gt.append(y_gt_tmp)

            target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output1)
            y_pred_tmp = target_output1.copy()
            y_pred_tmp[y_pred_tmp >= 0.5] = 1
            y_pred_tmp[y_pred_tmp < 0.5] = 0
            y_pred.append(y_pred_tmp)
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)

        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))

        Jaccard.append(adm_ja)
        PRAUC.append(adm_prauc)
        F1.append(adm_avg_f1)
        llprint('\rEval--Epoch: %d, Patient: %d/%d' % (epoch, index1, len(data_eval)))

    # DDI rate
    DDI = ddi_rate_score(smm_record)
    llprint('\tDDI Rate: %.4f, Jaccard: %.4f,  PRAUC: %.4f, AVG_F1: %.4f\n' % (DDI, np.mean(Jaccard), np.mean(PRAUC), np.mean(F1)))

    print('avg med', med_cnt / visit_cnt)

    return DDI, np.mean(Jaccard), np.mean(PRAUC), np.mean(F1)

def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = torch.sparse_tensor_dense_matmul(x, y)
    else:
        res = torch.matmul(x, y)
    return res
