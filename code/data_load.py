
import torch
import numpy as np
import dill
torch.manual_seed(1203)
np.random.seed(1203)

def load_data():

    path1 = '../data/records_final.pkl'
    path2 = '../data/voc_final.pkl'

    path3 = '../data/ehr_adj_final.pkl'
    path4 = '../data/ddi_A_final.pkl'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = dill.load(open(path1, 'rb'))
    sequence_data = dill.load(open(path2, 'rb'))
    ehr_data = dill.load(open(path3, 'rb'))
    ddi_data = dill.load(open(path4, 'rb'))

    diagnosis, procedure, medication = sequence_data['diag_voc'], sequence_data['pro_voc'], sequence_data['med_voc']

    partition = int(len(data) * 2 / 3)
    data_train = data[:partition]
    eval_len = int(len(data[partition:]) / 2)
    data_test = data[partition:partition + eval_len]
    data_eval = data[partition+eval_len:]

    return device,ehr_data,ddi_data,diagnosis, procedure, medication,data_train,data_test,data_eval