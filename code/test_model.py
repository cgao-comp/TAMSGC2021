import torch
import os
from models import TAMSGC, eval
from data_load import load_data

def main():
    # load data
    device, ehr_data, ddi_data, diagnosis, procedure, medication, data_train, data_test, data_eval = load_data()

    # parameter setting
    DDI_IN_MEM = True
    size = (len(diagnosis.idx2word), len(procedure.idx2word), len(medication.idx2word))
    model = TAMSGC(size, ehr_data, ddi_data, emb_dim=64, device=device, ddi_in_memory=DDI_IN_MEM)  # 加载模型

    path = '..\code\saved\TAMSGC'
    files = os.listdir(path)

    for file in files:
        model.load_state_dict(torch.load(open(path + '\\' + file, 'rb')))
        model.to(device=device)
        model.to()
        eval(model, data_test, size, 0)

if __name__ == '__main__':
    main()