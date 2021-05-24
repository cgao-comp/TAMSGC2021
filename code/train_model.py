
import torch
import numpy as np
import dill
from torch.optim import SGD, Adam
import os
import torch.nn.functional as F
from collections import defaultdict
from models import TAMSGC,eval
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params
from data_load import load_data

model_name = 'TAMSGC'
def main():
    if not os.path.exists(os.path.join("saved", model_name)):
        os.makedirs(os.path.join("saved", model_name))

    # load data
    device, ehr_data, ddi_data, diagnosis, procedure, medication, data_train, data_test, data_eval = load_data()

    # parameter setting
    EPOCH = 40
    LR = 0.0001
    Neg_Loss = 'store_true'
    DDI_IN_MEM = True
    TARGET_DDI = 0.05
    T = 0.5
    decay_weight = 0.85
    gamma1 = 0.9
    gamma2 = 0.01

    size = (len(diagnosis.idx2word), len(procedure.idx2word), len(medication.idx2word))
    model = TAMSGC(size, ehr_data, ddi_data, emb_dim=64, device=device, ddi_in_memory=DDI_IN_MEM)

    model.to(device=device)
    print('parameters', get_n_params(model))
    optimizer = Adam(list(model.parameters()), lr=LR)

    history = defaultdict(list)
    for epoch in range(EPOCH):
        loss_record1 = []
        model.train()
        prediction_loss_cnt = 0
        neg_loss_cnt = 0
        for index1, patient in enumerate(data_train):
            for index2, admission_time in enumerate(patient):
                seq_input = patient[:index2 + 1]
                loss1_target = np.zeros((1, size[2]))
                loss1_target[:, admission_time[2]] = 1
                loss2_target = np.full((1, size[2]), -1)
                for index3, elements in enumerate(admission_time[2]):
                    loss2_target[0][index3] = elements

                target_output1, batch_neg_loss = model(seq_input)
                # loss function
                loss1 = F.binary_cross_entropy_with_logits(target_output1, torch.FloatTensor(loss1_target).to(device))
                loss2 = F.multilabel_margin_loss(F.sigmoid(target_output1), torch.LongTensor(loss2_target).to(device))

                if Neg_Loss:
                    target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
                    target_output1[target_output1 >= 0.5] = 1
                    target_output1[target_output1 < 0.5] = 0
                    y_label = np.where(target_output1 == 1)[0]
                    current_ddi_rate = ddi_rate_score([[y_label]])
                    if current_ddi_rate <= TARGET_DDI:
                        loss = 0.9 * loss1 + 0.01 * loss2
                        prediction_loss_cnt += 1
                    else:
                        rnd = np.exp((TARGET_DDI - current_ddi_rate) / T)
                        if np.random.rand(1) < rnd:
                            loss = batch_neg_loss
                            neg_loss_cnt += 1
                        else:
                            loss = gamma1 * loss1 + gamma2 * loss2
                            prediction_loss_cnt += 1
                else:
                    loss = gamma1 * loss1 + gamma2 * loss2
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                loss_record1.append(loss.item())

            llprint('\rTrain--Epoch: %d, Patient: %d/%d, L_p cnt: %d, L_neg cnt: %d' % (
            epoch, index1, len(data_train), prediction_loss_cnt, neg_loss_cnt))
        T *= decay_weight
        DDI, Jaccard, PRAUC, F1 = eval(model, data_eval, size, epoch)
        history['Jaccard'].append(Jaccard)
        history['DDI'].append(DDI)
        history[' F1'].append(F1)
        history['PRAUC'].append(PRAUC)

        llprint('\tEpoch: %d, Loss: %.4f\n' % (epoch, np.mean(loss_record1)))
        torch.save(model.state_dict(), open(os.path.join('saved', model_name, 'Epoch_%d' % (epoch)), 'wb'))
    dill.dump(history, open(os.path.join('saved', model_name, 'history.pkl'), 'wb'))

if __name__ == '__main__':
    main()

