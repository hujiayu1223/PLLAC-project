import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

def accuracy_check(loader, model, device):
    with torch.no_grad():
        total, num_samples = 0, 0
        for images, labels, label, index in loader:
            labels, images = labels.to(device), images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += (predicted == labels).sum().item()
            num_samples += labels.size(0)
    return total / num_samples

def test_accuracy_check_v2(loader, model,device):
    model.eval()
    AUC, Macro_F1 = 0, 0
    correct = 0
    total = 0
    for images, label, labels, index in loader:
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)

        logit = F.softmax(output, dim=1)
        back = torch.zeros(size=(len(logit), 1)).to(device)
        logit = torch.cat((logit, back), dim=1)
        # 更改，根据概率是否大于0.8判断是否为ac
        sm_outputs, pred = torch.max(logit.data, 1)
        num_classes = logit.shape[1]
        pred[sm_outputs < 0.82] = num_classes
        # add_new = torch.arange(0,num_classes,step=1).to(device)
        # labels = torch.cat((labels,add_new),dim=0)
        select_index = [i == num_classes for i in pred]
        especial = torch.zeros(size=(1,num_classes))[0].to(device)
        especial[num_classes-1] = 1
        logit[select_index] = especial
        label, predict, prediction_scores = labels.cpu().numpy(), pred.detach().cpu().numpy(), logit.detach().cpu().numpy()
        one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        one_hot_encoder.fit(np.array(label).reshape(-1, 1))
        true_scores = one_hot_encoder.transform(np.array(label).reshape(-1, 1))
        # true_scores = true_scores[:-num_classes,:]
        # print(prediction_scores.shape,true_scores.shape)
        assert prediction_scores.shape == true_scores.shape
        # AUC = 0
        AUC = roc_auc_score(true_scores, prediction_scores, multi_class='ovo')
        Macro_F1 = f1_score(label, predict, average='macro')
        total += images.size(0)
        correct += (pred == labels).sum().item()
    acc = 100 * float(correct) / float(total)
    return AUC, Macro_F1, acc



def test_accuracy_check(loader,model,device,num_classes):
    with torch.no_grad():
        AUC, Macro_F1, total, num_samples = 0, 0, 0, 0
        for images,given_labels,label,index in loader:
            labels, images = label.to(device), images.to(device)
            if len(images) < 50:
                continue
            outputs = model(images)
            logit = F.softmax(outputs, dim=1)
            # back = torch.zeros(size=(len(logit),1)).to(device)
            # logit = torch.cat((logit,back),dim=1)
            sm_outputs, predicted = torch.max(outputs.data, 1)
            # select_index = [i==num_classes for i in predicted]
            # especial = torch.zeros(size=(1,num_classes+1))[0].float().to(device)
            # especial[num_classes] = 1
            # especial = torch.tensor([0,0,0,0,0,0,1]).float().to(device)
            # logit[select_index] = especial
            # predicted[sm_outputs < 0.75] = num_classes # 认为是ac
            label, predict, prediction_scores = label.cpu().numpy(), predicted.detach().cpu().numpy(), logit.detach().cpu().numpy()
            one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            one_hot_encoder.fit(np.array(label).reshape(-1, 1))
            true_scores = one_hot_encoder.transform(np.array(label).reshape(-1, 1))

            # print(prediction_scores.shape,true_scores.shape)
            # exit()
            assert prediction_scores.shape == true_scores.shape
            AUC = roc_auc_score(true_scores, prediction_scores, multi_class='ovo')
            Macro_F1 = f1_score(label, predict, average='macro')
            total += (predicted == labels).sum().item()
            num_samples += labels.size(0)
            # for i in range(len(label)):
            #     print(label[i], " ", predicted[i])
    return AUC,Macro_F1,total / num_samples

def train_accuracy_check(loader, model, device):
    with torch.no_grad():
        total, num_samples = 0, 0
        for i,data in enumerate(loader):
            images,_,labels,_ = data
            # images = torch.unsqueeze(images,dim=1)
            labels, images = labels.to(device), images.to(device)
            outputs = model(images)
            sm_outputs, predicted = torch.max(outputs.data, 1)
            # predicted[sm_outputs < 0.7] = num_classes  # 认为是ac
            total += (predicted == labels).sum().item()
            num_samples += labels.size(0)
    return total / num_samples

def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
    return A_pred



def test_accuracy_check_ac(loader, model,device,al=0):
    model.eval()
    AUC, Macro_F1 = 0, 0
    correct = 0
    total = 0
    batch = 0
    for images, label,labels,index in loader:
        batch += 1
        images = images.to(device)
        labels = labels.to(device)
        if al == 1:
            _,output = model(images)
        else:
            output = model(images)
        logit = F.softmax(output, dim=1)
        back = torch.zeros(size=(len(logit), 1)).to(device)
        logit = torch.cat((logit, back), dim=1)
        # 更改，根据概率是否大于0.8判断是否为ac
        sm_outputs, pred = torch.max(logit.data, 1)
        # num_classes = len(torch.unique(labels,sorted=True))-1
        num_classes = output.shape[1]
        pred[sm_outputs < 0.95] = num_classes
        select_index = [i == num_classes for i in pred]
        especial = torch.zeros(size=(1,num_classes+1))[0].to(device)
        especial[num_classes] = 1
        logit[select_index] = especial
        label, predict, prediction_scores = labels.cpu().numpy(), pred.detach().cpu().numpy(), logit.detach().cpu().numpy()
        one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        one_hot_encoder.fit(np.array(label).reshape(-1, 1))

        true_scores = one_hot_encoder.transform(np.array(label).reshape(-1, 1))
        # print(prediction_scores.shape,true_scores.shape)
        assert prediction_scores.shape == true_scores.shape
        # AUC = 0
        AUC += roc_auc_score(true_scores, prediction_scores, multi_class='ovo')
        Macro_F1 += f1_score(label, predict, average='macro')
        total += images.size(0)
        correct += (pred == labels).sum().item()
    AUC = float(AUC)/float(batch)
    Macro_F1 = float(Macro_F1)/float(batch)
    acc = float(correct) / float(total)
    return AUC, Macro_F1, acc

def cavl_confidence_update(model, confidence, batchX, batchY, batch_index):
    with torch.no_grad():
        batch_outputs = model(batchX)
        cav = (batch_outputs*torch.abs(1-batch_outputs))*batchY
        cav_pred = torch.max(cav,dim=1)[1]
        gt_label = F.one_hot(cav_pred,batchY.shape[1]) # label_smoothing() could be used to further improve the performance for some datasets
        confidence[batch_index,:] = gt_label.float()
    return confidence

def confidence_update_lw(model, confidence, batchX, batchY, batch_index):
    with torch.no_grad():
        device = batchX.device
        batch_outputs = model(batchX)
        sm_outputs = F.softmax(batch_outputs, dim=1)

        onezero = torch.zeros(sm_outputs.shape[0], sm_outputs.shape[1])
        onezero[batchY > 0] = 1
        counter_onezero = 1 - onezero
        onezero = onezero.to(device)
        counter_onezero = counter_onezero.to(device)

        new_weight1 = sm_outputs * onezero
        new_weight1 = new_weight1 / (new_weight1 + 1e-8).sum(dim=1).repeat(
            confidence.shape[1], 1).transpose(0, 1)
        new_weight2 = sm_outputs * counter_onezero
        new_weight2 = new_weight2 / (new_weight2 + 1e-8).sum(dim=1).repeat(
            confidence.shape[1], 1).transpose(0, 1)
        new_weight = new_weight1 + new_weight2

        confidence[batch_index, :] = new_weight
        return confidence

def confidence_update(model, confidence, batchX, batchY, batch_index):
    with torch.no_grad():
        batch_outputs = model(batchX)
        temp_un_conf = F.softmax(batch_outputs, dim=1)
        confidence[batch_index, :] = temp_un_conf * batchY # un_confidence stores the weight of each example
        base_value = confidence.sum(dim=1).unsqueeze(1).repeat(1, confidence.shape[1])
        confidence = confidence/base_value
    return confidence

def evaluate(model, X_test, y_test, device):
    X_test, y_test = X_test.to(device), y_test.to(device)
    outputs = model(X_test)
    pred = outputs.argmax(dim=1)
    logit = F.softmax(outputs, dim=1)
    label, predict, prediction_scores = y_test.cpu().numpy(), pred.detach().cpu().numpy(), logit.detach().cpu().numpy()
    one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    one_hot_encoder.fit(np.array(label).reshape(-1, 1))
    true_scores = one_hot_encoder.transform(np.array(label).reshape(-1, 1))
    assert prediction_scores.shape == true_scores.shape
    if torch.isnan(outputs).int().sum()>0 :
        print("出现了NaN")
        print(X_test)
        print(model.parameters())
        exit()
    AUC = roc_auc_score(true_scores, prediction_scores, multi_class='ovo')
    Macro_F1 = f1_score(label, predict, average='macro')
    accuracy = (pred==y_test).sum().item()/pred.shape[0]
    return accuracy, Macro_F1, AUC