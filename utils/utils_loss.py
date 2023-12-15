import torch
import torch.nn.functional as F
import torch.nn as nn
import math



def alpha_loss(alpha, prior_alpha):
    KLD = torch.mvlgamma(alpha.sum(1), p=1)-torch.mvlgamma(alpha, p=1).sum(1)-torch.mvlgamma(prior_alpha.sum(1), p=1)+torch.mvlgamma(prior_alpha, p=1).sum(1)+((alpha-prior_alpha)*(torch.digamma(alpha)-torch.digamma(alpha.sum(dim=1, keepdim=True).expand_as(alpha)))).sum(1)
    return KLD.mean()

def lws_loss(outputs, partialY, confidence, index, lw_weight, lw_weight0, epoch_ratio):
    device = outputs.device

    onezero = torch.zeros(outputs.shape[0], outputs.shape[1])
    onezero[partialY > 0] = 1
    counter_onezero = 1 - onezero
    onezero = onezero.to(device)
    counter_onezero = counter_onezero.to(device)

    sig_loss1 = 0.5 * torch.ones(outputs.shape[0], outputs.shape[1])
    sig_loss1 = sig_loss1.to(device)
    sig_loss1[outputs < 0] = 1 / (1 + torch.exp(outputs[outputs < 0]))
    sig_loss1[outputs > 0] = torch.exp(-outputs[outputs > 0]) / (
        1 + torch.exp(-outputs[outputs > 0]))
    l1 = confidence[index, :] * onezero * sig_loss1
    average_loss1 = torch.sum(l1) / l1.size(0)

    sig_loss2 = 0.5 * torch.ones(outputs.shape[0], outputs.shape[1])
    sig_loss2 = sig_loss2.to(device)
    sig_loss2[outputs > 0] = 1 / (1 + torch.exp(-outputs[outputs > 0]))
    sig_loss2[outputs < 0] = torch.exp(
        outputs[outputs < 0]) / (1 + torch.exp(outputs[outputs < 0]))
    l2 = confidence[index, :] * counter_onezero * sig_loss2
    average_loss2 = torch.sum(l2) / l2.size(0)

    average_loss = lw_weight0 * average_loss1 + lw_weight * average_loss2
    return average_loss, lw_weight0 * average_loss1, lw_weight * average_loss2


def partial_loss(output1, target, true): #proden
    output = F.softmax(output1, dim=1)
    l = target * torch.log(output) #log_softmax(output)
    loss = (-torch.sum(l)) / l.size(0)
    revisedY = target.clone()
    revisedY[revisedY > 0]  = 1 # 不为0的都置为1
    revisedY = revisedY * output # output过滤到为0的
    revisedY = revisedY / revisedY.sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)

    new_target = revisedY

    return loss, new_target

def revised_target(output, target):
    revisedY = target.clone()
    revisedY[revisedY > 0]  = 1
    revisedY = revisedY * (output.clone().detach())
    revisedY = revisedY / revisedY.sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)
    new_target = revisedY

    return new_target

def con_correct(outputs, confidence, index, theta):
    sm_outputs =F.softmax(outputs,dim=1) # 输出先进行softmax
    logsm_outputs = F.log_softmax(outputs,dim=1)
    ac_outptus = torch.log(1-sm_outputs[:,-1])
    ac_loss_1 = - theta * ac_outptus.mean()
    final_outputs = logsm_outputs * confidence[index, :]
    average_loss = - theta*((final_outputs).sum(dim=1)).mean()
    return average_loss,ac_loss_1


def gce_loss(outputs, Y, q):
    sm_outputs = F.softmax(outputs, dim=1)
    pow_outputs = torch.pow(sm_outputs, q)
    sample_loss = (1-(pow_outputs*Y).sum(dim=1))/q # n
    return sample_loss.sum()

# def unbiased(outputs, confidence, index, theta,device):
#     sm_outputs =F.softmax(outputs,dim=1) # 输出先进行softmax
#     ######################
#     n,k = outputs.shape[0],outputs.shape[1]
#     temp_loss = torch.zeros(n,k).to(device)
#     for i in range(k):
#         tempY = torch.zeros(n, k).to(device)
#         tempY[:, i] = 1.0  # calculate the loss with respect to the i-th label 计算预测值和每个类的误差(后面有独热向量，存在0项因此不会影响)
#         temp_loss[:, i] = gce_loss(outputs, tempY, 0.7)
#     ########################
#     # logsm_outputs = F.log_softmax(outputs,dim=1)
#     ac_outputs = temp_loss[:,-1] # 输出是ac
#     # ac_outputs = torch.reshape(ac_outputs,(len(ac_outputs),1))
#     #reshape成列向量
#     # ac_outputs = torch.log(sm_outputs[:,-1])  # 加1e-6防止输出NaN
#     ac_loss_1 = theta * ac_outputs.mean()  # 手动算平均值
#     final_outputs = temp_loss * confidence[index, :]
#     average_loss =  theta*((final_outputs).sum(dim=1)).mean()
#     return average_loss

def log_loss(outputs,Y):
    sm_outputs = F.softmax(outputs, dim=1)
    sample_loss = -Y*torch.log(sm_outputs)-(1-Y)*torch.log(1-sm_outputs)
    return sample_loss.sum()


def exp_loss(outputs,Y):
    sm_outputs = F.softmax(outputs, dim=1)
    sample_loss = torch.exp(-sm_outputs*Y)
    return sample_loss.sum()

def phuber_ce_loss(outputs, Y,device):
    trunc_point = 0.1
    n = Y.shape[0]
    soft_max = nn.Softmax(dim=1)
    sm_outputs = soft_max(outputs)
    final_outputs = sm_outputs * Y
    final_confidence = final_outputs.sum(dim=1)

    ce_index = (final_confidence > trunc_point)
    sample_loss = torch.zeros(n).to(device)

    if ce_index.sum() > 0:
        ce_outputs = outputs[ce_index, :]
        logsm = nn.LogSoftmax(dim=-1)
        logsm_outputs = logsm(ce_outputs)
        final_ce_outputs = logsm_outputs * Y[ce_index, :]
        sample_loss[ce_index] = - final_ce_outputs.sum(dim=-1)

    linear_index = (final_confidence <= trunc_point)

    if linear_index.sum() > 0:
        sample_loss[linear_index] = -math.log(trunc_point) + (-1 / trunc_point) * final_confidence[linear_index] + 1

    return sample_loss.sum()


def gce_loss(outputs, Y):
    q = 0.7
    sm_outputs = F.softmax(outputs, dim=1)
    pow_outputs = torch.pow(sm_outputs, q)
    sample_loss = (1-(pow_outputs*Y).sum(dim=1))/q # n
    return sample_loss.sum()

def mae_loss(outputs, partialY):
    sm_outputs = F.softmax(outputs, dim=1)
    loss_fn = nn.L1Loss(reduction='none')
    loss_matrix = loss_fn(sm_outputs, partialY.float())
    sample_loss = loss_matrix.sum(dim=-1)
    return sample_loss.sum()

def mse_loss(outputs, Y):
    sm_outputs = F.softmax(outputs, dim=1)
    loss_fn = nn.MSELoss(reduction='none')
    loss_matrix = loss_fn(sm_outputs, Y.float())
    sample_loss = loss_matrix.sum(dim=-1)
    return sample_loss.sum()

def unbiased(outputs, confidence, index, theta):
    logsm_outputs = F.log_softmax(outputs,dim=1)
    ac_outputs = logsm_outputs[:,-1] # 输出是ac
    # ac_outputs = torch.reshape(ac_outputs,(len(ac_outputs),1))
    #reshape成列向量
    ac_loss_1 = theta * ac_outputs.mean()  # 手动算平均值
    final_outputs = logsm_outputs * confidence[index, :]
    average_loss = - theta*((final_outputs).sum(dim=1)).mean()
    return average_loss,ac_loss_1

def unbiased_cc(outputs, partialY, theta):
    logsm_outputs = F.softmax(outputs, dim=1)
    ac_outputs = logsm_outputs[:, -1]  # 输出是ac
    ac_loss_1 = theta * ac_outputs.mean()
    final_outputs = cc_loss(outputs,partialY)
    average_loss = theta * final_outputs
    return average_loss,ac_loss_1

def cc_loss(outputs, partialY):
    sm_outputs = F.softmax(outputs, dim=1)
    final_outputs = sm_outputs * partialY
    average_loss = - torch.log(final_outputs.sum(dim=1)).mean()
    return average_loss

def unbiased_proden(outputs,Y,theta,trues):
    logsm_outputs = F.log_softmax(outputs,dim=1)
    ac_outputs = logsm_outputs[:,-1] # 输出是ac
    # ac_outputs = torch.reshape(ac_outputs,(len(ac_outputs),1))
    #reshape成列向量
    ac_loss_1 = theta * ac_outputs.mean()  # 手动算平均值
    final_outputs, new_targets = partial_loss(outputs,Y,trues)
    average_loss = theta*final_outputs
    return average_loss,ac_loss_1,new_targets

def rc_loss(outputs, confidence, index):
    logsm_outputs = F.log_softmax(outputs, dim=1)
    final_outputs = logsm_outputs * confidence[index, :]
    average_loss = - ((final_outputs).sum(dim=1)).mean()
    return average_loss



def unlabeled_loss(outputs):
    final_outputs = F.log_softmax(outputs,dim=1)
    # print(final_outputs)
    average_loss = - final_outputs[:,-1].mean()
    return average_loss
