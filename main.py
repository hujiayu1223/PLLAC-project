import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from utils.models import mlp_model, linear_model, LeNet
import torch
from scipy.io import loadmat
from painting import unlabel_num_painting
from utils.utils_data import make_realworld_mpu_train_set_v3,prepare_cv_datasets, prepare_train_loaders_for_uniform_cv_candidate_labels,make_cv_mpu_train_set,make_uci_mpu_train_set
from utils.utils_algo import accuracy_check, confidence_update, evaluate,test_accuracy_check,test_accuracy_check_ac
from utils.utils_loss import rc_loss, cc_loss, unlabeled_loss,unbiased,con_correct,unbiased_cc,unbiased_proden
from cifar_models import densenet, resnet
import numpy as np
from utils.algorithm import cavl,proden,lwpll,valen,rc,cc,gen,rc_loss_plus
from utils.utils_func import KernelPriorEstimator

from hujiayu import data_output
import itertools
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils.utils_earlystop import EarlyStopping
from utils.gen_index_dataset import gen_index_dataset
parser = argparse.ArgumentParser()
parser.add_argument('-un', help='num of unlabel data', default=0, type=int)
parser.add_argument('-lr', help='optimizer\'s learning rate', default=1e-3, type=float)
parser.add_argument('-bs', help='batch_size of ordinary labels.', default=256, type=int)
parser.add_argument('-al',help='compared experiment',default='con_correct',type=str)
parser.add_argument('-ds', help='specify a dataset', default='mnist', type=str,
                    required=False)  # mnist, kmnist, fashion, cifar10
parser.add_argument('-dt', help='specify a dataset type', default='benchmark', type=str,
                    required=False)  # benchmark,uci,realworld
parser.add_argument('-mo', help='model name', default='mlp', choices=['linear', 'mlp', 'resnet', 'densenet', 'lenet'],
                    type=str, required=False)
parser.add_argument('-ep', help='number of epochs', type=int, default=150)
parser.add_argument('-wd', help='weight decay', default=1e-5, type=float)
parser.add_argument('-lo', help='specify a loss function', default='con_correct', type=str, required=False)
parser.add_argument('-seed', help='Random seed', default=0, type=int, required=False)
parser.add_argument('-gpu', help='used gpu id', default='0', type=str, required=False)
parser.add_argument('-uci', type=int, help='UCI dataset or not', default=0, choices=[1, 0])
parser.add_argument('-gamma', type=float, help='index of regularization', default=1)
parser.add_argument('-thetae', type=float, help='mixture proportion', default=0.1)
parser.add_argument('-t', type=float, help='mixture proportion', default=0.1)
args = parser.parse_args()
output_acc_list=[]
output_f1_list=[]
output_auc_list=[]
# unlabel_num_painting()
# exit()
# for method in ["unbiased_cc","unbiased_proden"]:
#     data_output(method)
# exit()
for i in range(1):
# for gamma in np.arange(0.1,0.6,0.1):
    gamma=1.0
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:"+args.gpu if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # print(device)
    # 先进行对labeled 和unlabeled 分类，然后在对labeled进行partial的分配
    num_labeled=4000
    num_unlabeled=1000
    num_test=500
    # uci and cv

    if args.dt == 'benchmark' or args.dt == 'uci':
        if args.dt == 'uci':
            train_loader,test_loader,train_givenY,y_labeled_given_Y_without_ac,train_without_ac_loader,dim, K ,X_labeled,X_unlabeled= make_uci_mpu_train_set(args, 500, 1000, 1000, args.seed)
        else:
            full_train_loader, train_loader, test_loader, full_test_loader, ordinary_train_dataset, test_dataset, K = prepare_cv_datasets(
                dataname=args.ds, batch_size=args.bs)
            train_loader,test_loader,train_givenY,train_without_ac_loader,y_labeled_given_Y_without_ac,dim, K,X_labeled,X_unlabeled = make_cv_mpu_train_set(full_train_loader,full_test_loader,
                                                                                                      num_labeled,
                                                                                                   num_unlabeled, num_test,
                                                                                                      args,0.6,args.thetae)
    else: # realworld
        dim, K, train_givenY, train_loader, test_loader,train_without_ac_loader,y_labeled_given_Y_without_ac,X_labeled,X_unlabeled = make_realworld_mpu_train_set_v3(args.ds,args.bs)
    if args.dt != "benchmark" and args.al == 'unbiased':
        theta = KernelPriorEstimator(X_labeled, X_unlabeled, 5)
    else:
        theta = 0.86
    if args.al != 'con_correct' and args.al != 'unbiased':
        train_givenY = y_labeled_given_Y_without_ac
        K = K-1
        train_loader = train_without_ac_loader
    tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
    confidence = train_givenY.float() / tempY
    confidence = confidence.to(device)
    print("make sure confidence")
    if args.lo == 'con_correct':
        loss_fn = con_correct
    elif args.lo == 'cc':
        loss_fn = cc_loss
    elif args.lo == 'unbiased':
        loss_fn = unbiased
    elif args.lo == 'rc':
        loss_fn = rc_loss
    elif args.lo == 'lws':
        n, c = train_givenY.shape[0], train_givenY.shape[1]
        confidence = torch.ones(n, c) / c
        confidence = confidence.to(device)

    if args.mo == 'mlp':
        model = mlp_model(input_dim=dim, hidden_dim=500, output_dim=K)
    elif args.mo == 'linear':
        model = linear_model(input_dim=dim, output_dim=K)
    elif args.mo == 'lenet':
        model = LeNet(output_dim=K) #  linear,mlp,lenet are for MNIST-type datasets.
    elif args.mo == 'densenet':
        model = densenet(num_classes=K)
    elif args.mo == 'resnet':
        model = resnet(depth=32, num_classes=K) # densenet,resnet are for CIFAR-10.

    if args.al != 'valen':
        model = model.to(device)  #valen的model要在算法里面进行选择

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.wd)
    if args.al != 'unbiased' and args.al != 'con_correct':
        if args.al == 'cavl':
            test_accuracy,f1_list,auc_list = cavl(args,model,train_loader,optimizer,device,test_loader,confidence)
        elif args.al == 'proden':
            test_accuracy, f1_list, auc_list = proden(args, model, train_loader, optimizer, device, test_loader)
        elif args.al == 'lwpll':
            test_accuracy, f1_list, auc_list = lwpll(args, model, train_loader, optimizer, device, test_loader,confidence)
        elif args.al == 'valen':
            test_accuracy, f1_list, auc_list = valen(args, dim,K, train_loader, device, test_loader)
        elif args.al == 'rc':
            test_accuracy, f1_list, auc_list = rc(args, model, train_loader, optimizer, device, test_loader, confidence)
        elif args.al == 'cc':
            test_accuracy, f1_list, auc_list = cc(args, model, train_loader, optimizer, device, test_loader, confidence)
        elif args.al != 'unbiased':
            test_accuracy, f1_list, auc_list = gen(args, model, train_loader, optimizer, device, test_loader, confidence)
        print("Learning Rate:", args.lr, "Weight Decay:", args.wd, "seed:", args.seed, "batch_size:", args.bs,"algorithm:", args.al,"loss:", args.lo)
        print("Average Test Accuracy over Last 10 Epochs:", str(test_accuracy)[:8])
        print('Mean Macro_F1 of last 10 epoch:', np.mean(f1_list))
        print('Mean AUC of last 10 epoch:', np.mean(auc_list))
    else:
        test_acc_list = []
        test_acc_last_list = []
        train_acc_list = []
        valid_loss_list = []
        f1_list = []
        auc_list = []
        test_list = []
        loss_list = []
        print("theta: ",theta)
        isNaN = False # 用于记录是否出现NaN
        for epoch in range(args.ep):
            model.train()
            loss = 0
            count = 1
            for i, (images, labels, true_labels, index) in enumerate(train_loader):
                X, Y, index = images.to(device), labels.to(device), index.to(device)
                unlabeled_index = (Y[:,-1] != 0)
                labeled_index = (unlabeled_index==False)
                label_index_num = index[labeled_index]
                optimizer.zero_grad()
                outputs = model(X)
                pred_labeled = outputs[labeled_index, :]
                labeled_batchY = Y[labeled_index, :]
                pred_unlabeled = outputs[unlabeled_index, :]
                unlabeled_batchY = Y[unlabeled_index, :]
                if args.al == 'con_correct' or args.lo == 'unbiased':
                    loss_1,ac_loss = loss_fn(pred_labeled, confidence, index[labeled_index==True], theta)
                elif args.lo == 'unbiased_cc':
                    loss_1, ac_loss = unbiased_cc(pred_labeled, labeled_batchY, theta)
                elif args.lo == 'unbiased_proden':
                    loss_1, ac_loss,new_targets = unbiased_proden(pred_labeled,labeled_batchY,theta,true_labels[labeled_index])
                    for j, k in enumerate(label_index_num):
                        train_loader.dataset.given_label_matrix[k, :] = new_targets[j, :].detach()
                else:
                    loss_1,ac_loss = rc_loss_plus(pred_labeled,labeled_batchY,theta,args,device)
                loss_2 = unlabeled_loss(pred_unlabeled)
                average_loss = loss_1 + loss_2 + ac_loss
                if args.al == 'con_correct':
                    average_loss = loss_1 + loss_2 + ac_loss
                else:
                    if ac_loss + loss_2 >= 0: # regulazition
                        average_loss = loss_1 + loss_2 + ac_loss
                    else:
                        average_loss = loss_1 + loss_2 + ac_loss + gamma*(-ac_loss-loss_2)**args.t
                ###########################
                average_loss.backward()
                loss += average_loss
                count += 1
                optimizer.step()
                if args.lo == "unbiased" or args.lo == "con_correct":
                    confidence = confidence_update(model, confidence, X, Y, index)
            model.eval()
            if isNaN:
                break
            if args.al == "unbiased" or args.lo == "con_correct":
                auc, f1, test_accuracy = test_accuracy_check(loader=test_loader, model=model, device=device, num_classes=K)
            else:
                auc, f1, test_accuracy = test_accuracy_check_ac(loader=test_loader, model=model, device=device)
            # print('Epoch: {}. Loss: {}. Test Acc: {}.'.format(epoch+1,loss, test_accuracy))
            test_list.append(test_accuracy)
            loss_list.append(loss)
            print('epoch:', epoch + 1, 'loss:', str(loss.data.item() / count)[:8], 'Te acc:', str(test_accuracy)[:8], 'F1:',str(f1)[:8], 'AUC:', str(auc)[:8])

            f1_list.append(f1)
            auc_list.append(auc)
            test_acc_list.append(test_accuracy)

        print('t',args.t,'gamma,',gamma)
        print("Learning Rate:", args.lr, "Weight Decay:", args.wd, "seed:", args.seed, "batch_size:", args.bs,"algorithm:", args.al,"loss:", args.lo)
        print("Average Test Accuracy over Last 10 Epochs:", np.mean(test_acc_list))
        print('Mean Macro_F1 of last 10 epoch:', np.mean(f1_list))
        print('Mean AUC of last 10 epoch:', np.mean(auc_list))
print(output_acc_list)
print(output_f1_list)
print(output_auc_list)
save_dir = "./results_cv_best"
save_name = args.ds
save_path = os.path.join(save_dir, save_name)
save_path2 = os.path.join(save_path, args.mo)
# os.makedirs(save_path)
# with open(save_path2, "w+") as f:
#     f.writelines("{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n".format(0, last_acc, std_acc,last_f1,std_f1,last_auc,std_auc))
