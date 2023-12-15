import torch
import torch.nn.functional as F
import numpy as np
import os
from copy import deepcopy
from torch.autograd import Variable # proden
from utils.utils_algo import cavl_confidence_update,train_accuracy_check,test_accuracy_check_ac,confidence_update_lw,dot_product_decode,confidence_update,test_accuracy_check
from utils.utils_loss import rc_loss,partial_loss,lws_loss,alpha_loss,revised_target,cc_loss,mae_loss,mse_loss,gce_loss,phuber_ce_loss,exp_loss,log_loss
from models.VGAE import VAE_Bernulli_Decoder
from models.resnet import resnet
from models.model import mlp_model,linear_model,LeNet
from models.linear import linear
import scipy.sparse as sp
from sklearn.metrics import euclidean_distances
acc_list = []
f1_list = []
auc_list = []

def rc_loss_plus(outputs, Y, theta,args,device):
    logsm_outputs = F.log_softmax(outputs,dim=1)
    ac_outputs = logsm_outputs[:, -1]  # 输出是ac
    ac_loss_1 = theta * ac_outputs.mean()  # 手动算平均值
    # final_outputs = logsm_outputs * confidence[index, :] #主要修改PLL部分的loss计算方式
    if args.al == 'mae':
        loss_fn = mae_loss
    elif args.al == 'mse':
        loss_fn = mse_loss
    elif args.al == 'exp':
        loss_fn = exp_loss
    elif args.al == 'log':
        loss_fn = log_loss
    elif args.al == 'gce':
        loss_fn = gce_loss
    elif args.al == 'pce':
        loss_fn = phuber_ce_loss
    if args.al != 'pce':
        final_outputs = loss_fn(logsm_outputs, Y)
    else:
        final_outputs = loss_fn(logsm_outputs, Y, device)
    average_loss = - theta*final_outputs.mean()
    return average_loss,ac_loss_1

def create_model(config, **args):
    if config.dt == "benchmark":
        if config.ds in ['mnist', 'kmnist', 'fashion']:
            if config.mo == 'mlp':
                net = mlp_model(input_dim=args['num_features'], hidden_dim=500, output_dim=args['num_classes'])
            elif config.mo == 'linear':
                net = linear_model(input_dim=args['num_features'], output_dim=args['num_classes'])
            elif config.mo == 'lenet':
                net = LeNet(output_dim=args['num_classes'])  # linear,mlp,lenet are for MNIST-type datasets.
            elif config.mo == 'densenet':
                net = densenet(num_classes=args['num_classes'])
            elif config.mo == 'resnet':
                net = resnet(depth=32, num_classes=args['num_classes'])  # densenet,resnet are for CIFAR-10.
        if config.ds in ['cifar10','svhn']:
            net = resnet(depth=32, n_outputs = args['num_classes'])
        enc = deepcopy(net)
        dec = VAE_Bernulli_Decoder(args['num_classes'], args['num_features'], args['num_features'])
        return net, enc, dec
    if config.dt == "realworld":
        net = linear_model(args['num_features'],args['num_classes'])
        enc = deepcopy(net)
        dec = VAE_Bernulli_Decoder(args['num_classes'], args['num_features'], args['num_features'])
        return net, enc, dec
    if config.dt == "uci":
        net = linear_model(args['num_features'], args['num_classes'])
        enc = deepcopy(net)
        dec = VAE_Bernulli_Decoder(args['num_classes'], args['num_features'], args['num_features'])
        return net, enc, dec

def adj_normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def gen_adj_matrix2(X, k=10, dir_path="",path=""):
    if os.path.exists(path):
        print("Found adj matrix file and Load.")
        adj_m = np.load(path)
        print("Adj matrix Finished.")
    else:
        print("Not Found adj matrix file and Compute.")
        dm = euclidean_distances(X, X)
        adj_m = np.zeros_like(dm)
        row = np.arange(0, X.shape[0])
        dm[row, row] = np.inf
        for _ in range(0, k):
            col = np.argmin(dm, axis=1)
            dm[row, col] = np.inf
            adj_m[row, col] = 1.0
        # os.makedirs(dir_path)
        np.save(path, adj_m)
        print("Adj matrix Finished.")
    adj_m = sp.coo_matrix(adj_m)
    adj_m = adj_normalize(adj_m + sp.eye(adj_m.shape[0]))
    adj_m = sparse_mx_to_torch_sparse_tensor(adj_m)
    return adj_m

def warm_up_realworld(config, model, train_loader, test_loader,device):
    opt = torch.torch.optim.SGD(list(model.parameters()), lr=config.lr, weight_decay=config.wd, momentum=0.9)
    partial_weight = train_loader.dataset.given_label_matrix.clone().detach().to(device)
    partial_weight = partial_weight / partial_weight.sum(dim=1, keepdim=True)
    print("Begin warm-up, warm up epoch {}".format(10))
    for _ in range(0, 10):
        for features, targets, trues, indexes in train_loader:
            features, targets, trues = map(lambda x: x.to(device), (features, targets, trues))
            _,outputs = model(features)
            L_ce, new_labels = partial_loss(outputs, partial_weight[indexes,:].clone().detach(), None)
            partial_weight[indexes,:] = new_labels.clone().detach()
            opt.zero_grad()
            L_ce.backward()
            opt.step()
    AUC,F1,test_acc = test_accuracy_check_ac(test_loader,model, device,1)
    print('After warm up','test_acc:', str(test_acc)[:8], 'F1:',
          str(F1)[:8], 'AUC:', str(AUC)[:8])
    return model, partial_weight

def warm_up_benchmark(config, model, train_loader, test_loader,device):
    opt = torch.torch.optim.SGD(list(model.parameters()), lr=config.lr, weight_decay=config.wd, momentum=0.9)
    partial_weight = train_loader.dataset.given_label_matrix.clone().detach().to(device)
    partial_weight = partial_weight / partial_weight.sum(dim=1, keepdim=True)
    print("Begin warm-up, warm up epoch 10")
    for _ in range(0, 10):
        for i,(features, targets, trues, indexes) in enumerate(train_loader):
            features, targets, trues = map(lambda x: x.to(device), (features, targets, trues))
            phi, outputs = model(features)
            L_ce, new_labels = partial_loss(outputs, partial_weight[indexes,:].clone().detach(), None)
            partial_weight[indexes,:] = new_labels.clone().detach()
            opt.zero_grad()
            L_ce.backward()
            opt.step()
    test_acc,auc,f1= test_accuracy_check_ac(test_loader,model,device,1)
    print('After warm up', 'acc:', str(test_acc)[:8], 'F1:', str(f1)[:8], 'AUC:', str(auc)[:8])
    print("Extract feature.")
    feature_extracted = torch.zeros((train_loader.dataset.images.shape[0], phi.shape[-1])).to(device)
    with torch.no_grad():
        for features, targets, trues, indexes in train_loader:
            features, targets, trues = map(lambda x: x.to(device), (features, targets, trues))
            feature_extracted[indexes, :] = model(features)[0]
    return model, feature_extracted, partial_weight

def cavl(args,model,train_loader,optimizer,device,test_loader,confidence):
    for epoch in range(args.ep):
        model.train()
        for i, (images, labels, true_labels, index) in enumerate(train_loader):
            X, Y, index = images.to(device), labels.to(device), index.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            average_loss = rc_loss(outputs, confidence, index)
            average_loss.backward()
            optimizer.step()
            confidence = cavl_confidence_update(model, confidence, X, Y, index)
        model.eval()
        # train_accuracy = train_accuracy_check(loader=train_loader, model=model, device=device)
        AUC, F1, test_accuracy = test_accuracy_check_ac(loader=test_loader, model=model, device=device)
        # print('epoch:', epoch + 1, 'train_acc', str(train_accuracy)[:8], 'test_acc:', str(test_accuracy)[:8], 'F1:',
        #       str(F1)[:8], 'AUC:', str(AUC)[:8])
        print('epoch:', epoch + 1,'test_acc:', str(test_accuracy)[:8], 'F1:',
              str(F1)[:8], 'AUC:', str(AUC)[:8])
        if epoch >= (args.ep - 10):
            acc_list.append(test_accuracy)
            f1_list.append(F1)
            auc_list.append(AUC)
    return np.mean(acc_list),f1_list,auc_list

def proden(args, model, train_loader, optimizer, device, test_loader):
    for epoch in range(0, args.ep):
        # print ('training...')
        model.train()

        for i, (images, labels, trues, indexes) in enumerate(train_loader):
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)
            trues = trues.to(device)
            output = model(images)
            loss, new_label = partial_loss(output, labels, trues)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update weights
            for j, k in enumerate(indexes):
                print(j,k,indexes)
                print(train_loader.dataset.given_label_matrix[k, :])
                print(new_label[j, :])
                exit()
                train_loader.dataset.given_label_matrix[k, :] = new_label[j, :].detach()

        # print ('evaluating model...')
        train_accuracy = train_accuracy_check(loader=train_loader, model=model, device=device)
        AUC, F1, test_accuracy = test_accuracy_check_ac(loader=test_loader, model=model, device=device)
        print('epoch:', epoch + 1, 'train_acc', str(train_accuracy)[:8], 'test_acc:', str(test_accuracy)[:8], 'F1:',
              str(F1)[:8], 'AUC:', str(AUC)[:8])
        if epoch >= (args.ep - 10):
            acc_list.append(test_accuracy)
            f1_list.append(F1)
            auc_list.append(AUC)
    return np.mean(acc_list),f1_list,auc_list

def lwpll(args,model,train_loader,optimizer,device,test_loader,confidence):

    for epoch in range(args.ep):
        model.train()
        for i, (images, labels, true_labels, index) in enumerate(train_loader):
            X, Y, index = images.to(device), labels.to(device), index.to(device)
            optimizer.zero_grad()
            outputs = model(X)

            average_loss, _, _ = lws_loss(outputs, Y.float(), confidence, index,
                                             2, 1, None)
            average_loss.backward()
            optimizer.step()
            confidence = confidence_update_lw(model, confidence, X, Y, index)
        model.eval()
        train_accuracy = train_accuracy_check(loader=train_loader,
                                              model=model,
                                              device=device)
        AUC, F1, test_accuracy = test_accuracy_check_ac(loader=test_loader,
                                                     model=model,
                                                     device=device)
        print('epoch:', epoch + 1, 'train_acc', str(train_accuracy)[:8], 'test_acc:', str(test_accuracy)[:8], 'F1:',
              str(F1)[:8], 'AUC:', str(AUC)[:8])
        if epoch >= (args.ep - 10):
            acc_list.append(test_accuracy)
            f1_list.append(F1)
            auc_list.append(AUC)
    return np.mean(acc_list), f1_list, auc_list

def valen(args,num_features,num_classes,train_loader,device,test_loader):
    train_X = train_loader.dataset.images
    net, enc, dec = create_model(args, num_features=num_features, num_classes=num_classes)
    net, enc, dec = map(lambda x: x.to(device), (net, enc, dec))
    if args.dt == 'benchmark':
        net, feature_extracted, o_array = warm_up_benchmark(args, net, train_loader, test_loader,device)
        print("here")
        adj = gen_adj_matrix2(feature_extracted.cpu().numpy(), k=3, dir_path=os.path.abspath("valen/middle/adjmatrix/"+args.dt),path=os.path.abspath("valen/middle/adjmatrix/"+args.dt+"/"+args.ds+".npy"))
    else:
        net, o_array = warm_up_realworld(args, net, train_loader, test_loader , device)
        adj = gen_adj_matrix2(train_X.cpu().numpy(), k=3, dir_path="valen/middle/adjmatrix/" + args.dt,path=os.path.abspath("valen/middle/adjmatrix/" + args.dt + "/" + args.ds + ".npy"))
        if train_X.shape[0] < adj.shape[0]:
            adj = gen_adj_matrix2(train_X.cpu().numpy(), k=3, dir_path="valen/middle/adjmatrix/" + args.dt,path=os.path.abspath("valen/middle/adjmatrix/" + args.dt + "/" + args.ds + ".npy"))

    d_array = deepcopy(o_array)
    if  args.ds in ["kmnist", "cifar10"]:
        print("Copy Net.")
        enc = deepcopy(net)
    # compute adj matrix
    print("Compute adj maxtrix or Read.")
    # adj = gen_adj_matrix2(feature_extracted.cpu().numpy(), k=3, dir_path=os.path.abspath("valen/middle/adjmatrix/"+args.dt),path=os.path.abspath("valen/middle/adjmatrix/"+args.dt+"/"+args.ds+".npy"))
    A = adj.to_dense()
    adj = adj.to(device)
    if args.ds == "cifar10":
        print("Use SGD with 0.9 momentum")
        opt = torch.optim.SGD(list(net.parameters()) + list(enc.parameters()) + list(dec.parameters()), lr=args.lr,
                              weight_decay=args.wd, momentum=0.9)
    else:
        print("Use Adam.")
        opt = torch.optim.Adam(list(net.parameters()) + list(enc.parameters()) + list(dec.parameters()), lr=args.lr,
                               weight_decay=args.wd)
    # compute gcn embedding
    embedding = train_X.to(device)
    prior_alpha = torch.Tensor(1, num_classes).fill_(1.0).to(device)
    for epoch in range(0, args.ep):
        for features, targets, trues, indexes in train_loader:
            features, targets, trues = map(lambda x: x.to(device), (features, targets, trues))
            _, outputs = net(features)
            _, alpha = enc(embedding[indexes, :])
            s_alpha = F.softmax(alpha, dim=1)
            revised_alpha = torch.zeros_like(targets)
            revised_alpha[o_array[indexes, :] > 0] = 1.0
            s_alpha = s_alpha * revised_alpha
            s_alpha_sum = s_alpha.clone().detach().sum(dim=1, keepdim=True)
            s_alpha = s_alpha / s_alpha_sum + 1e-2
            L_d, new_d = partial_loss(alpha, o_array[indexes, :], None)
            alpha = torch.exp(alpha / 4)
            alpha = F.hardtanh(alpha, min_val=1e-2, max_val=30)
            L_alpha = alpha_loss(alpha, prior_alpha)
            dirichlet_sample_machine = torch.distributions.dirichlet.Dirichlet(s_alpha)
            d = dirichlet_sample_machine.rsample()
            x_hat = dec(d)
            x_hat = x_hat.view(features.shape)
            A_hat = F.softmax(dot_product_decode(d), dim=1)
            L_recx = 0.01 * F.mse_loss(x_hat, features)
            L_recy = 0.01 * F.binary_cross_entropy_with_logits(d, targets)
            L_recA = F.mse_loss(A_hat, A[indexes, :][:, indexes].to(device))
            L_rec = L_recx + L_recy + L_recA
            L_o, new_o = partial_loss(outputs, d_array[indexes, :], None)
            L =  L_rec + 0.1 * L_alpha + 5 * L_d +  L_o
            opt.zero_grad()
            L.backward()
            opt.step()
            new_d = revised_target(d, new_d)
            # new_d = args.correct * new_d + (1 - args.correct) * o_array[indexes, :]
            d_array[indexes, :] = new_d.clone().detach()
            o_array[indexes, :] = new_o.clone().detach()
        test_acc, auc, f1,  = test_accuracy_check_ac(test_loader, net,  device,1)
        print('epoch:', epoch + 1, 'acc:', str(test_acc)[:8], 'F1:', str(f1)[:8], 'AUC:', str(auc)[:8])
        if epoch >= (args.ep - 10):
            acc_list.append(test_acc)
            f1_list.append(f1)
            auc_list.append(auc)
    return np.mean(acc_list),f1_list,auc_list

def gen(args,model,train_loader,optimizer,device,test_loader,confidence):
    for epoch in range(args.ep):
        model.train()
        for i, (images, labels, true_labels, index) in enumerate(train_loader):
            X, Y, index = images.to(device), labels.to(device), index.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            K = outputs.shape[1]
            if args.al == 'mae':
                loss_fn = mae_loss
            elif args.al == 'mse':
                loss_fn = mse_loss
            elif args.al == 'exp':
                loss_fn = exp_loss
            elif args.al == 'log':
                loss_fn = log_loss
            elif args.al == 'gce':
                loss_fn = gce_loss
            elif args.al == 'pce':
                loss_fn = phuber_ce_loss
            if args.al != 'pce':
                average_loss = loss_fn(outputs, Y)
            else:
                average_loss = loss_fn(outputs, Y,device)
            average_loss.backward()
            optimizer.step()
        model.eval()
        train_accuracy = train_accuracy_check(loader=train_loader,
                                              model=model,
                                              device=device)
        AUC, F1, test_accuracy = test_accuracy_check_ac(loader=test_loader,
                                                     model=model,
                                                     device=device)
        print('epoch:', epoch + 1, 'train_acc', str(train_accuracy)[:8], 'test_acc:', str(test_accuracy)[:8], 'F1:',
              str(F1)[:8], 'AUC:', str(AUC)[:8])
        if epoch >= (args.ep - 10):
            acc_list.append(test_accuracy)
            f1_list.append(F1)
            auc_list.append(AUC)
    return np.mean(acc_list), f1_list, auc_list

def rc(args,model,train_loader,optimizer,device,test_loader,confidence):
    for epoch in range(args.ep):
        model.train()
        for i, (images, labels, true_labels, index) in enumerate(train_loader):
            X, Y, index = images.to(device), labels.to(device), index.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            K = outputs.shape[1]
            average_loss = rc_loss(outputs, confidence, index)
            average_loss.backward()
            optimizer.step()
            confidence = confidence_update(model, confidence, X, Y, index)
        model.eval()
        train_accuracy = train_accuracy_check(loader=train_loader,
                                              model=model,
                                              device=device)
        AUC, F1, test_accuracy = test_accuracy_check_ac(loader=test_loader,
                                                     model=model,
                                                     device=device)
        print('epoch:', epoch + 1, 'train_acc', str(train_accuracy)[:8], 'test_acc:', str(test_accuracy)[:8], 'F1:',
              str(F1)[:8], 'AUC:', str(AUC)[:8])
        if epoch >= (args.ep - 10):
            acc_list.append(test_accuracy)
            f1_list.append(F1)
            auc_list.append(AUC)
    return np.mean(acc_list), f1_list, auc_list

def cc(args,model,train_loader,optimizer,device,test_loader,confidence):
    for epoch in range(args.ep):
        model.train()
        for i, (images, labels, true_labels, index) in enumerate(train_loader):
            X, Y, index = images.to(device), labels.to(device), index.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            average_loss = cc_loss(outputs, Y.float())
            average_loss.backward()
            optimizer.step()
        model.eval()
        train_accuracy = train_accuracy_check(loader=train_loader,
                                              model=model,
                                              device=device)
        AUC, F1, test_accuracy = test_accuracy_check_ac(loader=test_loader,
                                                     model=model,
                                                     device=device)
        print('epoch:', epoch + 1, 'train_acc', str(train_accuracy)[:8], 'test_acc:', str(test_accuracy)[:8], 'F1:',
              str(F1)[:8], 'AUC:', str(AUC)[:8])
        if epoch >= (args.ep - 10):
            acc_list.append(test_accuracy)
            f1_list.append(F1)
            auc_list.append(AUC)
    return np.mean(acc_list), f1_list, auc_list