import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from scipy.special import comb
from utils.gen_index_dataset import gen_index_dataset
from scipy.io import loadmat
import scipy.io as sio
import math


def generate_uniform_cv_candidate_labels(dataname, train_labels):
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = torch.max(train_labels) - torch.min(train_labels) + 1 # 类别
    n = train_labels.shape[0]
    cardinality = (2 ** K - 2).float()
    number = torch.tensor([comb(K, i + 1) for i in range(
        K - 1)]).float()  # 1 to K-1 不能是空的或者全部标签都在
    frequency_dis = number / cardinality
    prob_dis = torch.zeros(K - 1)  # tensor of K-1
    for i in range(K - 1):
        if i == 0:
            prob_dis[i] = frequency_dis[i]
        else:
            prob_dis[i] = frequency_dis[i] + prob_dis[i - 1]

    random_n = torch.from_numpy(np.random.uniform(0, 1, n)).float()  # tensor: n
    mask_n = torch.ones(n)  # n 是训练数据的数量
    partialY = torch.zeros(n, K)

    partialY[torch.arange(n), train_labels] = 1.0
    temp_num_partial_train_labels = 0  # 候选标记的个数

    for j in range(n):
        for jj in range(K - 1):  # 0 to K-2
            if random_n[j] <= prob_dis[jj] and mask_n[j] == 1:
                temp_num_partial_train_labels = jj + 1
                mask_n[j] = 0

        temp_num_fp_train_labels = temp_num_partial_train_labels - 1
        candidates = torch.from_numpy(np.random.permutation(K.item())).long()  # because K is tensor type
        candidates = candidates[candidates != train_labels[j]]
        temp_fp_train_labels = candidates[:temp_num_fp_train_labels]

        partialY[j, temp_fp_train_labels] = 1.0  # fulfill the partial label matrix
    print("Finish Generating Candidate Label Sets!")
    return partialY

def prepare_cv_datasets(dataname, batch_size):
    if dataname == 'mnist':
        ordinary_train_dataset = dsets.MNIST(root='./data/mnist', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = dsets.MNIST(root='./data/mnist', train=False, transform=transforms.ToTensor())
    elif dataname == 'kmnist':
        ordinary_train_dataset = dsets.KMNIST(root='../data/KMNIST', train=True, transform=transforms.ToTensor(), download=False)
        test_dataset = dsets.KMNIST(root='../data/KMNIST', train=False, transform=transforms.ToTensor())
    elif dataname == 'fashion':
        ordinary_train_dataset = dsets.FashionMNIST(root='../data/FashionMnist', train=True, transform=transforms.ToTensor(), download=False)
        test_dataset = dsets.FashionMNIST(root='../data/FashionMnist', train=False, transform=transforms.ToTensor())
    elif dataname == 'cifar10':
        train_transform = transforms.Compose(
            [transforms.ToTensor(), # transforms.RandomHorizontalFlip(), transforms.RandomCrop(32,4),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        ordinary_train_dataset = dsets.CIFAR10(root='../data', train=True, transform=train_transform, download=False)
        test_dataset = dsets.CIFAR10(root='../data', train=False, transform=test_transform)
    elif dataname == 'svhn':
        train_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        ordinary_train_dataset = dsets.SVHN(root='../data', split="train", transform=train_transform, download=False)
        test_dataset = dsets.SVHN(root='../data', split="test", transform=test_transform)
    train_loader = torch.utils.data.DataLoader(dataset=ordinary_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    full_train_loader = torch.utils.data.DataLoader(dataset=ordinary_train_dataset, batch_size=len(ordinary_train_dataset.data), shuffle=True, num_workers=0)
    full_test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset.data),shuffle = True,num_workers = 0)
    num_classes = 10
    return full_train_loader, train_loader, test_loader, full_test_loader, ordinary_train_dataset, test_dataset, num_classes

def prepare_train_loaders_for_uniform_cv_candidate_labels(args, full_train_loader, batch_size):
    # if uci == 0:
    if args.dt != 'uci':
        for i, (data, labels) in enumerate(full_train_loader):
            K = torch.max(labels)+1 # K is number of classes, full_train_loader is full batch
    else:
        for i,(data,labels,labels,index) in enumerate(full_train_loader):
            K = torch.max(labels)+1
    partialY = generate_uniform_cv_candidate_labels(data, labels)
    # partial_matrix_dataset = gen_index_dataset(data, partialY.float(), partialY.float())
    partial_matrix_dataset = gen_index_dataset(data,partialY.float(),labels)
    # if uci ==1 :
    #     partial_matrix_train_loader = torch.utils.data.DataLoader(dataset=partial_matrix_dataset, batch_size=data.shape[0],
    #                                                               shuffle=True, num_workers=0)
    # else:
    #     partial_matrix_train_loader = torch.utils.data.DataLoader(dataset=partial_matrix_dataset, batch_size=data.shape[0], shuffle=True, num_workers=0)
    dim = int(data.reshape(-1).shape[0]/data.shape[0])
    # exit()
    return partial_matrix_dataset, data, partialY, dim

def make_cv_mpu_train_set(full_train_loader, full_test_loader, labeled_num, unlabeled_num, test_num, args, label_rate,theta):

    print("Step 1: generate candidate sets")
    partial_matrix_dataset, data, partialY, dim = prepare_train_loaders_for_uniform_cv_candidate_labels(
        args=args, full_train_loader=full_train_loader, batch_size=args.bs)
    data,partial_targets,labels = partial_matrix_dataset.images,partial_matrix_dataset.given_label_matrix,partial_matrix_dataset.true_labels
    for i, (test_data, test_label) in enumerate(full_test_loader):
        # print("\n")  10000
        print("load_full_test_data!")
    label_set = torch.unique(labels, sorted=True)  # 计算label的个数
    class_prior = torch.zeros(label_set.shape)
    y_hat = labels.clone()
    unlabel_class_num = 0
    for i in range(len(label_set)):
        is_equal = (labels== label_set[i])
        index = torch.arange(0, labels.shape[0])
        selected_index = index[is_equal]
        class_prior[i] = selected_index.shape[0] / labels.shape[0]  # 某个class在整个label中的占比

    selected_index = torch.zeros(labels.shape[0])
    print("Step 2: choose unlabel class, at least 30% of total train data")
    for label in label_set:
        unlabel_class_num += 1
        is_in = (partial_targets[:, label] == 1)
        index = torch.arange(0, labels.shape[0])
        selected_index[index[is_in]] = 1
        is_unlabel = (selected_index == 1)
        unlabel_num = len(index[is_unlabel])
        print(unlabel_num)
        if unlabel_num > 0.3 * labels.shape[0]:  # 当选择的unlabel总数量占比大于0.3  按照数量进行选取
            break
        # if unlabel_class_num > unlabel_class:
        #     break
    labeled_set = label_set[unlabel_class_num:]
    unlabeled_set = label_set[:unlabel_class_num]
    K = len(labeled_set)+1  # 当用的是correct consistent 和unbiased这里需要变成 len(labeled_set)+1,即包括ac在内
    partialY = partialY[:,unlabel_class_num:]
    ac = torch.zeros(size=(data.shape[0], 1))
    partialY_without_ac = partialY
    partialY = torch.cat((partialY,ac),dim=1)
    unlabel_selected_index = index[is_unlabel]
    unlabel_train = len(unlabel_selected_index)//1000*1000  # 选出的unlabel一部分用于训练一部分用于测试
    print('unlabel num: ', unlabel_train) # 此时是将所有unlabel都放入训练
    y_hat[unlabel_selected_index[:unlabel_train]] = -2  # unlabel train
    is_label = (is_unlabel == False)  # 需要从前面unlabel剩余的数据中进行挑选
    label_num = len(index[is_label])//1000*1000

    print('label num: ', label_num)
    ## unlabeled set
    X_unlabeled_train = data[is_unlabel]
    labels[is_unlabel] = labeled_set[-1] - unlabel_class_num +1 # 应该是labeled_set的最后一个大一
    y_unlabeled_train = labels[is_unlabel]
    y_unlabeled_given_Y = torch.zeros(len(partialY[is_unlabel]), K)
    y_unlabeled_given_Y[:, -1] = 1.0 # unlabel的标记为[0,0,0,...,0,1]格式,最后一个表示ac
    ### labeled set
    X_labeled_train = data[is_label][:label_num]
    y_labeled_train = labels[is_label][:label_num]- unlabel_class_num
    y_labeled_given_Y = partialY[is_label][:label_num]
    y_labeled_given_Y_without_ac = partialY_without_ac[is_label][:label_num]

    # unlabel_train = args.un
    X_unlabeled_train = X_unlabeled_train[:unlabel_train]
    y_unlabeled_given_Y = y_unlabeled_given_Y[:unlabel_train]
    y_unlabeled_train = y_unlabeled_train[:unlabel_train]
    data = torch.cat((X_labeled_train,X_unlabeled_train),dim=0)
    partialY = torch.cat((y_labeled_given_Y,y_unlabeled_given_Y),dim=0)
    label = torch.cat((y_labeled_train,y_unlabeled_train),dim=0)
    train_dataset = gen_index_dataset(data,partialY,label)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=0)
    print(X_labeled_train.shape,y_labeled_given_Y_without_ac.shape,y_labeled_train.shape)
    train_without_ac_dataset = gen_index_dataset(X_labeled_train, y_labeled_given_Y_without_ac, y_labeled_train)
    train_without_ac_loader = torch.utils.data.DataLoader(dataset=train_without_ac_dataset, batch_size=args.bs,
                                                          shuffle=True, num_workers=0)

    print("Step 3: revise test samples' labels which is in unlabel set")
    selected_index = torch.arange(0, test_data.shape[0])
    for label in unlabeled_set:
        test_unlabel_data = (label == test_label)
        print(len(selected_index[test_unlabel_data]))
        test_label[selected_index[test_unlabel_data]] = -2  # 先命成-1
    for label in labeled_set:
        test_label_data = (label == test_label)
        test_label[selected_index[test_label_data]] = label - unlabel_class_num
    test_unlabel_data = ( test_label == -2 )
    test_label[selected_index[test_unlabel_data]] = labeled_set[-1] - unlabel_class_num + 1  # ac要+1
    # 只选属于label_set的data
    test_dataset = gen_index_dataset(test_data,test_label,test_label)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=True, num_workers=0)
    return train_loader,test_loader,partialY,train_without_ac_loader,y_labeled_given_Y_without_ac,dim, K,X_labeled_train,X_unlabeled_train

def make_realworld_mpu_train_set_v3(ds,batch_size):
    mat_path = "../data/realworld/" + ds + '.mat'
    data = loadmat(mat_path)
    data, targets, partial_targets = data['data'], data['target'], data['partial_target']
    data = (data - data.mean(axis=0, keepdims=True)) / data.std(axis=0, keepdims=True)
    print(len(data))
    data = torch.from_numpy(data).float()
    dim = data.shape[1]
    targets = np.transpose(targets)
    partial_targets = np.transpose(partial_targets)
    targets = torch.from_numpy(targets).float()
    partial_targets = torch.from_numpy(partial_targets).float()
    random_index = torch.randperm(data.shape[0])  # 打乱样本的顺序
    data = data[random_index]
    targets = targets[random_index]
    partial_targets = partial_targets[random_index]
    labels = torch.argmax(targets, dim=1)  # convert one-hot to integer
    if torch.min(labels) == 1:
        labels = labels - 1
    label_set = torch.unique(labels, sorted=True)
    print('Step 1: split dataset into 0.8 trainset and 0.2 testset by class')
    y_hat = labels.clone()
    total_train = 0
    total_test = 0
    for label in label_set:
        is_equal = (labels == label)
        index = torch.arange(0, labels.shape[0])
        selected_index = index[is_equal]
        #####################
        train_num = int(len(selected_index) * 0.8)
        test_num = math.ceil(len(selected_index) * 0.2)
        total_train += train_num
        total_test += test_num
        y_hat[selected_index[:test_num]] = -2  # pick test data test在前防止测试集里对应class数量为0
        y_hat[selected_index[test_num:train_num + test_num]] = -1  # pick train data
    print("train num: ", total_train, "test num: ", total_test)
    is_train = (y_hat == -1)
    is_test = (y_hat == -2)
    train_data = data[is_train]
    train_label = labels[is_train]
    partial_targets = partial_targets[is_train]
    test_data = data[is_test]
    test_label = labels[is_test]
    labels = train_label
    data = train_data
    num = train_data.shape[0]
    y_hat = labels.clone()
    unlabel_class_num = 0
    selected_index = torch.zeros(labels.shape[0])
    print('Step 2: choose unlabel class, at least 30% of total train data')
    for label in label_set:
        unlabel_class_num += 1
        is_in = (partial_targets[:, label] == 1)
        index = torch.arange(0, labels.shape[0])
        selected_index[index[is_in]] = 1
        is_unlabel = (selected_index == 1)
        unlabel_num = len(index[is_unlabel])
        if unlabel_num > 0.3 * num: # 当选择的unlabel总数量占比大于0.3
            break
    partial_targets = partial_targets[:, :-2] # lost需要进行这步
    labeled_set = label_set[unlabel_class_num:]
    unlabeled_set = label_set[:unlabel_class_num]
    partial_targets = partial_targets[:,unlabel_class_num:]
    K = len(labeled_set) +1   # 当用的是correct consistent 和unbiased这里需要变成 len(labeled_set)+1,即包括ac在内
    unlabel_selected_index = index[is_unlabel]
    unlabel_train = len(unlabel_selected_index)
    print('unlabel num: ',unlabel_train)
    is_label = (is_unlabel == False) #需要从前面unlabel剩余的数据中进行挑选
    label_train = len(index[is_label])
    print('label num: ',label_train)
    # 剩下的都是候选标记中不存在unlabel的数据
    X_unlabeled_train = data[is_unlabel]
    labels[is_unlabel] = labeled_set[-1]-unlabel_class_num+1
    y_unlabeled_train = labels[is_unlabel]

    y_labeled_given_Y_without_ac = partial_targets
    ac = torch.zeros(size=(data.shape[0], 1))
    partialY = torch.cat((partial_targets, ac), dim=1)
    y_unlabeled_partial_train = torch.zeros(len(partialY[is_unlabel]), K)
    y_unlabeled_partial_train[:, -1] = 1.0 # unlabel的标记为[0,0,0,...,0,1]格式,最后一个表示ac

    X_labeled_train = data[is_label]
    y_labeled_train = labels[is_label]-unlabel_class_num
    y_labeled_train_given_Y = partialY[is_label]
    y_labeled_given_Y_without_ac = partial_targets[is_label]
    data = torch.cat((X_labeled_train, X_unlabeled_train), dim=0)
    labels = torch.cat((y_labeled_train, y_unlabeled_train), dim=0)
    print(y_labeled_train_given_Y.shape,y_unlabeled_partial_train.shape)
    train_givenY = torch.cat((y_labeled_train_given_Y,y_unlabeled_partial_train),dim=0)
    train_dataset = gen_index_dataset(data,train_givenY,labels) # 包括unlabel
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_without_ac_dataset = gen_index_dataset(X_labeled_train, y_labeled_given_Y_without_ac, y_labeled_train)
    train_without_ac_loader = torch.utils.data.DataLoader(dataset=train_without_ac_dataset, batch_size=batch_size,
                                                          shuffle=True, num_workers=0)

    selected_index = torch.arange(0, test_data.shape[0])
    for label in unlabeled_set:
        test_unlabel_data = (label == test_label)
        test_label[selected_index[test_unlabel_data]] = -2
    for label in labeled_set:
        test_label_data = (label == test_label)
        test_label[selected_index[test_label_data]] = label - unlabel_class_num
    test_unlabel_data = (test_label == -2)
    test_label[selected_index[test_unlabel_data]] = labeled_set[-1] - unlabel_class_num + 1  # ac要+1
    test_dataset = gen_index_dataset(test_data,test_label,test_label)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=True, num_workers=0)
    return dim,K,train_givenY,train_loader,test_loader,train_without_ac_loader,y_labeled_given_Y_without_ac,X_labeled_train,X_unlabeled_train


def make_uci_mpu_train_set(args, num_labeled, num_unlabeled, num_test, seed):
    dataname = './dataset/' + args.ds + '.mat'
    current_data = sio.loadmat(dataname)
    data = current_data['data']
    label = current_data['label']
    data = torch.from_numpy(data).float()
    label = torch.from_numpy(label).float()
    random_index = torch.randperm(data.shape[0])  # 打乱样本的顺序
    print("dataset:",len(data))
    data = data[random_index]
    label = label[random_index]
    labels = label.argmax(dim=1)  # convert one-hot to integer
    label_set = torch.unique(labels, sorted=True)
    print('Step 1: split dataset into 0.8 trainset and 0.2 testset by each class')
    y_hat = labels.clone()
    total_train = 0
    total_test = 0
    for label in label_set:
        is_equal = (labels == label)
        index = torch.arange(0, labels.shape[0])
        selected_index = index[is_equal]
        #####################
        train_num = int(len(selected_index)*0.8)
        test_num = int(len(selected_index)*0.2)
        total_train += train_num
        total_test += test_num
        y_hat[selected_index[:train_num]] = -1  # pick train data
        y_hat[selected_index[train_num:train_num + test_num]] = -2 # pick test data
    print("train num: ",total_train,"test num: ",total_test)
    is_train = (y_hat == -1)
    is_test = (y_hat == -2)
    train_data = data[is_train]
    train_label = labels[is_train]
    train_dataset = gen_index_dataset(train_data,train_label,train_label)
    test_data = data[is_test]
    test_label = labels[is_test]
    full_train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size = len(train_dataset), shuffle = True, num_workers = 0)
    print("Step 2: generate candidate sets")
    partial_matrix_dataset, data, partialY, dim = prepare_train_loaders_for_uniform_cv_candidate_labels(
        args= args, full_train_loader=full_train_loader, batch_size=args.bs)
    data, partial_targets, labels = partial_matrix_dataset.images, partial_matrix_dataset.given_label_matrix, partial_matrix_dataset.true_labels
    if torch.min(labels) == 1:
        labels = labels - 1
    label_set = torch.unique(labels, sorted=True)
    selected_index = torch.zeros(data.shape[0])
    unlabel_class_num = 0
    print('Step 3: choose unlabel class, at least 30% of total train data')
    for label in label_set:
        unlabel_class_num += 1
        is_in = (partial_targets[:, label] == 1)
        index = torch.arange(0, labels.shape[0])
        selected_index[index[is_in]] = 1
        is_unlabel = (selected_index == 1)
        unlabel_num = len(index[is_unlabel])
        if unlabel_num > 0.3 * labels.shape[0]:  # 当选择的unlabel总数量占比大于0.3
            break
    labeled_set = label_set[unlabel_class_num:]
    unlabeled_set = label_set[:unlabel_class_num]
    K = len(labeled_set)+1  # 当用的是correct consistent 和unbiased这里需要变成 len(labeled_set)+1,即包括ac在内
    partialY = partialY[:,unlabel_class_num:]
    ac = torch.zeros(size=(data.shape[0], 1))
    y_labeled_given_Y_without_ac = partialY
    partialY = torch.cat((partialY,ac),dim=1)
    unlabel_selected_index = index[is_unlabel]
    unlabel_train = len(unlabel_selected_index)//100*100  # 选出的unlabel一部分用于训练一部分用于测试
    # unlabel_train = unlabel # 增加unlabel判断无偏性
    print('unlabel num: ', unlabel_train," ",len(unlabel_selected_index)) # 此时是将所有unlabel都放入训练
    is_label = (is_unlabel == False)  # 需要从前面unlabel剩余的数据中进行挑选
    label_train = len(index[is_label])//100*100
    print('label num: ', label_train," ",len(index[is_label]))
    ## unlabeled set
    X_unlabeled_train = data[is_unlabel]

    labels[is_unlabel] = labeled_set[-1] - unlabel_class_num +1 # 应该是labeled_set的最后一个大一
    y_unlabeled_train = labels[is_unlabel]
    y_unlabeled_given_Y = torch.zeros(len(partialY[is_unlabel]), K)
    y_unlabeled_given_Y[:, -1] = 1.0 # unlabel的标记为[0,0,0,...,0,1]格式,最后一个表示ac
    ### labeled set
    X_labeled_train = data[is_label][:label_train]
    y_labeled_train = labels[is_label][:label_train]- unlabel_class_num
    y_labeled_given_Y = partialY[is_label][:label_train]
    y_labeled_given_Y_without_ac = y_labeled_given_Y_without_ac[is_label][:label_train]
    X_unlabeled_train = X_unlabeled_train[:unlabel_train]
    y_unlabeled_given_Y = y_unlabeled_given_Y[:unlabel_train]
    y_unlabeled_train = y_unlabeled_train[:unlabel_train]
    data = torch.cat((X_labeled_train,X_unlabeled_train),dim=0)
    partialY = torch.cat((y_labeled_given_Y,y_unlabeled_given_Y),dim=0)
    label = torch.cat((y_labeled_train,y_unlabeled_train),dim=0)
    train_dataset = gen_index_dataset(data,partialY,label)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=0)
    train_without_ac_dataset = gen_index_dataset(X_labeled_train,y_labeled_given_Y_without_ac,y_labeled_train)
    train_without_ac_loader = torch.utils.data.DataLoader(dataset=train_without_ac_dataset, batch_size=args.bs, shuffle=True, num_workers=0)
    print("Step 3: revise test samples' labels which is in unlabel set")
    selected_index = torch.arange(0, test_data.shape[0])
    for label in unlabeled_set:
        test_unlabel_data = (label == test_label)
        test_label[selected_index[test_unlabel_data]] = -2
    for label in labeled_set:
        test_label_data = (label == test_label)
        test_label[selected_index[test_label_data]] = label - unlabel_class_num
    test_unlabel_data = (test_label == -2)
    test_label[selected_index[test_unlabel_data]] = labeled_set[-1] - unlabel_class_num + 1  # ac要+1
    test_dataset = gen_index_dataset(test_data,test_label,test_label)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=True, num_workers=0)
    print("out of make uci")
    return train_loader, test_loader, partialY,y_labeled_given_Y_without_ac,train_without_ac_loader, dim, K,X_labeled_train,X_unlabeled_train

