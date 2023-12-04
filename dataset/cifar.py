import logging
import math
import medmnist
<<<<<<< HEAD
from medmnist import INFO
=======
from medmnist import INFO, Evaluator
>>>>>>> 995281dc8279565590a78c9810613896e8904aa3
import random

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from . import cifar10_inds
from . import cifar100_inds
from . import MedMNIST_inds
from .randaugment import RandAugmentMC

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar10_mean_cld_aug = (0.4914, 0.4822, 0.4465)
cifar10_std_cld_aug = (0.2023, 0.1994, 0.2010)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_cifar10(args, root, force_no_expand=False):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets, force_no_expand=force_no_expand)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset



def get_cifar10_cld_aug(args, root, force_no_expand=False):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean_cld_aug, std=cifar10_std_cld_aug)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean_cld_aug, std=cifar10_std_cld_aug)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets, force_no_expand=force_no_expand)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

def get_cifar100(args, root, force_no_expand=False):

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(
        root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets, force_no_expand=force_no_expand)

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std))

    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_medmnist(args,force_no_expand=False):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=28,
                              padding=int(28*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)
    ])

    info = INFO[args.MedMNIST]
    DataClass = getattr(medmnist, info['python_class'])
    base_dataset = DataClass(split='train', download=True)

    # base_dataset = datasets.CIFAR10(root, train=True, download=True)
    # targets = np.load("/data/UnsupervisedSelectiveLabeling/semisup-fixmatch-cifar/target/DermaMNIST_target.npy")

    # 这里的target是个问题，还有x_u_split/MedMNIST没改完
    # train_labeled_idxs, train_unlabeled_idxs = x_u_split(
    #     args, base_dataset.targets, force_no_expand=force_no_expand)
    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.labels, force_no_expand=force_no_expand)

    train_labeled_dataset = MedMNISTSSL(
        train_labeled_idxs, split='train',
        transform=transform_labeled)

    train_unlabeled_dataset = MedMNISTSSL(
        train_unlabeled_idxs, split='train',
        transform=MedTransformFixMatch(mean=normal_mean, std=normal_std))
    
    test_dataset = DataClass(split='test', transform=transform_val,  download=False, index = None)

    sample_db = make_imb_data(243, 8, 1)
    imb_idxs = createImbIdxs(test_dataset.labels, sample_db)
    random.seed(10)
    random.shuffle(imb_idxs)

    test_dataset = DataClass(split='test', transform=transform_val,  download=False, index = imb_idxs)

    # test_dataset = datasets.CIFAR10(
    #     root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_labeled_inds(args, labels):
    print("Get sliced labels")
    seed = int(args.get_labeled_inds.split("slice")[1])
    print("Using seed {}".format(seed))
    random_state = np.random.RandomState(seed=seed)
    label_per_class = args.num_labeled // args.num_classes
    labeled_idx = []
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = random_state.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled
    print(labeled_idx)
    return labeled_idx

def x_u_split(args, labels, force_no_expand=False):
    
    labels = np.array(labels)
    
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    
    if args.get_labeled_inds.startswith("slice"):
        labeled_idx = get_labeled_inds(args, labels)
    elif "random" in args.get_labeled_inds:
        if args.dataset == "cifar10":
            labeled_idx = cifar10_inds.get_labeled_inds_random(args, labels, args.get_labeled_inds)
        elif args.dataset == "cifar100":
            labeled_idx = cifar100_inds.get_labeled_inds_random(args, labels, args.get_labeled_inds)
    elif "selected" in args.get_labeled_inds:
        if args.dataset == "cifar10":
            labeled_idx = cifar10_inds.get_labeled_inds_select(args, labels, args.get_labeled_inds)
        elif args.dataset == "cifar100":
            labeled_idx = cifar100_inds.get_labeled_inds_select(args, labels, args.get_labeled_inds)
        elif args.dataset == "MedMNIST":
            labeled_idx = MedMNIST_inds.get_labeled_inds_select(args, labels, args.get_labeled_inds)
    else:
        raise ValueError("get_labeled_inds not in type")

    labeled_class_freq = np.zeros(10 if args.dataset == "cifar10" else 100)
    for ind, cnt in zip(*np.unique(labels[labeled_idx], return_counts=True)):
        labeled_class_freq[ind] = cnt
    labeled_class_freq = labeled_class_freq / labeled_class_freq.sum()

    args.labeled_class_freq = labeled_class_freq

    if not force_no_expand and (args.expand_labels or args.num_labeled < args.batch_size):
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx

def createImbIdxs(labels, n_data_per_class) :
    '''
    Creates a List containing Indexes of the Imbalanced Classification

    Input: 
        labels: Ground Truth of Dataset
        n_data_per_class: Class Distribution of Dataset desired

    Output:
        data_idxs: List containing indexes for Dataset 
    '''
    labels = np.array(labels) # Classification Ground Truth 
    data_idxs = []  # Collect Ground Truth Indexes

    for i in range(len(n_data_per_class) ) :
        idxs = np.where(labels == i)[0]
        data_idxs.extend(idxs[ :n_data_per_class[i] ])

    return data_idxs

def checkReverseDistb(imb_ratio) :
    reverse = False
    if imb_ratio / abs(imb_ratio) == -1 :
        reverse = True
        imb_ratio = imb_ratio * -1

    return reverse, imb_ratio

def make_imb_data(max_num, class_num, gamma):
    reverse, gamma = checkReverseDistb(gamma)

    mu = np.power(1/gamma, 1/(class_num - 1))
    class_num_list = []
    for i in range(class_num):
        if i == (class_num - 1):
            class_num_list.append(int(max_num / gamma))
        else:
            class_num_list.append(int(max_num * np.power(mu, i)))
    if reverse :
        class_num_list.reverse()
    #print(class_num_list)
    return list(class_num_list)

class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

class MedTransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=28,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=28,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


info = INFO["bloodmnist"]
DataClass = getattr(medmnist, info['python_class'])


class MedMNISTSSL(DataClass):
    def __init__(self, indexs, split="train",
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(split=split,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.imgs = self.imgs[indexs]
            self.labels = np.array(self.labels)[indexs]

    def __getitem__(self, index):
        img, target = self.imgs[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100,
                   'cifar10_cld_aug': get_cifar10_cld_aug,
                   'MedMNIST':get_medmnist}

##
def createImbIdxs(labels, n_data_per_class) :
    '''
    Creates a List containing Indexes of the Imbalanced Classification

    Input: 
        labels: Ground Truth of Dataset
        n_data_per_class: Class Distribution of Dataset desired

    Output:
        data_idxs: List containing indexes for Dataset 
    '''
    labels = np.array(labels) # Classification Ground Truth 
    data_idxs = []  # Collect Ground Truth Indexes

    for i in range(len(n_data_per_class) ) :
        idxs = np.where(labels == i)[0]
        data_idxs.extend(idxs[ :n_data_per_class[i] ])

    return data_idxs

def checkReverseDistb(imb_ratio) :
    reverse = False
    if imb_ratio / abs(imb_ratio) == -1 :
        reverse = True
        imb_ratio = imb_ratio * -1

    return reverse, imb_ratio

def make_imb_data(max_num, class_num, gamma):
    reverse, gamma = checkReverseDistb(gamma)

    mu = np.power(1/gamma, 1/(class_num - 1))
    class_num_list = []
    for i in range(class_num):
        if i == (class_num - 1):
            class_num_list.append(int(max_num / gamma))
        else:
            class_num_list.append(int(max_num * np.power(mu, i)))
    if reverse :
        class_num_list.reverse()
    #print(class_num_list)
    return list(class_num_list)
