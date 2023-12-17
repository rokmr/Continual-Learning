import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR10, iCIFAR100, iImageNet100, iImageNet1000, iCIFAR100_224, iImageNetR, iCUB200_224, iResisc45_224, iCARS196_224, iSketch345_224
from copy import deepcopy
import random
from utils.nncsl_functions import init_transform

class DataManager(object): # _train() in trainer.py calls this class
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment):
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)
        assert init_cls <= len(self._class_order), 'No enough classes.'
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)   #Finally self._increments: [10, 10, 10, 10, 10, 10, 10, 10, 10]
        offset = len(self._class_order) - sum(self._increments) #offset: 10
        if offset > 0:
            self._increments.append(offset) #Now self._increments: [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task): # incremental_train() in slca.py calls this function
        return self._increments[task] 

    def get_dataset(self, indices, source, mode, tasks , task_idx, buffer_lst, appendent=None, ret_data=False, with_raw=False, with_noise=False, keep_file= None): # incremental_train() in slca.py calls this function
        if source == 'train':
            x, y = self._train_data, self._train_targets
        elif source == 'test':
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        if mode == 'train':
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
            training = True
        elif mode == 'flip':
            trsf = transforms.Compose([*self._test_trsf, transforms.RandomHorizontalFlip(p=1.), *self._common_trsf])
        elif mode == 'test':
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
            training = False
        else:
            raise ValueError('Unknown mode {}.'.format(mode))

        # data, targets = [], []
        # for idx in indices:
        #     class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx+1)
        #     data.append(class_data)
        #     targets.append(class_targets)

        # if appendent is not None and len(appendent) != 0:
        #     appendent_data, appendent_targets = appendent
        #     data.append(appendent_data)
        #     targets.append(appendent_targets)

        # data, targets = np.concatenate(data), np.concatenate(targets)

        # x: [50000, 32,32,3] , numpy  ||  y: 50000 , numpy
        targets, data = init_transform(y.tolist(), x, keep_file=keep_file, training=training, tasks=tasks, task_idx=task_idx, buffer_lst=buffer_lst) 
        if ret_data: #used in _compute_class_mean() in base.py
            return data, targets, DummyDataset(data, targets, trsf, self.use_path, with_raw, with_noise)
        else:
            return DummyDataset(data, targets, trsf, self.use_path, with_raw, with_noise)

    def get_dataset_with_split(self, indices, source, mode, appendent=None, val_samples_per_class=0):
        if source == 'train':
            x, y = self._train_data, self._train_targets
        elif source == 'test':
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        if mode == 'train':
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == 'test':
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError('Unknown mode {}.'.format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx+1)
            val_indx = np.random.choice(len(class_data), val_samples_per_class, replace=False)
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets))+1):
                append_data, append_targets = self._select(appendent_data, appendent_targets,
                                                           low_range=idx, high_range=idx+1)
                val_indx = np.random.choice(len(append_data), val_samples_per_class, replace=False)
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(train_targets)
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(train_data, train_targets, trsf, self.use_path), \
            DummyDataset(val_data, val_targets, trsf, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed): # DataMananger() calls this function 
        idata = _get_idata(dataset_name)
        idata.download_data()

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path        #False for CIFAR100

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # Order
        # order = [i for i in range(len(np.unique(self._train_targets)))]
        # if shuffle:
        #     np.random.seed(seed)
        #     order = np.random.permutation(len(order)).tolist()
        # else:
        #     order = idata.class_order
        # self._class_order = order

        # Map indices
        # self._train_targets = _map_new_class_index(self._train_targets, self._class_order) 
        # self._test_targets = _map_new_class_index(self._test_targets, self._class_order)
        self._class_order = np.arange(100)   #100 due to CIFAR100
        logging.info(self._class_order)  # if seed 1993: [68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50, 28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96, 98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69, 36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33]


    def _select(self, x, y, low_range, high_range): #get_dataset() in DataManager() calls this function
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]


class DummyDataset(Dataset): #get_dataset() in DataManager() calls this class
    def __init__(self, images, labels, trsf, use_path=False, with_raw=False, with_noise=False):
        assert len(images) == len(labels), 'Data size error!'
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path
        self.with_raw = with_raw
        if use_path and with_raw:
            self.raw_trsf = transforms.Compose([transforms.Resize((500, 500)), transforms.ToTensor()])
        else:
            self.raw_trsf = transforms.Compose([transforms.ToTensor()])
        if with_noise:
            class_list = np.unique(self.labels)
            self.ori_labels = deepcopy(labels)
            for cls in class_list:
                random_target = class_list.tolist()
                random_target.remove(cls)
                tindx = [i for i, x in enumerate(self.ori_labels) if x == cls]
                for i in tindx[:round(len(tindx)*0.2)]:
                    self.labels[i] = random.choice(random_target)

        self.target_indices = []
        for t in range(100):  #100 due to CIFAR100
            indices = np.squeeze(np.argwhere(self.labels == t)).tolist()
            if isinstance(indices, int):
                indices = [indices]
            self.target_indices.append(indices)
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            load_image = pil_loader(self.images[idx])
            image = self.trsf(load_image)
        else:
            load_image = Image.fromarray(self.images[idx])
            image = self.trsf(load_image)
        label = self.labels[idx]
        if self.with_raw:
            return idx, image, label, self.raw_trsf(load_image) 
        return idx, image, label


def _map_new_class_index(y, order): # _setup_data() calls this function
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name): # _setup_data() in DataManager() calls this function
    name = dataset_name.lower()
    if name == 'cifar10':
        return iCIFAR10()
    elif name == 'cifar100':
        return iCIFAR100()
    elif name == 'cifar100_224':
        return iCIFAR100_224()
    elif name == 'imagenet1000':
        return iImageNet1000()
    elif name == "imagenet100":
        return iImageNet100()
    elif name == "imagenet-r":
        return iImageNetR()
    elif name == 'cub200_224':
        return iCUB200_224()
    elif name == 'resisc45':
        return iResisc45_224()
    elif name == 'cars196_224':
        return iCARS196_224()
    elif name == 'sketch345_224':
        return iSketch345_224()
    else:
        raise NotImplementedError('Unknown dataset {}.'.format(dataset_name))


def pil_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    '''
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    '''
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    '''
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
