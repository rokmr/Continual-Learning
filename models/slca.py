import logging
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import FinetuneIncrementalNet
from torchvision import transforms
from torch.distributions.multivariate_normal import MultivariateNormal
import random
from utils.toolkit import tensor2numpy, accuracy
import copy
import os
from utils.nncsl_functions import make_buffer_lst, ClassStratifiedSampler

epochs = 20
lrate = 0.01 
milestones = [60,100,140]
lrate_decay = 0.1
batch_size = 128
split_ratio = 0.1
T = 2
weight_decay = 5e-4
num_workers = 8
ca_epochs = 5


class SLCA(BaseLearner): # get_model() in factory.py calls this CLASS
    def __init__(self, args):
        super().__init__(args)
        self._network = FinetuneIncrementalNet(args['convnet_type'], pretrained=True) # Here only we got self._network.convnet
        self.log_path = "logs/{}_{}".format(args['model_name'], args['model_postfix'])
        self.model_prefix = args['prefix']
        if 'epochs' in args.keys():
            global epochs
            epochs = args['epochs'] 
        if 'milestones' in args.keys():
            global milestones
            milestones = args['milestones']
        if 'lr' in args.keys():
            global lrate
            lrate = args['lr']
            print('set lr to ', lrate)
        if 'bcb_lrscale' in args.keys():
            self.bcb_lrscale = args['bcb_lrscale']
        else:
            self.bcb_lrscale = 1.0/100
        if self.bcb_lrscale == 0:
            self.fix_bcb = True
        else:
            self.fix_bcb = False
        print('fic_bcb', self.fix_bcb) # False


        
        if 'save_before_ca' in args.keys() and args['save_before_ca']:
            self.save_before_ca = True
        else:
            self.save_before_ca = False

        if 'ca_epochs' in args.keys():
            global ca_epochs
            ca_epochs = args['ca_epochs'] 

        if 'ca_with_logit_norm' in args.keys() and args['ca_with_logit_norm']>0:
            self.logit_norm = args['ca_with_logit_norm']
        else:
            self.logit_norm = None

        self.run_id = args['run_id']
        self.seed = args['seed']
        self.task_sizes = []
        self.buffer_lst=None
        self.buffer_size= args['buffer_size']
        self.subset_path= args['subset_path']
        self.subset_path_cls= args['subset_path_cls']
        self.s_batch_size = args['s_batch_size']
        self.g = torch.Generator()
        self.g.manual_seed(0)
        self._GLOBAL_SEED = 0

    def after_task(self): #Called by _train() in trainer.py
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))
        self.save_checkpoint(self.log_path+'/'+self.model_prefix+'_seed{}'.format(self.seed), head_only=self.fix_bcb)
        self._network.fc.recall() #load old_state_dict

    def incremental_train(self, data_manager): # _train() in trainer.py calls this function
        self._cur_task += 1 # initialized with -1 -> 0 -> 1 -> 2 
        task_size = data_manager.get_task_size(self._cur_task) # 10 -> 10 -> 10
        self.task_sizes.append(task_size) # [10] -> [10, 10] -> [10, 10, 10]
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task) # 10 -> 20 -> 30
        self.topk = self._total_classes if self._total_classes<5 else 5
        self._network.update_fc(data_manager.get_task_size(self._cur_task)) # calls update_fc() in inc_net.py in class FinetuneIncrementalNet
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        self._network.to(self._device)
        self.tasks = list(range(0, (self._cur_task+1)*10))


        train_dset = data_manager.get_dataset(np.arange(0, self._total_classes), source='train', mode='train', tasks =self.tasks, task_idx=self._cur_task, buffer_lst = self.buffer_lst, with_raw=False, keep_file = self.subset_path) #only for the current task
        test_dset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test', tasks =self.tasks, task_idx=self._cur_task, buffer_lst = self.buffer_lst, keep_file = self.subset_path)  # All previous classes including current classes
        dset_name = data_manager.dataset_name.lower()

        self.test_sampler = torch.utils.data.distributed.DistributedSampler(dataset=test_dset, num_replicas=1, rank=0)
        self.train_sampler = ClassStratifiedSampler(data_source=train_dset, world_size=1, rank= 0, batch_size = self.s_batch_size, classes_per_batch = self._total_classes, seed= self._GLOBAL_SEED)
        self.train_loader = DataLoader(train_dset, shuffle=True, batch_size=batch_size, num_workers=num_workers, worker_init_fn=self.seed_worker, generator = self.g)
        self.test_loader = DataLoader(test_dset, sampler=self.test_sampler, batch_size=batch_size, drop_last= True, shuffle=False, num_workers=num_workers, worker_init_fn=self.seed_worker, generator = self.g)
        logging.info(f"TASK ID: {self._cur_task}  || len(train_loader): {len(self.train_loader.dataset)}  ||  len(test_loader): {len(self.test_loader.dataset)}")
        
        # for idx, batch in enumerate(self.train_loader):
        #     logging.info(f"TRAIN Batch INFO => num_datapoint: {len(batch[2])} || img_shape: {batch[1].shape} ||targets_in_batch: {batch[2]} ")
        #     if idx == 0:
        #         break
        
        # for idx, batch in enumerate(self.test_loader):
        #     logging.info(f"TEST Batch INFO => num_datapoint: {len(batch[2])} || img_shape: {batch[1].shape} ||targets_in_batch: {batch[2]} ")
        #     if idx == 0:
        #         break

        # logging.info(f"===================================================================================")
        self._stage1_training(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module      

        # CA
        self._network.fc.backup() # creates deep copy : function in linear.py
        if self.save_before_ca:
            self.save_checkpoint(self.log_path+'/'+self.model_prefix+'_seed{}_before_ca'.format(self.seed), head_only=self.fix_bcb) # function in base.py
        
        self._compute_class_mean(data_manager, check_diff=False, oracle=False) #In base.py
        if self._cur_task>0 and ca_epochs>0: # Runs from second task onwards
            self._stage2_compact_classifier(task_size)
            if len(self._multiple_gpus) > 1:
                self._network = self._network.module

        classes = list(range(100))
        self.tasks_buffer = [classes[i:i + task_size] for i in range(0, len(classes), task_size)]
        self.buffer_lst = make_buffer_lst(self.buffer_lst, self.buffer_size, self.subset_path, self.subset_path_cls, tasks=self.tasks_buffer, task_idx = self._cur_task)

    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)



    def _run(self, train_loader, test_loader, optimizer, scheduler): # _stage1_training() in this class calls this function
        run_epochs = epochs #2
        # print(f"run_epochs: {run_epochs}")
        for epoch in range(1, run_epochs+1):
            self._network.train()
            losses = 0.
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                # print(f"inputs.shape: {inputs.shape}, targets.shape: {targets.shape}\n") #[128, 3, 224, 224], [128] -> [128, 3, 224, 224], [128] -> [128, 3, 224, 224], [128]
                logits = self._network(inputs, bcb_no_grad=self.fix_bcb)['logits'] # [128, 10] -> [128, 20] -> [128, 30]
                cur_targets = torch.where(targets-self._known_classes>=0,targets-self._known_classes,-100) #[128] -> [128] -> [128]
                loss = F.cross_entropy(logits[:, self._known_classes:], cur_targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            if epoch%5==0:
                train_acc = self._compute_accuracy(self._network, train_loader)
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.3f}, Test_accy {:.3f}'.format(
                    self._cur_task, epoch, epochs, losses/len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}'.format(
                    self._cur_task, epoch, epochs, losses/len(train_loader))
            logging.info(info)

    def _stage1_training(self, train_loader, test_loader): # incremental_train() in this class calls this function
        '''
        if self._cur_task == 0:
            loaded_dict = torch.load('./dict_0.pkl')
            self._network.load_state_dict(loaded_dict['model_state_dict'])
            self._network.to(self._device)
            return
        '''
        base_params = self._network.convnet.parameters()  #convnet is the backbone of the model : vit-b-p16
        base_fc_params = [p for p in self._network.fc.parameters() if p.requires_grad==True]
        head_scale = 1. if 'moco' in self.log_path else 1. #Always 1
        if not self.fix_bcb:
            base_params = {'params': base_params, 'lr': lrate*self.bcb_lrscale, 'weight_decay': weight_decay}
            base_fc_params = {'params': base_fc_params, 'lr': lrate*head_scale, 'weight_decay': weight_decay}
            network_params = [base_params, base_fc_params]
        else:
            for p in base_params:
                p.requires_grad = False
            network_params = [{'params': base_fc_params, 'lr': lrate*head_scale, 'weight_decay': weight_decay}]
        optimizer = optim.SGD(network_params, lr=lrate, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._run(train_loader, test_loader, optimizer, scheduler)


    def _stage2_compact_classifier(self, task_size): # Called after first task # task_size = 10
        for p in self._network.fc.parameters():
            p.requires_grad=True
            
        run_epochs = ca_epochs            #5
        crct_num = self._total_classes    #20 
        param_list = [p for p in self._network.fc.parameters() if p.requires_grad]
        network_params = [{'params': param_list, 'lr': lrate,
                           'weight_decay': weight_decay}]
        optimizer = optim.SGD(network_params, lr=lrate, momentum=0.9, weight_decay=weight_decay)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[4], gamma=lrate_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)

        self._network.to(self._device)
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._network.eval() # Only dropout and batchnorm are affected
        for epoch in range(run_epochs): #5
            losses = 0.

            sampled_data = []
            sampled_label = []
            num_sampled_pcls = 256
        
            for c_id in range(crct_num): #20
                t_id = c_id//task_size
                decay = (t_id+1)/(self._cur_task+1)*0.1 #0.05
                cls_mean = torch.tensor(self._class_means[c_id], dtype=torch.float64).to(self._device)*(0.9+decay) # torch.from_numpy(self._class_means[c_id]).to(self._device)
                cls_cov = self._class_covs[c_id].to(self._device)
                
                m = MultivariateNormal(cls_mean.float(), cls_cov.float())

                sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,)) #[256, 768]
                sampled_data.append(sampled_data_single)                
                sampled_label.extend([c_id]*num_sampled_pcls)

            sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device) #[5120, 768]
            sampled_label = torch.tensor(sampled_label).long().to(self._device) #[5120]

            inputs = sampled_data
            targets= sampled_label

            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            
            for _iter in range(crct_num): #20
                inp = inputs[_iter*num_sampled_pcls:(_iter+1)*num_sampled_pcls]
                tgt = targets[_iter*num_sampled_pcls:(_iter+1)*num_sampled_pcls] 
                outputs = self._network(inp, bcb_no_grad=True, fc_only=True)
                logits = outputs['logits'] #[256, 20]

                if self.logit_norm is not None: #0.1
                    per_task_norm = []
                    prev_t_size = 0
                    cur_t_size = 0
                    for _ti in range(self._cur_task+1): #1+1 ->
                        cur_t_size += self.task_sizes[_ti]
                        temp_norm = torch.norm(logits[:, prev_t_size:cur_t_size], p=2, dim=-1, keepdim=True) + 1e-7 #[256, 1] : Calculate norm for each task
                        per_task_norm.append(temp_norm)
                        prev_t_size += self.task_sizes[_ti]
                    per_task_norm = torch.cat(per_task_norm, dim=-1) #[256, 2] ->
                    norms = per_task_norm.mean(dim=-1, keepdim=True) #[256, 1] -> Calculate mean of norms for per_task_norm along task dimension
                        
                    norms_all = torch.norm(logits[:, :crct_num], p=2, dim=-1, keepdim=True) + 1e-7 #Calculate norm for all classes ; [256, 1]
                    decoupled_logits = torch.div(logits[:, :crct_num], norms) / self.logit_norm #[256, 20]
                    loss = F.cross_entropy(decoupled_logits, tgt)

                else:
                    loss = F.cross_entropy(logits[:, :crct_num], tgt)

                optimizer.zero_grad()
                loss.backward() #Backpropagation for fc layers only
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            test_acc = self._compute_accuracy(self._network, self.test_loader)
            info = 'CA Task {} => Loss {:.3f}, Test_accy {:.3f}'.format(
                self._cur_task, losses/self._total_classes, test_acc)
            logging.info(info)


