import copy
import torch
from torch import nn
from convs.cifar_resnet import resnet32
from convs.resnet import resnet18, resnet34, resnet50
from convs.linears import SimpleContinualLinear
from convs.vits import vit_base_patch16_224_in21k, vit_base_patch16_224_mocov3
import torch.nn.functional as F

def get_convnet(convnet_type, pretrained=False): # BaseNet in inc_net.py calls this function
    name = convnet_type.lower()
    if name == 'resnet32':
        return resnet32()
    elif name == 'resnet18':
        return resnet18(pretrained=pretrained)
    elif name == 'resnet18_cifar':
        return resnet18(pretrained=pretrained, cifar=True)
    elif name == 'resnet18_cifar_cos':
        return resnet18(pretrained=pretrained, cifar=True, no_last_relu=True)
    elif name == 'resnet34':
        return resnet34(pretrained=pretrained)
    elif name == 'resnet50':
        return resnet50(pretrained=pretrained)
    elif name == 'vit-b-p16':
        return vit_base_patch16_224_in21k(pretrained=pretrained)
    elif name == 'vit-b-p16-mocov3':
        return vit_base_patch16_224_mocov3(pretrained=True)
    else:
        raise NotImplementedError('Unknown type {}'.format(convnet_type))


class BaseNet(nn.Module): # FinetuneIncrementalNet in inc_net.py calls this CLASS

    def __init__(self, convnet_type, pretrained):
        super(BaseNet, self).__init__()

        self.convnet = get_convnet(convnet_type, pretrained)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x): #Used in  _extract_vectors() in baselearner in base.py
        return self.convnet(x)['features']

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x['features'])
        '''
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        '''
        out.update(x)

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

class FinetuneIncrementalNet(BaseNet): # SLCA in slca.py calls this CLASS

    def __init__(self, convnet_type, pretrained, fc_with_ln=False):
        super().__init__(convnet_type, pretrained)
        self.old_fc = None
        self.fc_with_ln = fc_with_ln


    def extract_layerwise_vector(self, x, pool=True):
        with torch.no_grad():
            features = self.convnet(x, layer_feat=True)['features']
        for f_i in range(len(features)):
            if pool:
                features[f_i] = features[f_i].mean(1).cpu().numpy() 
            else:
                features[f_i] = features[f_i][:, 0].cpu().numpy() 
        return features


    def update_fc(self, nb_classes, freeze_old=True): # incremental_train() in class SLCA in slca.py calls this function
        if self.fc is None: # self.fc is None for the first task
            self.fc = self.generate_fc(self.feature_dim, nb_classes)
            # print(f'self.fc: {self.fc}')
        else:
            self.fc.update(nb_classes, freeze_old=freeze_old) # Runs for second task onwards
            # print(f'self.fc: {self.fc}')

    def save_old_fc(self):
        if self.old_fc is None:
            self.old_fc = copy.deepcopy(self.fc)
        else:
            self.old_fc.heads.append(copy.deepcopy(self.fc.heads[-1]))

    def generate_fc(self, in_dim, out_dim): #update_fc() in this class calls this function for the first task
        fc = SimpleContinualLinear(in_dim, out_dim)

        return fc

    def forward(self, x, bcb_no_grad=False, fc_only=False):
        if fc_only:
            fc_out = self.fc(x)
            if self.old_fc is not None:
                old_fc_logits = self.old_fc(x)['logits']
                fc_out['old_logits'] = old_fc_logits
            return fc_out
        if bcb_no_grad:
            with torch.no_grad():
                x = self.convnet(x)
        else:
            x = self.convnet(x)
        out = self.fc(x['features'])
        out.update(x)

        return out


