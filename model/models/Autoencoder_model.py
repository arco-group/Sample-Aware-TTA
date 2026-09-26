from models.UNet import UNet
import torch.nn as nn
import numpy as np
import copy
import torch
import os
import torch.nn.functional as F
from models import networks


# AE training:
# 1. define the AE and the task network T
# 2. load the pretrained task network weights and freeze it
# 3. pass each batch through T to obtain features, then feed them to the AE
# 4. compute the loss between T features and AE features
# Note: the AE script performs only the forward pass, so the AE output is produced from T features.

class AENet(nn.Module):
    def __init__(self, opt):
        super(AENet, self).__init__()
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        use_cuda = bool(opt.gpu_ids) and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.def_AENet()
        self.AELoss = nn.MSELoss()
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.metric = 0

        # Define optimizers for each subnet and store them in a list
        self.optimizers = []
        # Set AENet optimizers
        for subnets in self.AENet:
            params = []
            subnets.to(self.device)
            params.extend(list(subnets.parameters()))
            if opt.phase != 'test':
                self.optimizer_AENet = torch.optim.Adam(params, self.opt.aelr)
                self.optimizers.append(self.optimizer_AENet)
        if opt.phase != 'test':
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]


    def def_AENet(self):
        self.AENetMatch = [[0]]
        for i in range(1, len(self.opt.tnet_dim) - 1):
            self.AENetMatch += [[i, -i - 1]]
        self.AENetMatch += [[-1]]
        self.AENet = []
        n0 = self.opt.aenet_dim
        for i in range(len(self.AENetMatch)):
            if len(self.AENetMatch[i]) == 1:
                dims = self.opt.tnet_dim[i]
                self.AENet += [UNet(inplane=dims, midplane=[n0 // 2, n0 // 4, n0 // 8], \
                                    outplane=dims, skip=False, isn=True)]
            else:
                dims = self.opt.tnet_dim[i] * 2
                self.AENet += [UNet(inplane=dims, midplane=[n0, n0 // 2, n0 // 4], \
                                    outplane=dims, skip=False, isn=True)]

    # This class only performs the forward pass here; training and loss computation happen elsewhere.

    def forward(self, side_out):
        """Reconstruct one attached task-model feature map."""
        side_out = side_out.to(self.device)
        return self.AENet[-1](side_out, side_out=False)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=False for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def addnoise(self, feat):
        """ Add noise to features for auto-encoders
        feats [batch, channel, H, W]
        """
        if self.opt.feat_noise == False:
            return feat
        blks = [16, int(np.ceil(feat.shape[3] / feat.shape[2])) * 16]
        ratio = 0.25
        radius = [feat.shape[2] // blks[0] + 1, feat.shape[3] // blks[1] + 1]
        nums = np.round(blks[0] * blks[1] * ratio * ratio)
        wrong_labels = copy.deepcopy(feat)
        for i in range(feat.shape[0]):
            for _ in range(np.random.randint(nums)):
                rx = np.random.randint(1, radius[0] + 1)
                ry = np.random.randint(1, radius[1] + 1)
                mcx = np.random.randint(rx + 1, feat.shape[2] - rx - 1)
                mcy = np.random.randint(ry + 1, feat.shape[3] - ry - 1)
                mcx_src = np.random.randint(rx + 1, feat.shape[2] - rx - 1)
                mcy_src = np.random.randint(ry + 1, feat.shape[3] - ry - 1)
                wrong_labels[i, :, mcx - rx:mcx + rx, mcy - ry:mcy + ry] = feat[i, :, mcx_src - rx:mcx_src + rx,
                                                                           mcy_src - ry:mcy_src + ry]
        return wrong_labels

    def save_networks_AE(self, epoch, AE_to_train=None):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        path = os.path.join(self.save_dir, f'epoch{epoch}')
        if not os.path.exists(path):
            os.makedirs(path)
        if AE_to_train is None:
            for i in range(len(self.AENet)):
                weight_path = os.path.join(path,f'AE_{self.opt.return_layers[i]}_{epoch}.pt')

                if len(self.opt.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(self.AENet[i].cpu().state_dict(), weight_path)
                    self.AENet[i].cuda(self.gpu_ids[0])
                else:
                    torch.save(self.AENet[i].cpu().state_dict(), weight_path)
        else:
            weight_path = os.path.join(path, f'AE_{self.opt.return_layers[AE_to_train]}_{epoch}.pt')
            if len(self.opt.gpu_ids) > 0 and torch.cuda.is_available():
                torch.save(self.AENet[AE_to_train].cpu().state_dict(), weight_path)
                self.AENet[AE_to_train].cuda(self.gpu_ids[0])
            else:
                torch.save(self.AENet[AE_to_train].cpu().state_dict(), weight_path)

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_aelr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        aelr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_aelr, aelr))

  
