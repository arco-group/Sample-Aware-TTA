from models.UNet import UNet
import torch.nn as nn
import numpy as np
import copy
import torch
import os
import torch.nn.functional as F
from models import networks

def init_weights(x):
    # torch.manual_seed(0)
    if type(x) == nn.Conv2d:
        nn.init.kaiming_normal_(x.weight.data)
        nn.init.zeros_(x.bias.data)

"""
def init_weights_eye(x, channel=64):
    # indentity init, only works for same input/output channel
    if type(x) == nn.Conv2d:
        eye = nn.init.eye_(torch.empty(x.weight.shape[0], x.weight.shape[1])).unsqueeze(-1).unsqueeze(-1)
        init_bias = nn.init.zeros_(torch.empty(x.weight.shape[0]))
        x.weight.data = eye
        x.bias.data = init_bias
"""

def init_weights_eye(m):
    if isinstance(m, nn.Conv2d):
        device = m.weight.device
        with torch.no_grad():
            eye = torch.eye(m.in_channels, device=device).view(m.in_channels, m.in_channels, 1, 1)
            m.weight.copy_(eye)
            m.bias.zero_()
            # print(f"[init_eye] Conv({m.in_channels}x{m.out_channels}) set to identity on {device}")

class DTTAnorm(nn.Module):  # Karani-style method
    def __init__(self, opt):
        super(DTTAnorm, self).__init__()
        self.opt = opt
        self.usedtta = opt.usedtta

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self,x):
        x_ = self.conv1(x)
        scale = (torch.randn([1,16,1,1]) * 0.05 + 0.2).to(x_.device)
        x_ = torch.exp(-(x_**2) / (scale**2))
        x_ = self.conv2(x_)
        x_ = torch.exp(-(x_**2) / (scale**2))
        x_ = self.conv3(x_)
        return x_ + x


def init_weights_zero(m):
    if isinstance(m, nn.Conv2d):
        nn.init.zeros_(m.weight)
        nn.init.zeros_(m.bias)

class AdpNetIrene(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(64),
            nn.Conv2d(64, 64, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(64),
            nn.Conv2d(64, 1, 1),
            nn.InstanceNorm2d(1)
        )
        self.net.apply(init_weights_zero)

    def forward(self, x):
        return x + self.net(x)


class ANet(nn.Module):
    def __init__(self, opt):
        """Adaptor Net: Default is for a 4-level UNet with fixed 64 channels
        Args:
            AENet: nn.Module, pre-trained auto-encoder for the source image
            channel: int, input feature channel of the affine transform
            seq: list->int, index of the affine matrix to be used
        """
        super(ANet, self).__init__()
        self.opt = opt
        tnet_dim = [1, 64, 128, 256, 256, 256, 256, 256, 1]
        self.ALoss=nn.MSELoss()
        self.save_dir = os.path.join(opt.results_dir, 'weights')
        self.save_dir_config = os.path.join(opt.results_dir, 'config_weights')

        adpNet = None
        if self.opt.usedtta:
            adpNet = DTTAnorm(opt)
        seq = self.opt.seq

        self.conv = nn.ModuleList()
        feature_channel = tnet_dim[1:-1]
        self.channel = feature_channel + feature_channel[::-1]
        nums = len(self.channel)
        self.nums = nums
        if seq is None:
            self.seq = np.arange(nums)
        else:
            self.seq = seq
        # Use pre-contrast manipulation
        self.adpNet = adpNet
        if adpNet is None:
            """
            self.adpNet = nn.Sequential(
                nn.Conv2d(1, 64, 1),
                nn.LeakyReLU(negative_slope=0.2),
                nn.InstanceNorm2d(64),
                nn.Conv2d(64, 64, 1),
                nn.LeakyReLU(negative_slope=0.2),
                nn.InstanceNorm2d(64),
                nn.Conv2d(64, 1, 1),
                nn.LeakyReLU(negative_slope=0.2),
                nn.InstanceNorm2d(1)
            )
            self.adpNet.apply(init_weights)
            """
            self.adpNet = AdpNetIrene()
            # Use feature affine transform
        for c in self.channel:
            convs = nn.Conv2d(c, c, 1)
            self.conv.append(convs)
        self.conv.apply(init_weights_eye)

        self.optimizer_ANet = torch.optim.Adam(self.parameters(), opt.alr)


    def reset(self, default=True):
        # Reset the fine-tuned weights for a new test subject
        # np.random.seed(0)
        #torch.manual_seed(0)
        if default:
            self.conv.apply(init_weights_eye)
            # self.adpNet.apply(init_weights)
            self.adpNet.apply(init_weights_zero)
        else:
            path = os.path.join(self.save_dir, 'Reset')
            # Load adpNet
            weight_path = os.path.join(path, 'adpNet.pth')
            if os.path.exists(weight_path):
                state_dict = torch.load(weight_path)
                self.adpNet.load_state_dict(state_dict)
            else:
                print(f"File {weight_path} not found")
            # Load the conv layers
            for i in range(len(self.conv)):
                weight_path = os.path.join(path, f'conv_{i}.pth')
                if os.path.exists(weight_path):
                    state_dict = torch.load(weight_path)
                    self.conv[i].load_state_dict(state_dict)
                else:
                    print(f"File {weight_path} not found")
                    break
        self.cuda()

    def forward(self, batch, task_model, generator='pix2pix'):
        """
        Forward for a 4-level UNet
        Args:
            TNet: nn.Module. The pretrained task network
            side_out: bool. If true, output every intermediate results
            seq: list->int or np array. Position of 1x1 convolution
        """
        # Send the input image through the adaptor first.
        real_A = batch['A'].to('cuda')
        x = self.adpNet(real_A)
        task_model.set_input(batch)

        outputs = {}
        outputs['input']= x
        # Step 1: Reflection padding
        if generator == 'pix2pix':
            tm = task_model.netG
        elif generator == 'cycle_gan':
            tm = task_model.netG_A
        elif generator == 'cycle_gan_paired':
            tm = task_model.netG_A
        x = tm.module.model[0](x)

        # Step 2: First convolution
        first_conv_input = tm.module.model[1](x)
        if 'first_conv' in self.opt.return_layers:
            first_conv_input=self.conv[0](first_conv_input)
            outputs['first_conv'] = first_conv_input

        # Normalization + ReLU
        x = tm.module.model[2](first_conv_input)
        x=tm.module.model[3](x)
        second_conv_input = tm.module.model[4](x)
        if 'second_conv' in self.opt.return_layers:
            second_conv_input = self.conv[1](second_conv_input)
            outputs['second_conv'] = second_conv_input
        x = tm.module.model[5:7](second_conv_input)

        third_conv_input = tm.module.model[7](x)
        if 'third_conv' in self.opt.return_layers:
            third_conv_input = self.conv[2](third_conv_input)
            outputs['third_conv'] = third_conv_input
        x = tm.module.model[8](third_conv_input)
        x = tm.module.model[9](x)

        # Step 3: ResNet blocks
        resnet_block_1_feature = tm.module.model[10](x)  # First ResNet block
        if 'resnet_block_1' in self.opt.return_layers:
            resnet_block_1_feature = self.conv[3](resnet_block_1_feature)
            outputs['resnet_block_1'] = resnet_block_1_feature

        resnet_block_2_feature = tm.module.model[11](resnet_block_1_feature)  # Second ResNet block
        if 'resnet_block_2' in self.opt.return_layers:
            resnet_block_2_feature = self.conv[4](resnet_block_2_feature)
            outputs['resnet_block_2'] = resnet_block_2_feature

        resnet_block_3_feature = tm.module.model[12](resnet_block_2_feature)  # Third ResNet block output
        if 'resnet_block_3' in self.opt.return_layers:
            resnet_block_3_feature = self.conv[5](resnet_block_3_feature)
            outputs['resnet_block_3'] = resnet_block_3_feature

        resnet_block_4_feature = tm.module.model[13](resnet_block_3_feature)  # Fourth ResNet block
        if 'resnet_block_4' in self.opt.return_layers:
            resnet_block_4_feature = self.conv[6](resnet_block_4_feature)
            outputs['resnet_block_4'] = resnet_block_4_feature

        x = tm.module.model[14](resnet_block_4_feature)  # Fifth ResNet block
        if 'resnet_block_4' in self.opt.return_layers:
            x = self.conv[7](x)
            outputs['resnet_block_4'] = (resnet_block_4_feature, x)

        x = tm.module.model[15](x)  # Sixth ResNet block
        if 'resnet_block_3' in self.opt.return_layers:
            x = self.conv[8](x)
            outputs['resnet_block_3' ] = (resnet_block_3_feature, x)

        x = tm.module.model[16](x)  # Seventh ResNet block
        if 'resnet_block_2' in self.opt.return_layers:
            x = self.conv[9](x)
            outputs['resnet_block_2'] = (resnet_block_2_feature, x)

        x = tm.module.model[17](x)  # Eighth ResNet block
        if 'resnet_block_1' in self.opt.return_layers:
            x = self.conv[10](x)
            outputs['resnet_block_1'] = (resnet_block_1_feature, x)

        x = tm.module.model[18](x)  # Ninth ResNet block
        if 'third_conv' in self.opt.return_layers:
            x = self.conv[11](x)
            outputs['third_conv'] = (third_conv_input, x)

        x = tm.module.model[19](x)

        x = tm.module.model[20](x)
        if 'second_conv' in self.opt.return_layers:
            x = self.conv[12](x)
            outputs['second_conv'] = (second_conv_input, x)

        x = tm.module.model[21:24](x)

        x = tm.module.model[24](x)
        if 'first_conv' in self.opt.return_layers:
            x = self.conv[13](x)
            outputs['first_conv'] = (first_conv_input, x)

        x = tm.module.model[25](x)
        x = tm.module.model[26](x)
        final_output = tm.module.model[27](x)
        outputs['final_output'] = final_output
        return outputs

    # Pass the real input through T, collect the outputs, and adapt each block separately.
    # final_output is not adapted because it is already the task model output.

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

    def save_networks(self, epoch, config=False):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        if not config:
            path = os.path.join(self.save_dir, epoch)
        else:
            path = os.path.join(self.save_dir_config, epoch)
        if not os.path.exists(path):
            os.makedirs(path)
        weight_path = os.path.join(path, 'adpNet.pth')
        if len(self.opt.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.adpNet.cpu().state_dict(), weight_path)
            self.adpNet.cuda()
        else:
            torch.save(self.adpNet.cpu().state_dict(), weight_path)
        for i in range(len(self.conv)):
            weight_path = os.path.join(path, f'conv_{i}.pth')
            if len(self.opt.gpu_ids) > 0 and torch.cuda.is_available():
                torch.save(self.conv[i].cpu().state_dict(), weight_path)
                self.conv[i].cuda()
            else:
                torch.save(self.conv[i].cpu().state_dict(), weight_path)

    def load_networks(self, epoch, config=False):
        if not config:
            path = os.path.join(self.save_dir, epoch)
        else:
            path = os.path.join(self.save_dir_config, epoch)
        weight_path = os.path.join(path, 'adpNet.pth')
        if os.path.exists(weight_path):
            state_dict = torch.load(weight_path)
            self.adpNet.load_state_dict(state_dict)
        else:
            print(f"File {weight_path} not found")
        for i in range(len(self.conv)):
            weight_path = os.path.join(path, f'conv_{i}.pth')
            if os.path.exists(weight_path):
                state_dict = torch.load(weight_path)
                self.conv[i].load_state_dict(state_dict)
            else:
                print(f"File {weight_path} not found")
                break
