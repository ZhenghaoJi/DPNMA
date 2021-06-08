import torch
from torch import nn
from torch.utils import model_zoo
import torch.nn.functional as F

class WeightAverage(nn.Module):
    def __init__(self, c_in, R=3):
        super(WeightAverage, self).__init__()
        c_out = c_in // 2

        self.conv_theta = nn.Conv2d(c_in, c_out, 1)
        self.conv_phi = nn.Conv2d(c_in, c_out, 1)
        self.conv_g = nn.Conv2d(c_in, c_out, 1)
        self.conv_back = nn.Conv2d(c_out, c_in, 1)
        self.CosSimLayer = nn.CosineSimilarity(dim=3) #norm
        #self.relu=nn.ReLU(inplace=True)
        self.R = R
        self.c_out = c_out

    def forward(self, x):
        """
        x: torch.Tensor(batch_size, channel, h, w)
        """

        batch_size, c, h, w = x.size()
        padded_x = F.pad(x, (1,1,1,1), 'replicate')
        neighbor = F.unfold(padded_x, kernel_size=self.R, dilation=1, stride=1) # BS, C*R*R, H*W
        neighbor = neighbor.contiguous().view(batch_size, c, self.R, self.R, h, w)
        neighbor = neighbor.permute(0, 2, 3, 1, 4, 5) # BS, R, R, c, h ,w
        neighbor = neighbor.reshape(batch_size * self.R * self.R, c, h, w)
        

        theta = self.conv_theta(x) # BS, C', h, w
        phi = self.conv_phi(neighbor)   # BS*R*R, C', h, w
        g = self.conv_g(neighbor)     # BS*R*R, C', h, w

        phi = phi.contiguous().view(batch_size, self.R, self.R, self.c_out, h, w)
        phi = phi.permute(0, 4, 5, 3, 1, 2) # BS, h, w, c, R, R
        theta = theta.permute(0, 2, 3, 1).contiguous().view(batch_size, h, w, self.c_out)   # BS, h, w, c
        theta_dim = theta
        
        cos_sim = self.CosSimLayer(phi, theta_dim[:,:,:,:,None,None]) # BS, h, w, R, R

        softmax_sim = F.softmax(cos_sim.contiguous().view(batch_size, h, w, -1), dim=3).contiguous().view_as(cos_sim) # BS, h, w, R, R
      
        g = g.contiguous().view(batch_size, self.R, self.R, self.c_out, h, w)
        g = g.permute(0, 4, 5, 1, 2, 3) # BS, h, w, R, R, c_out

        weighted_g = g * softmax_sim[:,:,:,:,:,None]
        weighted_average = torch.sum(weighted_g.contiguous().view(batch_size, h, w, -1, self.c_out), dim=3)
        weight_average = weighted_average.permute(0,3,1,2).contiguous() # BS, c_out, h, w

        x_res = self.conv_back(weight_average)
        ret = x + x_res

        #return self.relu(ret)
        return ret
class NonLocal(nn.Module):
    def __init__(self):
        super(NonLocal, self).__init__()
        self.non_local_1 = WeightAverage(128)
        self.non_local_2 = WeightAverage(256)
        self.non_local_3 = WeightAverage(512)
        self.non_local_4 = WeightAverage(512)

    def forward(self, input):
        conv2_2, conv3_3, conv4_3, conv5_3 = input
        conv2_2 = self.non_local_1(conv2_2)
        conv3_3 = self.non_local_2(conv3_3)
        conv4_3 = self.non_local_3(conv4_3)
        conv5_3 = self.non_local_4(conv5_3)
        return conv2_2, conv3_3, conv4_3, conv5_3    
            
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.vgg = VGG()
        self.load_vgg()
        self.non_local = NonLocal()
        
        self.amp = BackEnd()
        self.dmp = BackEnd()
        self.conv_att = BaseConv(32, 1, 1, 1, activation=nn.Sigmoid(), use_bn=True)
        self.conv_out = BaseConv(32, 1, 1, 1, activation=None, use_bn=False)

    def forward(self, input):
        input = self.vgg(input)
        input = self.non_local(input)
        amp_out = self.amp(*input)
        dmp_out = self.dmp(*input)        
        amp_out = self.conv_att(amp_out)
        dmp_out = amp_out * dmp_out
        dmp_out = self.conv_out(dmp_out)

        return dmp_out, amp_out

    def load_vgg(self):
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth')
        old_name = [0, 1, 3, 4, 7, 8, 10, 11, 14, 15, 17, 18, 20, 21, 24, 25, 27, 28, 30, 31, 34, 35, 37, 38, 40, 41]
        new_name = ['1_1', '1_2', '2_1', '2_2', '3_1', '3_2', '3_3', '4_1', '4_2', '4_3', '5_1', '5_2', '5_3']
        new_dict = {}
        for i in range(13):
            new_dict['conv' + new_name[i] + '.conv.weight'] = \
                state_dict['features.' + str(old_name[2 * i]) + '.weight']
            new_dict['conv' + new_name[i] + '.conv.bias'] = \
                state_dict['features.' + str(old_name[2 * i]) + '.bias']
            new_dict['conv' + new_name[i] + '.bn.weight'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.weight']
            new_dict['conv' + new_name[i] + '.bn.bias'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.bias']
            new_dict['conv' + new_name[i] + '.bn.running_mean'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.running_mean']
            new_dict['conv' + new_name[i] + '.bn.running_var'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.running_var']

        self.vgg.load_state_dict(new_dict)


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_1 = BaseConv(3, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv1_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_1 = BaseConv(64, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_1 = BaseConv(128, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_3 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_1 = BaseConv(256, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, input):
        input = self.conv1_1(input)
        input = self.conv1_2(input)
        input = self.pool(input)
        input = self.conv2_1(input)
        conv2_2 = self.conv2_2(input)

        input = self.pool(conv2_2)
        input = self.conv3_1(input)
        input = self.conv3_2(input)
        conv3_3 = self.conv3_3(input)

        input = self.pool(conv3_3)
        input = self.conv4_1(input)
        input = self.conv4_2(input)
        conv4_3 = self.conv4_3(input)

        input = self.pool(conv4_3)
        input = self.conv5_1(input)
        input = self.conv5_2(input)
        conv5_3 = self.conv5_3(input)

        return conv2_2, conv3_3, conv4_3, conv5_3




class BackEnd(nn.Module):
    def __init__(self):
        super(BackEnd, self).__init__()

        self.conv5 = BaseConv(256, 64, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv6 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv7 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, *input):
        conv2_2, conv3_3, conv4_3, conv5_3 = input

        input = torch.cat([conv2_2, conv2_2], 1)
        input = self.conv5(input)
        input = self.conv6(input)
        input = self.conv7(input)

        return input


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, activation=None, use_bn=False):
        super(BaseConv, self).__init__()
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2)
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.zero_()
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)

        return input


if __name__ == '__main__':
    input = torch.randn(8, 3, 400, 400).cuda()
    model = Model().cuda()
    output, attention = model(input)
    print(input.size())
    print(output.size())
    print(attention.size())
