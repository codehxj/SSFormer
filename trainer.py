from decimal import Decimal
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn import LeakyReLU, Conv2d, InstanceNorm2d
import imageio
import utility
import torch
import torch.nn.utils as utils
from tqdm import tqdm


class GanLosstrain(nn.Module):
    def __init__(self):
        super(GanLosstrain,self).__init__()
        self.loss = nn.MSELoss()
    def get_target_label(self,input,targel):
        return input.new_ones(input.size())*targel
    def forward(self,input):
        targels =1
        targel_label = self.get_target_label(input,targels)
        loss = self.loss(input ,targel_label)
        return loss
class GanLosstrain1(nn.Module):
    def __init__(self):
        super(GanLosstrain1,self).__init__()
        self.loss = nn.MSELoss()
    def get_target_label(self,input,targel):
        return input.new_ones(input.size())*targel
    def forward(self,input):
        targels =False
        targel_label = self.get_target_label(input,targels)
        loss = self.loss(input ,targel_label)
        return loss

class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            LeakyReLU(negative_slope=0.2, inplace=True),
            Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False),
            LeakyReLU(negative_slope=0.2, inplace=True),
            Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False),
            LeakyReLU(negative_slope=0.2, inplace=True),
            Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False),
            LeakyReLU(negative_slope=0.2, inplace=True),
            Conv2d(512, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))).cuda()
    def forward(self,x):
        x = self.body(x)
        return x



class ResBlock1(nn.Module):
    def __init__(self, nf, ksize, norm=nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()

        self.nf = nf
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, ksize, 1, ksize // 2),
            norm(nf), act(),
            nn.Conv2d(nf, nf, ksize, 1, ksize // 2)
        )
    def forward(self, x):
        return torch.add(x, self.body(x))


class Quantization(nn.Module):
    def __init__(self, n=5):
        super().__init__()
        self.n = n
    def forward(self, inp):
        out = inp * 255.0
        flag = -1
        for i in range(1, self.n + 1):
            out = out + flag / np.pi / i * torch.sin(2 * i * np.pi * inp * 255.0)
            flag = flag * (-1)
        return out / 255.0


class KernelModel(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        nc, nf, nb = 64,16,64
        ksize = 3
        deg_kernel = [
            nn.Conv2d(64, 16, 1, 1, 1 // 2).cuda(),
            nn.BatchNorm2d(16).cuda(), nn.ReLU(True).cuda(),
            *[
                ResBlock1(nf=16, ksize=3).cuda()
                for _ in range(nb)
            ],
            nn.Conv2d(16, 3 ** 2, 1, 1, 0).cuda(),
            nn.Softmax(1).cuda()
        ]
        self.deg_kernel = nn.Sequential(*deg_kernel)
        nn.init.constant_(self.deg_kernel[-2].weight, 0)
        nn.init.constant_(self.deg_kernel[-2].bias, 0)
        self.deg_kernel[-2].bias.data[ksize ** 2 // 2] = 1
        self.pad = nn.ReflectionPad2d(ksize // 2)

    def forward(self, x):
        B, C, H, W = x.shape
        h = H // self.scale
        w = W // self.scale
        zk = torch.randn(B, 64, 1, 1).to(x.device)
        inp = zk
        ksize = 3
        kernel = self.deg_kernel(inp).view(B, 1, ksize ** 2, *inp.shape[2:]).cuda()
        x = x.view(B * C, 1, H, W)
        x = F.unfold(
            self.pad(x), kernel_size=ksize, stride=self.scale, padding=0
        ).view(B, C, ksize ** 2, h, w)
        x = torch.mul(x, kernel).sum(2).view(B, C, h, w)
        kernel = kernel.view(B, ksize, ksize, *inp.shape[2:]).squeeze()
        return x, kernel

class DegModel(nn.Module):
    def __init__(self, scale=2, nc_img=3):
        super().__init__()
        self.scale = scale
        self.deg_kernel = KernelModel(scale)
        self.quant = Quantization()
    def forward(self, inp):
        # kernel
        x, kernel = self.deg_kernel(inp)
        x = self.quant(x)
        return x, kernel

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))
        self.error_last = 1e8
    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()
        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        timer_data, timer_model = utility.timer(), utility.timer()
        #model = DegModel(scale=2).cuda()
        #model.load_state_dict(torch.load('./model/model_lr.pt'))
        # TEMP
        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()
            #lr = model(hr)
            self.optimizer.zero_grad()
            sr = self.model(lr,0)
            loss = self.loss(sr, hr)
            if(batch-1%20==0):
                print("loss :{}".format(float(loss)))
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()
            timer_model.hold()
            if batch-1 % self.args.print_every == 0:
                self.args.save1 = True
            if (batch + 1) % 100 == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset)*2,
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))
                save_list = [sr]
                postfix = ('SR')
                for v, p in zip(save_list, postfix):
                    normalized = v[0].mul(255 / 1)
                    tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                    imageio.imwrite('{}{}.png'.format('./result_ours1/', batch), tensor_cpu.numpy())
            timer_data.tic()
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()
    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()
        # model = DegModel(scale=2).cuda()
        # model.load_state_dict(torch.load('./model_lr.pt'))
        timer_test = utility.timer()
        self.args.save_results = False
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename in tqdm(d, ncols=80 ):#
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)#
                    sr = utility.quantize(sr, self.args.rgb_range)
                    save_list = [sr,hr,lr]
                    t = sr.clone()
                    h =hr.clone()
                    normalized = t[0].mul(255 / self.args.rgb_range)
                    sr1 = normalized.byte().permute(1, 2, 0).cpu()
                    normalized = h[0].mul(255 / self.args.rgb_range)
                    hr1 = normalized.byte().permute(1, 2, 0).cpu()
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d   #hr
                    )
                    if self.args.save_gt:
                        save_list.extend([lr, hr])
                    self.args.save_results = True
                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale,epoch)
                    self.args.save_results = False
                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )


        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')
        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0,0] + 1 == epoch))#,0
            self.ckp.save(self, epoch)
        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch()
            return epoch >= 60
