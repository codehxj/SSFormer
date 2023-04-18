import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import torch
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import utility
import data
# noinspection PyUnresolvedReferences
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    torch.backends.cudnn.enabled = False
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:

            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            print('Total params: %.2fM' % (sum(p.numel() for p in _model.parameters())/1000000.0))
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None #损失函数
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
               # t.train()  #通过训练传递进来参数cpoch
                t.test()


            checkpoint.done()

if __name__ == '__main__':
    main()
