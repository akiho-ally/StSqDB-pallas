from comet_ml import Experiment
from tqdm import tqdm
from dataloader import StsqDB, Normalize, ToTensor
from model import EventDetector
from util import *
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os



if __name__ == '__main__':

    experiment = Experiment(api_key='d7Xjw6KSK6KL7pUOhXJvONq9j', project_name='stsqdb')
    hyper_params = {
    'batch_size': 8,
    'iterations' : 3000,
    }

    experiment.log_parameters(hyper_params)

    # training configuration
    split = 1
    iterations = 3000
    it_save = 100  # save model every 100 iterations
    n_cpu = 6
    seq_length = 300
    bs = 8  # batch size
    k = 10  # frozen layers

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Load Model')

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          device=device,
                          bidirectional=True,
                          dropout=False
                          )
    #print('model.py, class EventDetector()')

    freeze_layers(k, model)
    #print('utils.py, func freeze_laters()')
    model.train()
    model.to(device)
    print('Loading Data')


    # TODO: vid_dirのpathをかえる。stsqの動画を切り出したimage全部が含まれているdirにする
    dataset = StsqDB(data_file='train_split_{}.pkl'.format(split),
                     vid_dir='data/videos_40/',
                     seq_length=seq_length,
                     transform=transforms.Compose([ToTensor(),
                                                   Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                     train=True)
    print('dataloader.py, class StsqDB()')
    # dataset.__len__() : 1050


    data_loader = DataLoader(dataset,
                             batch_size=bs,
                             shuffle=True,
                             num_workers=n_cpu,
                             drop_last=True)

    # dataset.__len__() : 47 (dataset/bs)
                  

    # the 8 golf swing events are classes 0 through 7, no-event is class 8
    # the ratio of events to no-events is approximately 1:35 so weight classes accordingly:
    # TODO: edit weights shape from golf-8-element to stsq-12-element
    weights = torch.FloatTensor([1/3, 1, 2/5, 1/3, 1/6, 1, 1/4, 1, 1/4, 1/3, 1/2, 1/6, 1/60]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)  ##lambda:無名関数

    losses = AverageMeter()
    #print('utils.py, class AverageMeter()')

    if not os.path.exists('models'):
        os.mkdir('models')

    i = 0

    while i < iterations:
        for sample in tqdm(data_loader):
            images, labels = sample['images'].to(device), sample['labels'].to(device)
            logits = model(images)       
            labels = labels.view(bs*seq_length)  ##??
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward() 
            losses.update(loss.item(), images.size(0))
            optimizer.step()

            

            print('Iteration: {}\tLoss: {loss.val:.4f} ({loss.avg:.4f})'.format(i, loss=losses))
            i += 1
            if i % it_save == 0:
                torch.save({'optimizer_state_dict': optimizer.state_dict(),
                            'model_state_dict': model.state_dict()}, 'models/swingnet_{}.pth.tar'.format(i))
            if i == iterations:
                break

        experiment.log_metrics("train_loss", losses, step=iterations)