import time
import torch
import random

import torch.nn.functional as F

import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid

from utils import *
from options import TrainOptions
from models import Dehaze
from losses import LossCont, LossFreqReco
from datasets import PairedImgDataset
print('---------------------------------------- step 1/5 : parameters preparing... ----------------------------------------')

opt = TrainOptions().parse()

set_random_seed(opt.seed)

models_dir, log_dir, train_images_dir, val_images_dir = prepare_dir(opt.results_dir, opt.experiment, delete=(not opt.resume))

writer = SummaryWriter(log_dir=log_dir)

print('---------------------------------------- step 2/5 : data loading... ------------------------------------------------')
print('training data loading...')

train_dataset = PairedImgDataset(data_source=opt.data_source, mode='train', crop=opt.crop, random_resize=None)
train_dataloader = DataLoader(train_dataset, batch_size=opt.train_bs, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
print('successfully loading training pairs. =====> qty:{} bs:{}'.format(len(train_dataset),opt.train_bs))

print('validating data loading...')

val_dataset = PairedImgDataset(data_source=opt.data_source, mode='val')
val_dataloader = DataLoader(val_dataset, batch_size=opt.val_bs, shuffle=False, num_workers=opt.num_workers, pin_memory=True)
print('successfully loading validating pairs. =====> qty:{} bs:{}'.format(len(val_dataset),opt.val_bs))

print('---------------------------------------- step 3/5 : model defining... ----------------------------------------------')
model = Dehaze().cuda()

if opt.data_parallel:
    model = nn.DataParallel(model)
print_para_num(model)

if opt.pretrained is not None:
    model.load_state_dict(torch.load(opt.pretrained))
    print('successfully loading pretrained model.')

print('---------------------------------------- step 4/5 : requisites defining... -----------------------------------------')
criterion_cont = LossCont()
criterion_fft = LossFreqReco()

optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100,200,300,400,500,600], 0.5)

print('---------------------------------------- step 5/5 : training... ----------------------------------------------------')
def main():
    
    optimal = [0.]
    start_epoch = 1
    if opt.resume:
        state = torch.load(models_dir + '/latest.pth')
        model.load_state_dict(state['model'])
        start_epoch = state['epoch'] + 1
        optimal = state['optimal']
        print('Resume from epoch %d' % (start_epoch), optimal)
    
    for epoch in range(start_epoch, opt.n_epochs + 1):
        train(epoch, optimal)
        
        if (epoch) % opt.val_gap == 0:
            val(epoch, optimal)
        
    writer.close()

def train(epoch, optimal):
    model.train()
    
    
    max_iter = len(train_dataloader)
        
    iter_cont_meter_left = AverageMeter()
    iter_cont_meter_right = AverageMeter()
    iter_fft_meter_left = AverageMeter()
    iter_fft_meter_right = AverageMeter()
    iter_timer = Timer()
    
    for i, (haze_left,haze_right,gt_left,gt_right) in enumerate(train_dataloader):
        haze_left, haze_right, gt_left, gt_right = haze_left.cuda(), haze_right.cuda(), gt_left.cuda(), gt_right.cuda()
        cur_batch = haze_left.shape[0]
        
        H = gt_left.size(2)          
        W = gt_left.size(3)
        gt_left_1 = gt_left[:,:,0:int(H/2),:]
        gt_left_2 = gt_left[:,:,int(H/2):H,:]

        gt_right_1 = gt_right[:,:,0:int(H/2),:]
        gt_right_2 = gt_right[:,:,int(H/2):H,:]
        
        optimizer.zero_grad()
        dehaze_left, dehaze_right, left_lv3_top,left_lv3_bot,right_lv3_top,right_lv3_bot,left_lv2,right_lv2 = model(haze_left,haze_right)
        
        # main loss
        loss_cont_left = criterion_cont(dehaze_left, gt_left)
        loss_cont_right = criterion_cont(dehaze_right, gt_right)
        loss_fft_left = criterion_fft(dehaze_left, gt_left)
        loss_fft_right = criterion_fft(dehaze_right, gt_right)
        loss3 = loss_cont_left + loss_cont_right + 0.1 * loss_fft_left + 0.1 * loss_fft_right
        
        # stage1 loss
        loss_cont_left_lv3_top = criterion_cont(left_lv3_top, gt_left_1)
        loss_cont_left_lv3_bot = criterion_cont(left_lv3_bot, gt_left_2)
        
        loss_cont_right_lv3_top = criterion_cont(right_lv3_top, gt_right_1)
        loss_cont_right_lv3_bot = criterion_cont(right_lv3_bot, gt_right_2)
        
        loss_fft_left_lv3_top = criterion_fft(left_lv3_top, gt_left_1)
        loss_fft_left_lv3_bot = criterion_fft(left_lv3_bot, gt_left_2)
        loss_fft_right_lv3_top = criterion_fft(right_lv3_top, gt_right_1)
        loss_fft_right_lv3_bot = criterion_fft(right_lv3_bot, gt_right_2)
        
        loss1 = loss_cont_left_lv3_top + loss_cont_left_lv3_bot + loss_cont_right_lv3_top +loss_cont_right_lv3_bot +0.1 * (loss_fft_left_lv3_top + loss_fft_left_lv3_bot + loss_fft_right_lv3_top + loss_fft_right_lv3_bot)
        
        # stage2 loss
        loss_cont_left_lv2 = criterion_cont(left_lv2, gt_left)
        loss_cont_right_lv2 = criterion_cont(right_lv2, gt_right)
        loss_fft_left_lv2 = criterion_fft(left_lv2, gt_left)
        loss_fft_right_lv2 = criterion_fft(right_lv2, gt_right)
        loss2 = loss_cont_left_lv2 + loss_cont_right_lv2 + 0.1 *(loss_fft_left_lv2 + loss_fft_right_lv2)                

        loss = loss1 + loss2 + loss3
        
        loss.backward()
        optimizer.step()
        
        iter_cont_meter_left.update(loss_cont_left.item()*cur_batch, cur_batch)
        iter_cont_meter_right.update(loss_cont_right.item()*cur_batch, cur_batch)
        iter_fft_meter_left.update(loss_fft_left.item()*cur_batch, cur_batch)
        iter_fft_meter_right.update(loss_fft_right.item()*cur_batch, cur_batch)
        
        
        if i == 0:
            save_image(torch.cat((haze_left,dehaze_left.detach(),gt_left),0), train_images_dir + '/epoch_{:0>4}_iter_{:0>4}.png'.format(epoch, i+1), nrow=opt.train_bs, normalize=True, scale_each=True)
            
        if (i+1) % opt.print_gap == 0:
            #print('Training: Epoch[{:0>4}/{:0>4}] Iteration[{:0>4}/{:0>4}] Loss_cont: {:.4f} Loss_fft: {:.4f} Time: {:.4f}'.format(epoch, opt.n_epochs, i + 1, max_iter, iter_cont_meter.average(), iter_fft_meter.average(), iter_timer.timeit()))
            print(f'\rTraining: Epoch: {epoch}/{opt.n_epochs} Iteration: {i + 1}/{max_iter}  Loss_cont_left: {iter_cont_meter_left.average():.4f}  Loss_cont_right: {iter_cont_meter_right.average():.4f}  Loss_fft_left: {iter_fft_meter_left.average():.4f}   Loss_fft_right: {iter_fft_meter_right.average():.4f}  Time: {iter_timer.timeit():.4f}',end='',flush=True)
            
            writer.add_scalar('Loss_cont_left', iter_cont_meter_left.average(auto_reset=True), i+1 + (epoch - 1) * max_iter)
            writer.add_scalar('Loss_cont_right', iter_cont_meter_right.average(auto_reset=True), i+1 + (epoch - 1) * max_iter)
            writer.add_scalar('Loss_fft_left', iter_fft_meter_left.average(auto_reset=True), i+1 + (epoch - 1) * max_iter)
            writer.add_scalar('Lossfft_right', iter_fft_meter_right.average(auto_reset=True), i+1 + (epoch - 1) * max_iter)

            
    writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
    torch.save({'model': model.state_dict(), 'epoch': epoch, 'optimal': optimal}, models_dir + '/latest.pth')
    scheduler.step()

def check_image_size(x):
    n, c, h, w = x.shape
    pad_pow = 2**3
    h_pad = pad_pow - h % pad_pow if not h % pad_pow == 0 else 0
    w_pad = pad_pow - w % pad_pow if not w % pad_pow == 0 else 0
    x = F.pad(x, (0, w_pad, 0, h_pad), 'replicate')
    return x

def val(epoch, optimal):
    model.eval()
    
    print(''); print('Validating...', end=' ')
    
    psnr_meter_left = AverageMeter()
    psnr_meter_right = AverageMeter()
    timer = Timer()
    
    for i, (haze_left,haze_right,gt_left,gt_right) in enumerate(val_dataloader):
        haze_left, haze_right, gt_left, gt_right = haze_left.cuda(), haze_right.cuda(), gt_left.cuda(), gt_right.cuda()
        
        haze_left = check_image_size(haze_left)
        haze_right = check_image_size(haze_right)
        gt_left = check_image_size(gt_left)
        gt_right = check_image_size(gt_right)

        max_iter = len(val_dataloader)
        
        with torch.no_grad():
            dehaze_left, dehaze_right, left_lv3_top,left_lv3_bot,right_lv3_top,right_lv3_bot,left_lv2,right_lv2 = model(haze_left,haze_right)
        preds_clip_left = torch.clamp(dehaze_left, 0, 1)
        preds_clip_right = torch.clamp(dehaze_right, 0, 1)
        
        psnr_meter_left.update(get_metrics(preds_clip_left, gt_left), haze_left.shape[0])
        psnr_meter_right.update(get_metrics(preds_clip_right,gt_right), haze_left.shape[0])
        
        if i == 0:
            #if epoch == opt.val_gap:
                #save_image(haze_left, val_images_dir + '/epoch_{:0>4}_iter_{:0>4}_img.png'.format(epoch, i+1), nrow=opt.val_bs, normalize=True, scale_each=True)
                #save_image(gt_left, val_images_dir + '/epoch_{:0>4}_iter_{:0>4}_gt.png'.format(epoch, i+1), nrow=opt.val_bs, normalize=True, scale_each=True)
            save_image(dehaze_left, val_images_dir + '/epoch_{:0>4}_iter_{:0>4}_restored.png'.format(epoch, i+1), nrow=opt.val_bs, normalize=True, scale_each=True)
        print(f'\rValidating: Iteration: {i + 1}/{max_iter}',end='', flush=True)
        
        if i == 50:
            break
    #print('Epoch[{:0>4}/{:0>4}] PSNR: {:.4f} Time: {:.4f}'.format(epoch, opt.n_epochs, psnr_meter.average(), timer.timeit())); print('')
    print(f'\rTraining: Epoch: {epoch}/{opt.n_epochs} PSNR_LEFT: {psnr_meter_left.average():.4f}   PSNR_RIGHT: {psnr_meter_right.average():.4f} Time: {timer.timeit():.4f}',end='',flush=True)
    
    if optimal[0] < psnr_meter_left.average():
        optimal[0] = psnr_meter_left.average()
        torch.save(model.state_dict(), models_dir + '/optimal_{:.2f}_epoch_{:0>4}.pth'.format(optimal[0], epoch))
        
    writer.add_scalar('psnr_left', psnr_meter_left.average(), epoch)
    writer.add_scalar('psnr_right', psnr_meter_right.average(), epoch)
    
    #torch.save(model.state_dict(), models_dir + '/epoch_{:0>4}.pth'.format(epoch))

if __name__ == '__main__':
    main()

