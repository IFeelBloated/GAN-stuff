from __future__ import print_function

import argparse
import os
import random
import numpy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import copy
from Network import Generator, Discriminator
from Loss import R1GradientPenalty, RelativisticHingeDiscriminatorLoss, RelativisticHingeGeneratorLoss, PathLengthRegularization


def run():
    torch.set_printoptions(threshold=1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=512, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=1000000, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')

    opt = parser.parse_args()
    print(opt)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
        
    opt.manualSeed = 42
        
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
      
        
    dataset = dset.ImageFolder('./Data',transform=transforms.Compose([
        transforms.Resize(opt.imageSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomHorizontalFlip(p=0.5)
    ]))
    

    

    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize * 2,
                                             shuffle=True, num_workers=int(opt.workers))
    
    device = torch.device("cuda:0" if opt.cuda else "cpu")
    nz = int(opt.nz)
    
    
    ema_kimg = opt.batchSize * 10 / 32
    w_avg_beta = 0.998
    w_avg_decay = 1 - w_avg_beta
    
    gp_weight = 1
    pl_weight = 2
    mean_path_length = 0
    w_avg = torch.zeros(nz)



    
    
    
  #  model = torch.load('epoch_613.pth', map_location='cpu')
   # mean_path_length = model['mean_path_length']
   # w_avg = model['w_avg']
        
        
    









    netG = Generator(nz)
    
   
    
    
    netG = netG.to(device)
    G_ema = copy.deepcopy(netG).eval()
    
  #  netG.load_state_dict(model['g_state_dict'], strict=True)
  #  G_ema.load_state_dict(model['g_ema_state_dict'], strict=True)
   
    print(netG)
    


    netD = Discriminator(nz)
    #netD.load_state_dict(model['d_state_dict'], strict=True)
    
    netD = netD.to(device)
    
    
    print(netD)
    
    
    
    print(sum(p.numel() for p in netG.parameters() if p.requires_grad))
    print([x for x in netG.buffers()])



    fixed_noise = torch.randn(opt.batchSize, nz, device=device)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0, 0.99))
    #optimizerD.load_state_dict(model['optimizerD_state_dict'])
    
    
    
    G_params = [
        {'params': netG.LatentLayer.parameters(), 'lr': 0.01 * opt.lr},
        
        {'params': netG.Layer4x4.parameters()},
        {'params': netG.ToRGB4x4.parameters()},
        
        {'params': netG.Layer8x8.parameters()},
        {'params': netG.ToRGB8x8.parameters()},
        
        {'params': netG.Layer16x16.parameters()},
        {'params': netG.ToRGB16x16.parameters()},
        
        {'params': netG.Layer32x32.parameters()},
        {'params': netG.ToRGB32x32.parameters()},
        
        {'params': netG.Layer64x64.parameters()},
        {'params': netG.ToRGB64x64.parameters()},
        
        {'params': netG.Layer128x128.parameters()},
        {'params': netG.ToRGB128x128.parameters()}
        
    ]
    
    
    optimizerG = optim.Adam(G_params, lr=opt.lr, betas=(0, 0.99))
    #optimizerG.load_state_dict(model['optimizerG_state_dict'])



    for epoch in range(opt.niter): # model['epoch'] + 1
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            
            netD.requires_grad = True
            netG.requires_grad = False
            
            netD.zero_grad(set_to_none=True)
            
            dual_batch_size = data[0].size(0)
            batch_size = dual_batch_size // 2
            
            assert(batch_size * 2 == dual_batch_size)
            
            real = data[0][0:batch_size,:,:,:].to(device)

            
            real.requires_grad = True

            
            noise = torch.randn(batch_size, nz, device=device)

            

            output_r = netD(real)
            
            w, fake = netG(noise)
            output_f = netD(fake.detach())
            
            
           

            r1_penalty = R1GradientPenalty(real, output_r)

            errD = RelativisticHingeDiscriminatorLoss(output_r, output_f) + gp_weight * r1_penalty
            
            
            errD.backward()
                 
            optimizerD.step()
            
            
            
            

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netD.requires_grad = False
            netG.requires_grad = True

            netG.zero_grad(set_to_none=True)
            
            noise = torch.randn(batch_size, nz, device=device)
            
            
            real = data[0][batch_size:2*batch_size,:,:,:].to(device)





            w, fake = netG(noise)
            output_f = netD(fake)
            output_r = netD(real)
            
            
            
            path_length_penalty, mean_path_length = PathLengthRegularization(fake, w, mean_path_length)
            errG = RelativisticHingeGeneratorLoss(output_r, output_f) + pl_weight * path_length_penalty

            
            
            
            
            
            
            errG.backward()
            
           
            

            
            optimizerG.step()
            
            
            
            
            
            
            
            ema_nimg = ema_kimg * 1000
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            
            with torch.no_grad():
                for p_ema, p in zip(G_ema.parameters(), netG.parameters()):
                    p_ema.copy_(p.lerp(p_ema, ema_beta))
                for b_ema, b in zip(G_ema.buffers(), netG.buffers()):
                    b_ema.copy_(b)
            
            
            
            
            noise = torch.randn(batch_size, nz, device=device)
            w = G_ema.LatentLayer(noise)
            w_avg = w_avg + w_avg_decay * (w.mean(0).detach().cpu() - w_avg)
            







            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f PL_Penalty: %.4f R1: %.4f Path_Length: %.4f'
                  % (epoch, opt.niter, i, len(dataloader),
                     errD.item(), errG.item(), pl_weight * path_length_penalty.item(), gp_weight * r1_penalty.item(), mean_path_length) + ' w_avg: ' + str(w_avg).removeprefix('tensor(').removesuffix(')'))
            if i % 100 == 0:
                vutils.save_image(real,
                        '%s/real_samples.png' % opt.outf,
                        normalize=True, nrow=4)


                w, fake = G_ema(fixed_noise)
                vutils.save_image(fake.detach(),
                        '%s/fake_samples_epoch_%04d_%04d.png' % (opt.outf, epoch, i),
                        normalize=True, nrow=4)

                

                
                

                
                
                
                

            if opt.dry_run:
                break
        # do checkpointing
        
        torch.save({
            'epoch': epoch,
            'g_ema_state_dict': G_ema.state_dict(),
            'g_state_dict': netG.state_dict(),
            'd_state_dict': netD.state_dict(),
            'optimizerG_state_dict': optimizerG.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict(),
            'mean_path_length': mean_path_length,
            'w_avg': w_avg,
            'loss_D': errD.detach().item(),
            'loss_G': errG.detach().item(),
            'r1_penalty': gp_weight * r1_penalty.detach().item(),
            'path_length_penalty': pl_weight * path_length_penalty.detach().item(),
            }, '%s/epoch_%d.pth' % (opt.outf, epoch))
        
        
        
        
        
if __name__ == '__main__':
    run()