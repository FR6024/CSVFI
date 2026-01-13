import time
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import config
import myutils
from loss import Loss
import shutil
import os
import numpy as np
import random
import torch.distributed as dist

# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

def load_checkpoint(args, model, optimizer, path, local_rank):
    print("loading checkpoint %s" % path)
    checkpoint = torch.load(path, map_location='cuda:{}'.format(local_rank))
    args.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = checkpoint.get("lr", args.lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def init_ddp(backend='nccl', **kwargs):
    """initialization for distributed training"""
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, **kwargs)


lr_schular = [8e-4, 4e-4, 20e-5, 10e-5, 20e-6, 4e-6]
training_schedule = [40, 60, 75, 85, 95, 100]


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate"""
    for i in range(len(training_schedule)):
        if epoch < training_schedule[i]:
            current_learning_rate = lr_schular[i]
            break

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_learning_rate
        print('Learning rate sets to {}.'.format(param_group['lr']))


def train(args, model, train_dloader, epoch, criterion, optimizer):
    torch.cuda.empty_cache()
    losses, psnrs, ssims = myutils.init_meters(args.loss)

    model.train()
    criterion.train()

    for i, (images, gt_image, images_edge, _) in enumerate(tqdm(train_dloader)):

        # Build input batch
        images = [img_.cuda() for img_ in images]
        images_edge = [img_.cuda() for img_ in images_edge]

        # Forward
        optimizer.zero_grad()
        out_ll, out_l, out = model(images, images_edge)

        gt = gt_image.cuda()

        loss, _ = criterion(out, gt)
        overall_loss = loss

        losses['total'].update(loss.item())
        overall_loss.backward()
        optimizer.step()
        # Calc metrics & print logs
        if i % args.log_iter == 0:  # 100
            myutils.eval_metrics(out, gt, psnrs, ssims)
            dist.reduce(psnrs.avg, 0, op=dist.ReduceOp.SUM)
            dist.reduce(ssims.avg, 0, op=dist.ReduceOp.SUM)
            dist.reduce(torch.tensor(losses['total'].avg).cuda(), 0, op=dist.ReduceOp.SUM)

            local_rank = dist.get_rank()
            if local_rank==0:
                world_size = dist.get_world_size()
                print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}\tPSNR: {:.4f}  Lr:{:.6f}  '.format(
                    epoch, i, len(train_dloader), losses['total'].avg/world_size, psnrs.avg/world_size, optimizer.param_groups[0]['lr'], flush=True))

            # Reset metrics
            losses, psnrs, ssims = myutils.init_meters(args.loss)


def tes(args, model, test_dloader, epoch, criterion):
    print('Testing for epoch = %d' % epoch)
    losses, psnrs, ssims = myutils.init_meters(args.loss)
    model.eval()
    criterion.eval()
    torch.cuda.empty_cache()

    t = time.time()
    with torch.no_grad():
        for i, (images, gt_image, gt_path, images_edge, _) in enumerate(tqdm(test_dloader)):

            images = [img_.cuda() for img_ in images]
            images_edge = [img_.cuda() for img_ in images_edge]
            gt = gt_image.cuda()

            out = model(images, images_edge)  # images is a list of neighboring frames

            # Save loss values
            loss, loss_specific = criterion(out, gt)
            for k, v in losses.items():
                if k != 'total':
                    v.update(loss_specific[k].item())
            losses['total'].update(loss.item())

            # Evaluate metrics
            myutils.eval_metrics(out, gt, psnrs, ssims)

    return losses['total'].avg, psnrs.avg, ssims.avg


def print_log(epoch, num_epochs, one_epoch_time, oup_pnsr, oup_ssim, Lr):
    print('({0:.0f}s) Epoch [{1}/{2}], Val_PSNR:{3:.2f}, Val_SSIM:{4:.4f}'
          .format(one_epoch_time, epoch, num_epochs, oup_pnsr, oup_ssim))
    # write training log
    with open('./training_log/train_log.txt', 'a') as f:
        print(
            'Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Val_PSNR:{4:.2f}, Val_SSIM:{5:.4f}, Lr:{6}'
            .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    one_epoch_time, epoch, num_epochs, oup_pnsr, oup_ssim, Lr), file=f)

def main(args):
    init_ddp()
    local_rank = dist.get_rank()

    set_seed(args.random_seed+local_rank)

    if args.model == 'CSVFI_T':
        from model.CSVFI_T import UNet_3D_3D
    elif args.model == 'CSVFI_S':
        from model.CSVFI_S import UNet_3D_3D
    elif args.model == 'CSVFI_B':
        from model.CSVFI_B import UNet_3D_3D

    model = UNet_3D_3D(n_inputs=args.nbr_frame, joinType=args.joinType)  # nbr_frame，default=4
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    if local_rank==0:
        print('----- generator parameters: %f -----' % (sum(p.numel() for p in model.parameters() if p.requires_grad) / (10 ** 6)))

    ##### Define Loss & Optimizer #####
    criterion = Loss(args)  # 自实现Loss

    from torch.optim import Adamax
    optimizer = Adamax(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # DataLoader
    if args.dataset == "vimeo90K_septuplet":
        from dataset.vimeo90k_septuplet import VimeoSepTuplet
    else:
        raise NotImplementedError

    train_dataset = VimeoSepTuplet(args.data_root, is_training=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    test_dataset = VimeoSepTuplet(args.data_root, is_training=False)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_dloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)

    if args.stop==True:
        load_checkpoint(args, model, optimizer, save_loc+'/checkpoint.pth', local_rank)

    best_psnr = 0
    for epoch in range(args.start_epoch, args.max_epoch):  # 100
        adjust_learning_rate(optimizer, epoch)
        start_time = time.time()
        train(args, model, train_dloader, epoch, criterion, optimizer)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{args.max_epoch}, Learning Rate: {current_lr:.6f}")

        if local_rank == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr': optimizer.param_groups[-1]['lr']
            }, os.path.join(save_loc, 'checkpoint.pth'))

        test_loss, psnr, ssim = tes(args, model, test_dloader, epoch, criterion)

        dist.reduce(torch.tensor(test_loss).cuda(), 0, op=dist.ReduceOp.SUM)
        dist.reduce(psnr, 0, op=dist.ReduceOp.SUM)
        dist.reduce(ssim, 0, op=dist.ReduceOp.SUM)

        if local_rank == 0:
            world_size = dist.get_world_size()

        # save checkpoint
            is_best = psnr/world_size > best_psnr
            best_psnr = max(psnr/world_size, best_psnr)
            if is_best:
                shutil.copyfile(os.path.join(save_loc, 'checkpoint.pth'), os.path.join(save_loc, 'model_best.pth'))

            one_epoch_time = time.time() - start_time
            print_log(epoch, args.max_epoch, one_epoch_time, psnr/world_size, ssim/world_size, optimizer.param_groups[0]['lr'])

    dist.destroy_process_group()


if __name__ == "__main__":
    args, unparsed = config.get_args()
    cwd = os.getcwd()
    print(args)

    save_loc = os.path.join(args.checkpoint_dir, "checkpoints")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    main(args)
