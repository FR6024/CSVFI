import os
import time
import random
import torch
from tqdm import tqdm
import config
import myutils
import math
import torch.nn.functional as F
from lpips import lpips
from stlpips_pytorch import stlpips
from piq import PieAPP


##### Parse CmdLine Arguments #####
# os.environ["CUDA_VISIBLE_DEVICES"]='0'
args, unparsed = config.get_args()
cwd = os.getcwd()

device = torch.device('cuda' if args.cuda else 'cpu')

torch.manual_seed(args.random_seed)
random.seed(args.random_seed)  # 新加的
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)

if args.dataset == "vimeo90K_septuplet":
    from dataset.vimeo90k_septuplet import get_loader
    test_loader = get_loader('test', args.data_root, args.test_batch_size, shuffle=False, num_workers=args.num_workers)
    print(len(test_loader))
elif args.dataset == "ucf101":
    from dataset.ucf101_test import get_loader
    test_loader = get_loader(args.data_root, args.test_batch_size, shuffle=False, num_workers=args.num_workers)
elif args.dataset == "Davis":
    from dataset.Davis_test import get_loader
    test_loader = get_loader(args.data_root, args.test_batch_size, shuffle=False, num_workers=args.num_workers, test_mode=True)
elif args.dataset == "SNU":
    from dataset.snufilm import get_loader
    test_loader = get_loader(args.data_root, 1, shuffle=False, num_workers=1, mode=args.snu_model)


if args.model == 'CSVFI_T':
    from model.CSVFI_T import UNet_3D_3D
elif args.model == 'CSVFI_S':
    from model.CSVFI_S import UNet_3D_3D
elif args.model == 'CSVFI_B':
    from model.CSVFI_B import UNet_3D_3D


print("Building model: %s" % args.model)
model = UNet_3D_3D(n_inputs=args.nbr_frame, joinType=args.joinType)

model = torch.nn.DataParallel(model).to(device)
print("#params", sum([p.numel() for p in model.parameters()]))


def to_psnr(rect, gt):
    mse = F.mse_loss(rect, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]
    psnr_list = [-10.0 * math.log10(mse) for mse in mse_list]
    return psnr_list


lpips_model = lpips.LPIPS(net='alex').to(device)
stlpips_metric = stlpips.LPIPS(net="alex", variant="shift_tolerant").to(device)
pieapp = PieAPP()


def calculate_lpips(imgs, gts):
    socers = 0
    for i in range(imgs.shape[0]):
        lpips_scores = lpips_model(imgs[i], gts[i])
        socers += lpips_scores.item()
    return socers/imgs.shape[0]


def calculate_st_lpips(imgs, gts):
    socers = 0
    for i in range(imgs.shape[0]):
        stlpips_scores = stlpips_metric(imgs[i], gts[i])
        socers += stlpips_scores.item()
    return socers/imgs.shape[0]


def tes(args):
    time_taken = []
    losses, psnrs, ssims = myutils.init_meters(args.loss)
    model.eval()
    with torch.no_grad():
        if args.dataset == "vimeo90K_septuplet":
            lpips_score = []
            stlpips_score = []
            pieapp_score = []
            for i, (images, gt_image, path, images_edge, _) in enumerate(tqdm(test_loader)):  # 90K
                images = [img_.cuda() for img_ in images]
                images_edge = [img_.cuda() for img_ in images_edge]
                gt = gt_image.cuda()
                torch.cuda.synchronize()
                start_time = time.time()
                out = model(images, images_edge)
                torch.cuda.synchronize()
                time_taken.append(time.time() - start_time)
                myutils.eval_metrics(out, gt, psnrs, ssims)

                lpips_score.append(calculate_lpips(out, gt))
                stlpips_score.append(calculate_st_lpips(out, gt))
                pieapp_score.append(pieapp(torch.clamp(out, 0, 1), gt).item())

            print("PSNR: %f, SSIM: %fn" %(psnrs.avg, ssims.avg))
            print("LPIPS: ", sum(lpips_score) / len(lpips_score))
            print("ST-LPIPS: ", sum(stlpips_score) / len(stlpips_score))
            print("PieAPP: ", sum(pieapp_score) / len(pieapp_score))
            print("Time , ", sum(time_taken) / len(time_taken))

            return psnrs.avg

        elif args.dataset == "ucf101":
            lpips_score = []
            stlpips_score = []
            pieapp_score = []
            for i, (images, images_edge, gt_image) in enumerate(tqdm(test_loader)):
                images = [img_.cuda() for img_ in images]
                images_edge = [img_.cuda() for img_ in images_edge]
                gt = gt_image.cuda()
                torch.cuda.synchronize()
                start_time = time.time()
                out = model(images, images_edge)
                torch.cuda.synchronize()
                time_taken.append(time.time() - start_time)
                myutils.eval_metrics(out, gt, psnrs, ssims)

                lpips_score.append(calculate_lpips(out, gt))
                stlpips_score.append(calculate_st_lpips(out, gt))
                pieapp_score.append(pieapp(torch.clamp(out, 0, 1), gt).item())

            print("PSNR: %f, SSIM: %fn" %(psnrs.avg, ssims.avg))
            print("LPIPS: ", sum(lpips_score) / len(lpips_score))
            print("ST-LPIPS: ", sum(stlpips_score) / len(stlpips_score))
            print("PieAPP: ", sum(pieapp_score) / len(pieapp_score))
            print("Time , ", sum(time_taken) / len(time_taken))

            return psnrs.avg

        elif args.dataset == "Davis":
            lpips_score = []
            stlpips_score = []
            pieapp_score = []
            for i, (images, images_edge, gt_image, path) in enumerate(tqdm(test_loader)):
                images = [img_.cuda() for img_ in images]
                images_edge = [img_.cuda() for img_ in images_edge]
                gt = gt_image.cuda()
                torch.cuda.synchronize()
                start_time = time.time()
                out = model(images, images_edge)
                torch.cuda.synchronize()
                time_taken.append(time.time() - start_time)
                myutils.eval_metrics(out, gt, psnrs, ssims)

                lpips_score.append(calculate_lpips(out, gt))
                stlpips_score.append(calculate_st_lpips(out, gt))
                pieapp_score.append(pieapp(torch.clamp(out, 0, 1), gt).item())

            print("PSNR: %f, SSIM: %fn" % (psnrs.avg, ssims.avg))
            print("LPIPS: ", sum(lpips_score) / len(lpips_score))
            print("ST-LPIPS: ", sum(stlpips_score) / len(stlpips_score))
            print("PieAPP: ", sum(pieapp_score) / len(pieapp_score))
            print("Time , ", sum(time_taken) / len(time_taken))

            return psnrs.avg

        elif args.dataset == "SNU":
            lpips_score = []
            stlpips_score = []
            pieapp_score = []
            for i, (images, images_edge, gt_image, sk, pad, (W, H), path) in enumerate(tqdm(test_loader)):
                if sk:
                    # print(f"Batch {i} contains None values and is skipped.")
                    continue

                images = [img_.cuda() for img_ in images]
                images_edge = [img_.cuda() for img_ in images_edge]
                gt = gt_image.cuda()
                torch.cuda.synchronize()
                start_time = time.time()
                out = model(images, images_edge)
                if pad:
                    out = out[:, :, (out.shape[2] - H) // 2:(out.shape[2] + H) // 2,
                          (out.shape[3] - W) // 2:(out.shape[3] + W) // 2]
                    gt = gt[:, :, (gt.shape[2] - H) // 2:(gt.shape[2] + H) // 2,
                         (gt.shape[3] - W) // 2:(gt.shape[3] + W) // 2]

                torch.cuda.synchronize()
                time_taken.append(time.time() - start_time)
                myutils.eval_metrics(out, gt, psnrs, ssims)

                lpips_score.append(calculate_lpips(out, gt))
                stlpips_score.append(calculate_st_lpips(out, gt))
                pieapp_score.append(pieapp(torch.clamp(out, 0, 1), gt).item())

            print("PSNR: %f, SSIM: %fn" % (psnrs.avg, ssims.avg))
            print("LPIPS: ", sum(lpips_score) / len(lpips_score))
            print("ST-LPIPS: ", sum(stlpips_score) / len(stlpips_score))
            print("PieAPP: ", sum(pieapp_score) / len(pieapp_score))
            print("Time , ", sum(time_taken) / len(time_taken))

            return psnrs.avg


""" Entry Point """
def main(args):
    assert args.load_from is not None
    model_dict = model.state_dict()
    model.load_state_dict(torch.load(args.load_from)["state_dict"], strict=True)
    tes(args)


if __name__ == "__main__":
    main(args)
