import torch
from torch.autograd import Variable
import os

from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from datetime import datetime
from utils.dataloader import get_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter
from utils.utils_synapse import test_single_volume, test_single_volume_V3, DiceLoss, powerset
from torch.nn.modules.loss import CrossEntropyLoss
from utils.dataset_synapse import Synapse_dataset, RandomGenerator
from torch.utils.data import DataLoader 
import torch.nn.functional as F
from torchvision import transforms
import random
import imageio
import numpy as np
from lib.Mpvt import HGM_Multi


def structure_loss(iout, label_batch, num_classes=9):
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    lc1, lc2 = 0.3, 0.7
    loss_ce = ce_loss(iout, label_batch[:].long())
    loss_dice = dice_loss(iout, label_batch, softmax=True)
    return (lc1 * loss_ce + lc2 * loss_dice)


def inference(args, model, test_save_path=None):
    classes = ['spleen', 'right kidney', 'left kidney', 'gallbladder', 'pancreas', 'liver', 'stomach', 'aorta']
    db_test = Synapse_dataset(base_dir=args.test_path, split="test_vol", list_dir=args.list_dir, nclass=args.num_classes)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    print("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume_V3(image, label, model, classes=args.num_classes, patch_size=[args.trainsize, args.trainsize],
                                      test_save_path=test_save_path, case=case_name, z_spacing=1, class_names=classes)
        metric_list += np.array(metric_i)
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        print('Mean class (%d) %s mean_dice %f' % (i, classes[i-1], metric_list[i-1]))
    performance = np.mean(metric_list, axis=0)
    print('Testing performance in best val model: mean_dice : %f' % (performance))
    return performance


def train(train_loader, model, optimizer, epoch, device, opt):
    # global Performance, Mean_hd95, Mean_jacard, Mean_asd
    global Performance
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record = AvgMeter()
    l = [0]
    ss = [x for x in powerset(l)]
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack['image'], pack['label']
            images = images.to(device)
            gts = gts.squeeze(1).to(device)
            # ---- rescale ----
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                if gts.dim() == 3:
                    gts = gts.unsqueeze(1).float()   
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = gts.long().squeeze(1)  
            # ---- forward ----

            prediction, prediction2 = model(images)
            # ---- loss function ----
            loss_prediction = structure_loss(prediction, gts, num_classes=opt.num_classes)
            loss = loss_prediction
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)
                
        # ---- train visualization ----
        if i % 100 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], ' '[loss: {:.4f} ]'.format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show()))

    save_path = opt.train_save

    if (epoch+1) % 1 == 0:
        performance = inference(opt, model, opt.results_save_place)
        if performance > Performance:
            Performance = performance
            torch.save(model.state_dict(), save_path)
            print('[Saving parameter:]', save_path)
        model.train()



Performance = 0

test_dataset_name = "Synapse"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int,
                        default=9, help='num_classes')
    parser.add_argument('--epoch', type=int,
                        default=1000, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')
    
    parser.add_argument('--train_path', type=str,
                        default='./Synapse/train_npz_new', help='path to train dataset')
    parser.add_argument('--test_path', type=str,
                        default='./Synapse/test_vol_h5_new', help='path to train dataset')
    parser.add_argument('--list_dir', type=str,
                        default='./Synapse/lists_Synapse', help='list_dir')
    # Preparameter 
    parser.add_argument('--pre_parameter', type=str, default="./HGM_Synapse_best_parameter.pth", help='Load model from a .pth file')
    # Save Checkpoint
    parser.add_argument('--train_save', type=str,default='./parameter/test.pth', help='save pth path')
    # Verification result save address
    parser.add_argument('--results_save_place', type=str, default='./results/'+ test_dataset_name)
    parser.add_argument('--seed', type=int, default=2222, help='random seed')
    opt = parser.parse_args()
    
    torch.backends.cudnn.enabled = False

    # ---- build models ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HGM_Multi(num_classes=opt.num_classes)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    model.to(device=device)


    if opt.pre_parameter:
        # Our best parameter trained with two GPU
        print("############# Use best Parameter #############")
        if torch.cuda.device_count() > 1:
            weights_dict = torch.load(opt.pre_parameter, map_location=device)
            model_state_dict = model.state_dict()
            # if you want to use one GPU, please use this:
            #cleaned_weights_dict = {name.replace('module.', ''): param for name, param in weights_dict.items()}
            for name, param in weights_dict.items():
                if name in model_state_dict and model_state_dict[name].shape == param.shape:
                    model_state_dict[name].copy_(param)
                else:
                    print(f"Skipped loading parameter: {name} - Shape mismatch or not found")     
        else:
            weights_dict = torch.load(opt.pre_parameter, map_location=device)
            model_state_dict = model.state_dict()
            cleaned_weights_dict = {name.replace('module.', ''): param for name, param in weights_dict.items()}

            for name, param in cleaned_weights_dict.items():
                # print(name)
                if name in model_state_dict and model_state_dict[name].shape == param.shape:
                    model_state_dict[name].copy_(param)
                else:
                    print(f"Skipped loading parameter: {name} - Shape mismatch or not found")

        model.load_state_dict(model_state_dict)

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    db_train = Synapse_dataset(base_dir=opt.train_path, list_dir=opt.list_dir, split="train", nclass=opt.num_classes,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[opt.trainsize, opt.trainsize])]))
    def worker_init_fn(worker_id):
        random.seed(opt.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=opt.batchsize, shuffle=True, num_workers=8, pin_memory=True,worker_init_fn=worker_init_fn)
    print("The length of train set is: {}".format(len(db_train)))

    total_step = len(trainloader)
    print("Number of batch: {} ".format(total_step))
    print("#"*20, "Start Training", "#"*20)

    

    for epoch in range(0, opt.epoch):
        # adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(trainloader, model, optimizer, epoch, device, opt)
    
    print("-----------------------------------------")
    print(f"####### Best Average Dice #######: {Performance:.4f}")



