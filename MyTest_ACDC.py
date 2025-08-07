import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from tqdm import tqdm
import imageio
from utils.dataloader import test_dataset
from utils.dataloader import test_dataset, EvalDataset
from utils.dataset_ACDC import ACDCdataset, RandomGenerator
from utils.utils_synapse import test_single_volume, DiceLoss, powerset
from torch.utils.data import DataLoader 
from lib.Mpvt import HGM_Multi
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# segmentation-mask-overlay==0.3.4
# pip install segmentation-mask-overlay==0.3.4

def inference(args, model, test_save_path=None):
    classes = ['RV', 'Myo', 'LV']
    db_test =ACDCdataset(base_dir=args.test_path,list_dir=args.list_dir, split="test")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False)
    print("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.trainsize, args.trainsize],
                                      test_save_path=test_save_path, case=case_name, z_spacing=10)
        metric_list += np.array(metric_i)
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        print('Mean class (%d) %s mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (i, classes[i-1], metric_list[i-1][0], metric_list[i-1][1], metric_list[i-1][2], metric_list[i-1][3]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    mean_jacard = np.mean(metric_list, axis=0)[2]
    mean_asd = np.mean(metric_list, axis=0)[3]
    print('Testing performance in best val model: mean_dice : %f mean_hd95 : %f, mean_jacard : %f mean_asd : %f' % (performance, mean_hd95, mean_jacard, mean_asd))
    return performance, mean_hd95, mean_jacard, mean_asd


test_dataset_name = "ACDC"

parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=4, help='num_classes')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--pth_path', type=str, default='./HGM_ACDC_best_parameter.pth')
parser.add_argument('--test_path', type=str,default='./ACDC/test', help='path to train dataset')
parser.add_argument('--results_save_place', type=str, default='./results/'+ test_dataset_name +'_01')
parser.add_argument('--list_dir', type=str, default='./lists_ACDC', help='list_dir')
opt = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = HGM_Multi(num_classes=opt.num_classes)

weights_dict = torch.load(opt.pth_path, map_location=device)
model_state_dict = model.state_dict()

cleaned_weights_dict = {name.replace('module.', ''): param for name, param in weights_dict.items()}

for name, param in cleaned_weights_dict.items():
    if name in model_state_dict and model_state_dict[name].shape == param.shape:
        model_state_dict[name].copy_(param)
    else:
        print(f"Skipped loading parameter: {name} - Shape mismatch or not found")

model.load_state_dict(model_state_dict)

model = model.to(device)
model.eval()
performance, mean_hd95, mean_jacard, mean_asd = inference(opt, model, opt.results_save_place)