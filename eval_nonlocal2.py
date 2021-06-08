import torch
from torch.utils import data
from dataset import Dataset
from models_nonlocal_2 import Model
import os
import argparse
import cv2
import numpy as np
from visualize import save_results,save_density_map
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='SHA', type=str, help='dataset')
parser.add_argument('--data_path', default=r'D:\dataset', type=str, help='path to dataset')
parser.add_argument('--save_path', default=r'D:\checkpoint\SFANet', type=str, help='path to save checkpoint')
parser.add_argument('--gpu', default=3, type=int, help='gpu id')
parser.add_argument('--isvis', default=0, type=int, help='gpu id')
args = parser.parse_args()

test_dataset = Dataset(args.data_path, args.dataset, False)
test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device('cuda:' + str(args.gpu))

model = Model().to(device)

checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint_best.pth'))
model.load_state_dict(checkpoint['model'])

output_dir='./vis/'
model.eval()
with torch.no_grad():
    mae, mse = 0.0, 0.0
    for i, (images, gt,density_gt) in enumerate(test_loader):
        images = images.to(device)

        predict, _ = model(images)

        if args.isvis:
            density_map = predict.data.cpu().numpy()
            density_gt = density_gt.cpu().numpy()
            #images = images.cpu().numpy()
            save_results(images, density_gt,density_map,output_dir, fname=str(i)+'.png')
        #print(density_map.shape) 
        #save_density_map(density_map,output_dir, fname='results.png')
        #save_density_map(gt[0],output_dir, fname='gt.png')
        print('predict:{:.2f} label:{:.2f}'.format(predict.sum().item(), gt.item()))
        mae += torch.abs(predict.sum() - gt).item()
        mse += ((predict.sum() - gt) ** 2).item()

    mae /= len(test_loader)
    mse /= len(test_loader)
    mse = mse ** 0.5
    print('MAE:', mae, 'MSE:', mse)
