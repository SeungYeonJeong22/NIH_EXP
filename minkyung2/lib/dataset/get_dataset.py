import torchvision.transforms as transforms
from dataset.nihdataset import NIHDataset
from utils.cutout import CutoutPIL_
from randaugment import RandAugment
import os.path as osp

import argparse

def parser_args():
    parser = argparse.ArgumentParser(description='Training')
    args = parser.parse_args()
    return args

def get_args():
    args = parser_args()
    return args

##aprile 2024 TODO
from PIL import Image
import numpy as np
class AdjustContrast(object):
    def __call__(self, img):
        img_array = np.array(img)
        x_min = img_array.min()
        img_array = 255.0 * ((img_array - x_min) / (img_array.max() - x_min))
        return Image.fromarray(img_array.astype('uint8')) #, 'L')






def get_datasets(args):
    # if args.orid_norm:
    #     normalize = transforms.Normalize(mean=[0, 0, 0],
    #                                      std=[1, 1, 1])
    # else:
    #     print("GET_DATASET: applying no normalization (e.g., N(0;1)) to the data")
    #     normalize = transforms.Normalize(mean=[0, 0, 0],
    #                                      std=[1, 1, 1])

    # train_data_transform_list = [transforms.Resize((args.img_size, args.img_size)), #ORIGINALE DI LORO
    #                                         RandAugment(),
    #                                            transforms.ToTensor(),
    #                                            normalize]


    # if args.cutout:
    #     print("Using Cutout!!!")
    #     train_data_transform_list.insert(1, CutoutPIL_(n_holes=args.n_holes, length=args.length))
    #     train_data_transform = transforms.Compose(train_data_transform_list)

    #     test_data_transform = transforms.Compose([
    #                                         transforms.Resize((args.img_size, args.img_size)),
    #                                         transforms.ToTensor(),
    #                                         normalize])







    if args.dataname == 'nih':
        dataset_dir = args.dataset_dir
        # nih_transform = transforms.Compose([
        #     transforms.Resize((args.img_size, args.img_size)),
        #     transforms.ToTensor(),
        #     # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        # ])
        # TODO aprile 2024 replaced the above nih_transform with the new one that adjusts the contrast and normalizes:
        nih_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            AdjustContrast(),  # Custom contrast adjustment, april 2024    
            transforms.ToTensor(), #0...1 range
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) #-1...1 range to speed up and stabilize the training process 
        ]) 
        

        train_dataset = NIHDataset(
            data_path = dataset_dir,
            input_transform = nih_transform,
            train=True
        )
        val_dataset = NIHDataset(
            data_path=dataset_dir,
            input_transform = nih_transform,
            train=False
        )
    else:
        raise NotImplementedError("Unknown dataname %s" % args.dataname)

    print("len(train_dataset):", len(train_dataset)) 
    print("len(val_dataset):", len(val_dataset))
    return train_dataset, val_dataset
