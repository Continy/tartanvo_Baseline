from Datasets.utils import ToTensor, Compose, CropCenter, DownscaleFlow
from Datasets.transformation import motion2pose_pypose, cvtSE3_pypose
from Datasets.TrajFolderDataset import TrajFolderDataset


from pathlib import Path
from TartanVO import TartanVO
import torch
from torch.utils.data import DataLoader
from yacs.config import CfgNode as CN


import pypose as pp
import numpy as np
import os
from tqdm import tqdm
import gc
import yaml


EDN2NED = pp.from_matrix(
    torch.tensor([[0., 0., 1., 0.], [1., 0., 0., 0.], [0., 1., 0., 0.],
                  [0., 0., 0., 1.]],
                 dtype=torch.float32), pp.SE3_type).to('cuda')


class ColoredTqdm(tqdm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args,
                         colour="yellow",
                         ascii="░▒█",
                         dynamic_ncols=True,
                         **kwargs)

    def close(self, *args, **kwargs):
        if self.n < self.total:
            self.colour = "red"
            self.desc = "❌ Error"
        else:
            self.colour = "#35aca4"
            self.desc = "✅ Finish"
        super().close(*args, **kwargs)


def build_cfg(yaml_path):
    with open(yaml_path, "r") as f:
        cfg = CN(yaml.safe_load(f))
    return cfg


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        default='configs/local_tartanv1.yaml')
    parser.add_argument('--savepath', type=str, default='./Results')
    parser.add_argument('--data_type', type=str, default='v1')

    cfg = parser.parse_args()
    args = build_cfg(cfg.config)
    args.update(vars(cfg))
    
    cropsize = (448, 640)

    root = args.ROOT

    for subset in args.DATASETS:
        motion_list = []
        torch.cuda.empty_cache()
        path = str(Path(root) / subset)
        print('Loading dataset:', path)

        transform = Compose([
            CropCenter(cropsize),
            DownscaleFlow(),
            ToTensor()
        ])

        dataset = TrajFolderDataset(datadir=path,
                                    datatype=args.data_type,
                                    transform=transform,
                                    start_frame=0,
                                    end_frame=-1)

        dataloader = DataLoader(dataset,
                                batch_size=1,
                                num_workers=0,
                                shuffle=False,
                                drop_last=True)

        tartanvo = TartanVO(args.VO_MODEL)
        for index, sample in ColoredTqdm(enumerate(dataloader),
                                         total=len(dataloader),
                                         desc=subset):
            with torch.no_grad():
                motion, _ = tartanvo.test_batch(sample)
                motion = cvtSE3_pypose(motion)
                motion_list.append(motion)

            if index % 1000 == 0:
                print(f'{index}/{len(dataloader)}')
                del tartanvo
                gc.collect()
                torch.cuda.empty_cache()
                tartanvo = TartanVO(args.VO_MODEL)

        motion_list = torch.cat(motion_list, dim=0)
        poses = motion2pose_pypose(motion_list)
        # poses = EDN2NED @ poses @ EDN2NED.Inv()
        poses_np = poses.detach().cpu().numpy()

        os.makedirs(Path(args.savepath), exist_ok=True)
        subset = subset.replace('/', '_')
        np.savetxt(Path(args.savepath) / f'{subset}.txt', poses_np)
        print(f'{path} done')

        del dataset, dataloader, tartanvo, motion_list, poses, poses_np
        gc.collect()
        torch.cuda.empty_cache()
