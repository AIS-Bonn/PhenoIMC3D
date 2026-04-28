import glob
import json
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
import os
from argparse import ArgumentParser, Namespace
import shutil
import sys

from tqdm import tqdm
from natsort import natsorted

_original_utils = sys.modules.get('utils')
if 'utils' in sys.modules:
    del sys.modules['utils']
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'submodules/RAFT/core')))
from submodules.RAFT.core.raft import RAFT
from submodules.RAFT.demo import load_image
from submodules.RAFT.core.utils import flow_viz
# Restore original utils
if _original_utils:
    sys.modules['utils'] = _original_utils


def parse_args():
    parser = ArgumentParser(description="Train a model for 3D reconstruction for Iterative Motion Compensation.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to save the trained model.")
    parser.add_argument("--imc_iterations", type=int, default=30, help="Number of iterations for IMC training.")
    parser.add_argument("--dataset_img_folder", type=str, default="images", help="Folder name within the dataset that contains the images.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    return parser.parse_args()

def run_cmd(arg):
	print(f"RUNNING CMD: {arg}")
	err = os.system(arg)
	if err: sys.exit(err)

def main():
    args = parse_args()
    config = json.load(open(args.config, 'r'))

    # Copy Dataset so images can be motion compensated
    shutil.rmtree(os.path.join(args.model_path, "dataset"), ignore_errors=True) # Delete old dataset if it exists
    shutil.copytree(args.dataset_path, os.path.join(args.model_path, "dataset"), dirs_exist_ok=True)

    # Load RAFT
    raft_args = Namespace()
    raft_args.model = "submodules/RAFT/models/raft-things.pth"
    raft_args.small = False
    raft_args.mixed_precision = False
    raft_args.alternate_corr = False
    model = torch.nn.DataParallel(RAFT(raft_args))
    model.load_state_dict(torch.load(raft_args.model))
    model = model.module
    model.cuda()
    model.eval()
    
    for i in range(args.imc_iterations):
        print(f"\nOptical Flow Motion Compensation Iteration {i} / {args.imc_iterations}")

        log_path = os.path.join(args.model_path, f"iter_{i}")
        os.makedirs(log_path, exist_ok=True)

        train_cmd = config["train_cmd"].replace("${dataset_path}", os.path.join(args.model_path, "dataset")).replace("${model_path}", args.model_path)
        run_cmd(train_cmd)
        if not config["model_weights"] == None: shutil.copy(config["model_weights"].replace("${model_path}", args.model_path), os.path.join(log_path))

        img_predictions = config["img_predictions"].replace("${model_path}", args.model_path)
        shutil.rmtree(os.path.join(img_predictions), ignore_errors=True)
        render_cmd = config["render_cmd"].replace("${dataset_path}", os.path.join(args.model_path, "dataset")).replace("${model_path}", args.model_path)
        run_cmd(render_cmd)

        files = natsorted(next(os.walk(img_predictions), (None, None, []))[2])
        logimg1 = files[0]
        logimg2 = files[int(len(files)/3)]
        shutil.copyfile(os.path.join(img_predictions, logimg1), os.path.join(log_path, logimg1))
        shutil.copyfile(os.path.join(img_predictions, logimg2), os.path.join(log_path, logimg2))

        shutil.rmtree(os.path.join(args.model_path, "deformed"), ignore_errors=True)
        shutil.rmtree(os.path.join(args.model_path, "flow"), ignore_errors=True)
        with torch.no_grad():
            images = glob.glob(os.path.join(args.dataset_path, args.dataset_img_folder, '*.png')) + \
                glob.glob(os.path.join(args.dataset_path, args.dataset_img_folder, '*.PNG'))+ \
                glob.glob(os.path.join(args.dataset_path, args.dataset_img_folder, '*.jpg'))+ \
                glob.glob(os.path.join(args.dataset_path, args.dataset_img_folder, '*.JPG'))
            images = natsorted(images)

            progress_bar = tqdm(range(len(images)), desc="RAFT progress")
            for i in progress_bar:
                gt = images[i]

                pred = os.path.join(img_predictions, files[i])
                gt_img = load_image(gt)
                pred_img = load_image(pred)
                progress_bar.set_description(f"{pred.split('/')[-1]}->{gt.split('/')[-1]}")
                size = (pred_img.shape[2], pred_img.shape[3])
                gt_img = F.interpolate(gt_img[:,:3,:,:], size=size, mode='bilinear')

                _, flow_up = model(F.interpolate(pred_img[:,:3,:,:], size=size, mode='bilinear'), F.interpolate(gt_img[:,:3,:,:], size=size, mode='bilinear'), iters=20, test_mode=True)
                flow_up = F.interpolate(flow_up, size=(gt_img.shape[2], gt_img.shape[3]), mode='bilinear')
                flow_up *= (gt_img.shape[3] / pred_img.shape[3])
            
                curr = gt_img.squeeze().permute(1,2,0).cpu().numpy()
                flo = flow_up.squeeze().permute(1,2,0).cpu().numpy()
                
                h, w = flo.shape[:2]
                flow = flo.copy()
                flow[:,:,0] += np.arange(w)
                flow[:,:,1] += np.arange(h)[:,np.newaxis]

                mask_w = (flow[:,:,0] < 0) | (flow[:,:,0] >= w - 1)
                mask_h = (flow[:,:,1] < 0) | (flow[:,:,1] >= h - 1)

                flow[:,:,0] = flow[:,:,0] * ~mask_w + np.arange(w) * mask_w
                flow[:,:,1] = flow[:,:,1] * ~mask_h + np.arange(h)[:,np.newaxis] * mask_h
                prevImg = np.asarray(cv2.remap(curr, flow, None, cv2.INTER_LINEAR))

                img = Image.fromarray(prevImg.astype(np.uint8))
                os.makedirs(os.path.join(args.model_path, "deformed"), exist_ok=True) 
                img.save(os.path.join(args.model_path, "deformed", gt.split("/")[-1]), quality=100)

                flo = flow_viz.flow_to_image(flo)
                img = Image.fromarray(flo.astype(np.uint8))
                os.makedirs(os.path.join(args.model_path, "flow"), exist_ok=True) 
                img.save(os.path.join(args.model_path, "flow", gt.split("/")[-1]), quality=100)


        files = natsorted(next(os.walk(os.path.join(args.model_path, "deformed")), (None, None, []))[2])
        logimg1 = files[0]
        logimg2 = files[int(len(files)/3)]

        shutil.copyfile(os.path.join(args.model_path, "deformed", logimg1), os.path.join(log_path, logimg1 + "_deformed.png"))
        shutil.copyfile(os.path.join(args.model_path, "deformed", logimg2), os.path.join(log_path, logimg2 + "_deformed.png"))
        shutil.copyfile(os.path.join(args.model_path, "flow", logimg1), os.path.join(log_path, logimg1 + "_flow.png"))
        shutil.copyfile(os.path.join(args.model_path, "flow", logimg2), os.path.join(log_path, logimg2 + "_flow.png"))
        shutil.rmtree(os.path.join(args.model_path, "dataset", args.dataset_img_folder))
        shutil.copytree(os.path.join(args.model_path, "deformed"), os.path.join(args.model_path, "dataset", args.dataset_img_folder))

if __name__ == "__main__":
    main()