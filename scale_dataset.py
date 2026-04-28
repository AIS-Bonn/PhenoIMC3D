import shutil
import os
import sys
from argparse import ArgumentParser
from PIL import Image
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = ArgumentParser(description="Convert and scale a dataset to be used for training with PhenoIMC3D.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--dataset_img_folder", type=str, default="images", help="Folder name within the dataset that contains the images.")
    parser.add_argument("--colmap_binary", type=str, required=True, help="Path to the COLMAP binary.")
    parser.add_argument("--sparse", type=str, default="sparse", help="Folder name within the dataset that contains the sparse COLMAP model.")
    parser.add_argument("--r", type=int, required=True, help="Scaling factor for the images.")
    return parser.parse_args()

def run_cmd(arg):
	print(f"RUNNING CMD: {arg}")
	err = os.system(arg)
	if err: sys.exit(err)

args = parse_args()
colmap_binary = args.colmap_binary
sparse = os.path.join(args.dataset_path, args.sparse)
text = os.path.join(args.dataset_path, "text")
images = os.path.join(args.dataset_path, args.dataset_img_folder)

new_dataset_path = f"{os.path.join(args.dataset_path, '')[:-1]}_{args.r}"
os.makedirs(new_dataset_path, exist_ok=True)
os.makedirs(os.path.join(new_dataset_path, args.dataset_img_folder), exist_ok=True)

img_list = os.listdir(os.path.join(args.dataset_path, args.dataset_img_folder))
progress_bar = tqdm(range(len(img_list)), desc="Image Downscale process")
for i in progress_bar:
    img = img_list[i]
    image = Image.open(os.path.join(args.dataset_path, args.dataset_img_folder, img))
    image_array = np.array(image)
    image = image.resize((image_array.shape[1] // int(args.r), image_array.shape[0] // int(args.r)))
    image.save(os.path.join(new_dataset_path, args.dataset_img_folder, img))

os.makedirs(text, exist_ok=True)

run_cmd(f"{colmap_binary} model_converter --input_path {sparse}/0 --output_path {text} --output_type TXT")

new_lines = []
with open(os.path.join(text, "cameras.txt"), 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith("#"): 
            new_lines.append(line.strip()) 
            continue
        parts = line.split()
        # scale width and height
        parts[2] = str(int(float(parts[2]) / float(args.r)))
        parts[3] = str(int(float(parts[3]) / float(args.r)))
        if parts[1] == "SIMPLE_RADIAL":
            parts[1] = "SIMPLE_PINHOLE" # For compatibility with Gaussian Splatting, we convert SIMPLE_RADIAL to SIMPLE_PINHOLE and ignore the radial distortion parameter.
            # scale focal length
            parts[4] = str(float(parts[4]) / float(args.r))

            # scale principal points
            parts[5] = str(float(parts[5]) / float(args.r))
            parts[6] = str(float(parts[6]) / float(args.r))
            del parts[7] # Remove radial distortion parameter
        elif parts[1] == "RADIAL":
            parts[1] = "PINHOLE" # For compatibility with Gaussian Splatting, we convert RADIAL to PINHOLE and ignore the radial distortion parameter.
            # scale focal length
            parts[4] = str(float(parts[4]) / float(args.r))
            parts[5] = str(float(parts[5]) / float(args.r))

            # scale principal points
            parts[6] = str(float(parts[6]) / float(args.r))
            parts[7] = str(float(parts[7]) / float(args.r))
            del parts[8] # Remove radial distortion parameter
        elif parts[1] == "PINHOLE":
            # scale focal length
            parts[4] = str(float(parts[4]) / float(args.r))
            parts[5] = str(float(parts[5]) / float(args.r))

            # scale principal points
            parts[6] = str(float(parts[6]) / float(args.r))
            parts[7] = str(float(parts[7]) / float(args.r))
        elif parts[1] == "SIMPLE_PINHOLE":
            # scale focal length
            parts[4] = str(float(parts[4]) / float(args.r))

            # scale principal points
            parts[5] = str(float(parts[5]) / float(args.r))
            parts[6] = str(float(parts[6]) / float(args.r))
        else:
            raise ValueError(f"Camera model {parts[1]} not supported.")

        new_lines.append(" ".join(parts))

with open(os.path.join(text, "cameras.txt"), 'w') as f:
    f.write("\n".join(new_lines))

os.makedirs(os.path.join(new_dataset_path, "text/0"), exist_ok=True)
shutil.copyfile(os.path.join(text, "cameras.txt"), os.path.join(new_dataset_path, "text/0/cameras.txt"))
shutil.copyfile(os.path.join(text, "images.txt"), os.path.join(new_dataset_path, "text/0/images.txt"))
shutil.copyfile(os.path.join(text, "points3D.txt"), os.path.join(new_dataset_path, "text/0/points3D.txt"))
os.makedirs(os.path.join(new_dataset_path, "sparse/0"), exist_ok=True)
run_cmd(f"{colmap_binary} model_converter --input_path {os.path.join(new_dataset_path, 'text/0')} --output_path {os.path.join(new_dataset_path, 'sparse/0')} --output_type BIN")

print(f"Dataset scaled and saved to {new_dataset_path}. You can now use this dataset for training with PhenoIMC3D.")