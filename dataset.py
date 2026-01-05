import os, glob, shutil, random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import config
import torchvision.transforms.functional as TF
from PIL import Image

def smart_prepare_data(zip_root, extract_root):
    os.makedirs(extract_root, exist_ok=True)
    
    zip_files = sorted(glob.glob(os.path.join(zip_root, "*.zip")))

    if len(zip_files) == 0:
        print("ERROR: No zip files found.")
        return {}

    video_path_map = {}

    for zip_path in tqdm(zip_files, desc="Smart Sync"):
        zip_filename = os.path.basename(zip_path)
        video_name = os.path.splitext(zip_filename)[0]

        if video_name not in config.VIDEO_CONFIG:
            continue

        target_dir = os.path.join(extract_root, video_name)
        video_path_map[video_name] = target_dir

        if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 5:
            continue

        os.makedirs(target_dir, exist_ok=True)
        os.system(f"unzip -q -o \"{zip_path}\" -d \"{target_dir}\"")

        nested = os.path.join(target_dir, video_name)
        if os.path.exists(nested):
            for f in glob.glob(os.path.join(nested, "*")):
                shutil.move(f, target_dir)
            os.rmdir(nested)

    print(f"{len(video_path_map)} videos ready.")
    return video_path_map

def preprocess_images(im1, im2):
    im1_t = im1.astype(np.float32)/255.0
    im2_t = im2.astype(np.float32)/255.0
    im1_t = torch.from_numpy(im1_t).permute(2, 0, 1)
    im2_t = torch.from_numpy(im2_t).permute(2, 0, 1)
    return im1_t, im2_t

class InMemoryVisualDynamicsDataset(Dataset):
    def __init__(self, video_map, config_dict, augment=False, split_mode='all', split_ratio=0.8):
        self.pairs = []
        self.image_cache = {}
        self.augment = augment
        
        self.jitter_params = {
            'brightness': 0.4,
            'contrast': 0.4,
            'saturation': 0.5,
            'hue': 0.1
        }

        files_to_load = []
        video_files_dict = {}

        for video_name, folder_path in video_map.items():
            if video_name not in config_dict: continue

            files = sorted(glob.glob(os.path.join(folder_path, '*.png')))
            total = len(files)
            if total < 2: continue

            split_idx = int(total * split_ratio)
            target_files = files[:split_idx] if split_mode == 'train' else files[split_idx:]

            video_files_dict[video_name] = target_files
            files_to_load.extend(target_files)

        def load_worker(path):
            img = cv2.imread(path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
                return path, img
            return path, None

        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(tqdm(executor.map(load_worker, files_to_load), total=len(files_to_load), desc="RAM Load", leave=False))

        for path, img in results:
            if img is not None: self.image_cache[path] = img

        for video_name, files in video_files_dict.items():
            settings = config_dict[video_name]
            gaps = settings['gaps']
            min_th = settings['min_th']

            valid_files = [f for f in files if f in self.image_cache]

            for gap in gaps:
                for i in range(len(valid_files) - gap):
                    p1 = valid_files[i]
                    p2 = valid_files[i+gap]
                    
                    im1 = self.image_cache[p1]
                    im2 = self.image_cache[p2]

                    diff = im2.astype(np.float32) - im1.astype(np.float32)
                    mse = np.mean(diff ** 2)

                    if mse < min_th: continue
                    if mse > config.GLOBAL_MAX_THRESH: continue

                    self.pairs.append((p1, p2))

        print(f"Total {len(self.pairs)} pairs created.")

    def apply_sync_augmentations(self, img1, img2):
        pil1 = Image.fromarray(img1)
        pil2 = Image.fromarray(img2)

        brightness_factor = random.uniform(1 - self.jitter_params['brightness'], 1 + self.jitter_params['brightness'])
        contrast_factor = random.uniform(1 - self.jitter_params['contrast'], 1 + self.jitter_params['contrast'])
        saturation_factor = random.uniform(1 - self.jitter_params['saturation'], 1 + self.jitter_params['saturation'])
        hue_factor = random.uniform(-self.jitter_params['hue'], self.jitter_params['hue'])
        
        do_flip = random.random() > 0.5
        do_grayscale = random.random() < 0.1

        pil1 = TF.adjust_brightness(pil1, brightness_factor)
        pil2 = TF.adjust_brightness(pil2, brightness_factor)
        
        pil1 = TF.adjust_contrast(pil1, contrast_factor)
        pil2 = TF.adjust_contrast(pil2, contrast_factor)
        
        pil1 = TF.adjust_saturation(pil1, saturation_factor)
        pil2 = TF.adjust_saturation(pil2, saturation_factor)
        
        pil1 = TF.adjust_hue(pil1, hue_factor)
        pil2 = TF.adjust_hue(pil2, hue_factor)

        if do_grayscale:
            pil1 = TF.to_grayscale(pil1, num_output_channels=3)
            pil2 = TF.to_grayscale(pil2, num_output_channels=3)

        if do_flip:
            pil1 = TF.hflip(pil1)
            pil2 = TF.hflip(pil2)

        return np.array(pil1), np.array(pil2)

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2 = self.pairs[idx]
        im1, im2 = self.image_cache[p1], self.image_cache[p2]

        if self.augment:
            im1, im2 = self.apply_sync_augmentations(im1, im2)

        i1_t, i2_t = preprocess_images(im1, im2)
        v = (i2_t - i1_t) * config.SIGNAL_GAIN
        return {'im1': i1_t, 'im2': i2_t, 'v': v}