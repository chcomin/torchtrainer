"""Benchmark different stages in a training pipeline to identify bottlenecks."""

import logging
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2 as tv_transf
from torchvision.transforms.v2 import functional as tv_transf_F

logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
DEVICE = "cuda"
torch.backends.cudnn.benchmark = True 

class SyntheticSegmentationDataset(Dataset):
    """A synthetic dataset that generates random images on disk for segmentation tasks."""

    def __init__(self, root_dir, num_imgs, img_size, transforms=None, ext="png"):
        root_dir = Path(root_dir)
        self.root_dir = root_dir
        self.num_imgs = num_imgs
        self.img_size = img_size
        self.transforms = transforms
        self.ext = ext
        self.imgs_dir = root_dir / "images"
        self.targets_dir = root_dir / "labels"
        self.file_list = self.setup_synthetic_data()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        img = Image.open(self.imgs_dir / filename).convert("RGB")
        target = Image.open(self.targets_dir / filename)

        if self.transforms:
            img_tensor, target_tensor = self.transforms(img, target)

        return img_tensor, target_tensor
    
    def setup_synthetic_data(self):
        """Generates a synthetic dataset on disk."""
        root_dir = self.root_dir
        imgs_dir = self.imgs_dir
        targets_dir = self.targets_dir
        img_size = self.img_size
        num_imgs = self.num_imgs

        if root_dir.exists():
            shutil.rmtree(root_dir)
        os.makedirs(imgs_dir)
        os.makedirs(targets_dir)
        logger.info(f"--- Generating {num_imgs} images ({img_size}x{img_size}) in '{root_dir}' ---")
        
        # Create random base image and target
        base_img = np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)
        base_target = np.random.randint(0, 2, (img_size, img_size), dtype=np.uint8)
        img_pil = Image.fromarray(base_img)
        target_pil = Image.fromarray(base_target)
        
        file_list = []
        for i in range(num_imgs):
            filename = f"img_{i:05d}.{self.ext}"
            img_pil.save(self.imgs_dir / f"{filename}")
            target_pil.save(self.targets_dir / f"{filename}")
            
            file_list.append(filename)
            
        return file_list

def measure_throughput(start_time, num_imgs, img_size, bytes_per_pixel=4):
    """Measures throughput given start time and number of images processed.
    bytes_per_pixel represents the number of bytes of each pixel.
    """
    total_time = time.time() - start_time
    if total_time == 0:
        return 0, 0, 0
    ips = num_imgs / total_time

    # Calculation in GB/s
    # Bytes = I * H * W * bytes_per_pixel
    total_bytes = num_imgs * img_size * img_size * bytes_per_pixel
    gb_per_sec = (total_bytes / 1e9) / total_time

    return ips, gb_per_sec

def run_benchmark(
    model,
    img_size=256,
    batch_size=8,
    num_workers=4,
    transforms=None,
    num_batches=200,
    num_imgs_dataset=2000,
    pin_memory=True,
    non_blocking=True,
    use_amp=True, 
    which="dpgf",
    root_dir="./_fake_data_bench",
    warmup_batches = 20,
    verbose=True
):
    """
    Measure throughput of different stages in a training pipeline.
    which can be "dpgf" (data, pcie, gpu, full) or any subset like "dp", "g", etc.
    """
    
    if verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    num_imgs_benchmark = num_batches * batch_size

    # Setup Dataset
    dataset = SyntheticSegmentationDataset(
        root_dir, num_imgs_dataset, img_size, transforms=transforms)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=(num_workers > 0)
    )
    
    # Model and Optimizer
    model.train()
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    scaler = GradScaler(enabled=use_amp) # Scaler for AMP

    # ==============================================================================
    # 1. DATA LOADING (Disk -> CPU isolated)
    # ==============================================================================
    if "d" in which:
        logger.info("\n[1] Benchmarking Dataloader (Disk -> CPU)...")
        iter_loader = iter(dataloader)
        
        # Warmup
        for _ in range(warmup_batches): 
            try:
                next(iter_loader) 
            except StopIteration:
                # Restart the dataloader iterator if needed
                iter_loader = iter(dataloader)
                next(iter_loader)
                
        # Measure
        start = time.time()
        for _ in range(num_batches):
            try:
                next(iter_loader)
            except StopIteration:
                iter_loader = iter(dataloader)
                next(iter_loader)
        
        ips_data, gb_per_sec_data = measure_throughput(
            start, 
            num_imgs_benchmark, 
            img_size, 
            bytes_per_pixel=4) # 3 bytes for input + 1 byte for target
        logger.info(f"    -> Throughput: {ips_data:.2f} images/s | {gb_per_sec_data:.2f} GB/s")
    else:
        ips_data = float("inf")

    # ==============================================================================
    # 2. PCIe TRANSFER (RAM -> VRAM isolated)
    # ==============================================================================
    if "p" in which:
        logger.info("\n[2] Benchmarking PCIe Bandwidth (RAM -> VRAM)...")
    
        # Create heavy tensors in RAM
        cpu_tensor = torch.randn(batch_size, 3, img_size, img_size, dtype=torch.float32)
        cpu_target = torch.randint(0, 2, (batch_size, img_size, img_size), dtype=torch.long)
        if pin_memory:
            cpu_tensor = cpu_tensor.pin_memory()
            cpu_target = cpu_target.pin_memory()
        
        # Warmup
        for _ in range(warmup_batches):
            _ = cpu_tensor.to(DEVICE, non_blocking=non_blocking)
            _ = cpu_target.to(DEVICE, non_blocking=non_blocking)
        torch.cuda.synchronize()

        # Measure
        start = time.time()
        for _ in range(num_batches):
            _ = cpu_tensor.to(DEVICE, non_blocking=non_blocking)
            _ = cpu_target.to(DEVICE, non_blocking=non_blocking)
            # Synchronize at every batch in this specific test to measure pure transfer latency
            torch.cuda.synchronize() 
            
        ips_pci, gb_per_sec_pci = measure_throughput(
            start, 
            num_imgs_benchmark, 
            img_size, 
            bytes_per_pixel=12+8) # 12 bytes for input + 8 bytes for target
        
        logger.info(f"    -> Throughput: {ips_pci:.2f} images/s | {gb_per_sec_pci:.2f} GB/s")
    else:
        ips_pci = float("inf")

    # ==============================================================================
    # 3. GPU COMPUTE (Compute isolated with AMP)
    # ==============================================================================
    if "g" in which:
        logger.info("\n[3] Benchmarking GPU Compute...")
        
        # Generate fake batch directly on GPU
        fake_input = torch.randn(batch_size, 3, img_size, img_size).to(DEVICE)
        fake_target = torch.randint(0, 2, (batch_size, img_size, img_size), 
                                    dtype=torch.long).to(DEVICE)
        
        # Warmup
        for _ in range(warmup_batches):
            optimizer.zero_grad()
            with autocast(device_type="cuda", enabled=use_amp):
                out = model(fake_input)["out"]
                loss = criterion(out, fake_target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        torch.cuda.synchronize()

        # Measure
        start = time.time()
        for _ in range(num_batches):
            optimizer.zero_grad()
            with autocast(device_type="cuda", enabled=use_amp):
                out = model(fake_input)["out"]
                loss = criterion(out, fake_target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        torch.cuda.synchronize()
        ips_gpu, _ = measure_throughput(
            start, 
            num_imgs_benchmark, 
            img_size, 
            bytes_per_pixel=12) # 12 bytes for input + 0 bytes for target 
        logger.info(f"    -> Throughput: {ips_gpu:.2f} images/s")
    else:
        ips_gpu = float("inf")

    # ==============================================================================
    # 4. FULL PIPELINE (End-to-End)
    # ==============================================================================
    if "f" in which:
        logger.info("\n[4] Benchmarking Full Pipeline (End-to-End)...")
        
        iter_loader = iter(dataloader)
        
        # Warmup
        for _ in range(warmup_batches):
            try: 
                img, target = next(iter_loader)
            except StopIteration: 
                iter_loader = iter(dataloader)
                img, target = next(iter_loader)
            img = img.to(DEVICE, non_blocking=non_blocking)
            target = target.to(DEVICE, non_blocking=non_blocking)
            optimizer.zero_grad()
            with autocast(device_type="cuda", enabled=use_amp):
                out = model(img)["out"]
                loss = criterion(out, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        torch.cuda.synchronize()

        # Measure
        start = time.time()
        for _ in range(num_batches):
            try: 
                img, target = next(iter_loader)
            except StopIteration: 
                iter_loader = iter(dataloader)
                img, target = next(iter_loader)
            
            img = img.to(DEVICE, non_blocking=non_blocking)
            target = target.to(DEVICE, non_blocking=non_blocking)
            
            optimizer.zero_grad()
            with autocast(device_type="cuda", enabled=use_amp):
                output = model(img)["out"]
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        torch.cuda.synchronize()
        ips_full, _ = measure_throughput(
            start, num_imgs_benchmark, img_size, bytes_per_pixel=12+8)
        logger.info(f"    -> Throughput: {ips_full:.2f} images/s")
    else:
        ips_full = float("inf")

    # ==============================================================================
    # ANALYSIS
    # ==============================================================================
    logger.info("\n--- BOTTLENECK DIAGNOSIS ---")
    stages = {
        "1. Data Loading (CPU)": ips_data,
        "2. Transfer (PCIe)": ips_pci,
        "3. Compute (GPU)": ips_gpu
    }
    
    # Identify the slowest stage
    slowest_stage_name = min(stages, key=stages.get)
    max_theoretical_speed = stages[slowest_stage_name]
    efficiency = (ips_full / max_theoretical_speed) * 100
    
    logger.info("Isolated Speeds (images/s):")
    for name, val in stages.items():
        logger.info(f"  {name}: {val:.2f}")
    
    logger.info(f"\nReal Speed: {ips_full:.2f} images/s")
    logger.info(f"Pipeline Efficiency: {efficiency:.1f}% (relative to the weakest link)")
    logger.info(f"PRIMARY BOTTLENECK: >> {slowest_stage_name} <<")

    # Cleanup
    shutil.rmtree(root_dir)

    results = {
            "ips_data": ips_data,
            "ips_pci": ips_pci,
            "ips_gpu": ips_gpu,
            "ips_full": ips_full,
            "bottleneck": slowest_stage_name
        }
    
    return results

class TrainTransforms:
    """Example transformations."""

    def __init__(self, img_size=256, resize_target = True):

        self.img_size = img_size
        self.resize_target = resize_target

        scale = tv_transf.RandomAffine(degrees=0, scale=(0.95, 1.20))
        transl = tv_transf.RandomAffine(degrees=0, translate=(0.05, 0))
        rotate = tv_transf.RandomRotation(degrees=45)
        scale_transl_rot = tv_transf.RandomChoice((scale, transl, rotate))
        brightness, contrast, saturation, hue = 0.25, 0.25, 0.25, 0.01
        jitter = tv_transf.ColorJitter(brightness, contrast, saturation, hue)
        hflip = tv_transf.RandomHorizontalFlip()
        vflip = tv_transf.RandomVerticalFlip()

        self.transform = tv_transf.Compose((
            scale_transl_rot,
            jitter,
            hflip,
            vflip,
        ))

    def __call__(self, img, target):

        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).to(dtype=torch.uint8)
        target = torch.from_numpy(np.array(target)).unsqueeze(0).to(dtype=torch.uint8)

        img = tv_transf_F.resize(img, self.img_size)
        if self.resize_target:
            target = tv_transf_F.resize(target, 
                                        self.img_size, 
                                        interpolation=tv_transf.InterpolationMode.NEAREST_EXACT)

        img = tv_tensors.Image(img)
        target = tv_tensors.Mask(target)

        img, target = self.transform(img, target)

        img = img.data.float()/255
        target = target.data.to(dtype=torch.int64)[0]

        return img, target

if __name__ == "__main__":
    from torchvision.models.segmentation import deeplabv3_resnet50

    
    model = deeplabv3_resnet50(num_classes=2).to(DEVICE)

    # Run the command below in the terminal while the GPU is computing:
    # nvidia-smi dmon -s u -d 1
    # sm: percentage of streaming multiprocessor utilization (this is not the same as GPU 
    #     utilization provided by nvidia-smi)
    # mem: memory controller utilization. Higher values indicate a bottleneck in memory speed.
    run_benchmark(
        model=model,
        img_size=256, 
        batch_size=8,      
        num_workers=4,
        transforms=TrainTransforms(img_size=256),
        num_batches=200,
        num_imgs_dataset=2000,
        pin_memory=True,
        non_blocking=False,
        use_amp=True,
    )