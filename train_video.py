import math
import time
from pathlib import Path
import argparse
import yaml
import numpy as np
import torch
import sys
from PIL import Image
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from utils import *
from tqdm import tqdm
import random
import torchvision.transforms as transforms
from deform.deform_model import DeformModel
from random import randint
import re


class DefTrainer2d:
    """Train 2d gaussians to fit frames."""

    def __init__(
            self,
            image_root: Path,
            num_points: int = 2000,
            model_name: str = "GaussianImage_Cholesky",
            iterations: int = 50000,
            model_path=None,
            args=None,
    ):
        self.device = torch.device("cuda:0")
        self.image_root = image_root
        self.num_points = num_points
        self.image_length = 40
        BLOCK_H, BLOCK_W = 16, 16
        self.H, self.W = 800,800
        self.iterations = iterations
        self.save_imgs = args.save_imgs
        self.log_dir = Path(
            f"./checkpoints/{args.data_name}/{model_name}_{args.iterations}_{num_points}/")
        # 2DGS
        if model_name == "GaussianImage_Cholesky":
            from gaussianvideo_cholesky import GaussianVideo_Cholesky
            self.gaussian_model = GaussianVideo_Cholesky(loss_type="L2", opt_type="adan",
                                                         num_points=self.num_points, H=self.H, W=self.W,
                                                         BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
                                                         device=self.device, lr=args.lr, quantize=False).to(self.device)

        elif model_name == "GaussianImage_RS":
            from gaussianimage_rs import GaussianImage_RS
            self.gaussian_model = GaussianImage_RS(loss_type="L2", opt_type="adan", num_points=self.num_points,
                                                   H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
                                                   device=self.device, lr=args.lr, quantize=False).to(self.device)

        elif model_name == "3DGS":
            from gaussiansplatting_3d import Gaussian3D
            self.gaussian_model = Gaussian3D(loss_type="Fusion2", opt_type="adan", num_points=self.num_points, H=self.H,
                                             W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
                                             device=self.device, sh_degree=args.sh_degree, lr=args.lr).to(self.device)
        # deform-NN
        # 类比gaussian_renderer, 获取偏移量
        self.deform_model = DeformModel()
        self.deform_model.train_setting()

        self.logwriter = LogWriter(self.log_dir)

        # if model_path is not None:
        #     print(f"loading model path:{model_path}")
        #     checkpoint = torch.load(model_path, map_location=self.device)
        #     model_dict = self.gaussian_model.state_dict()
        #     pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        #     model_dict.update(pretrained_dict)
        #     self.gaussian_model.load_state_dict(model_dict)
        #     self.deform_model.load_weights(model_path, iteration=-1)

    def train(self):
        psnr_list, iter_list = [], []
        progress_bar = tqdm(range(1, self.iterations + 1), desc="Training progress")
        self.gaussian_model.train()
        # frame_stack
        warm_up = 10_000
        frames = list(range(self.image_length))
        random.shuffle(frames)
        frame_stack = frames.copy()
        # train all frames
        start_time = time.time()
        for iter in range(1, self.iterations + 1):
            if not frame_stack:
                frame_stack = frames.copy()
            idx = frame_stack.pop(randint(0, len(frame_stack) - 1))  # 随机弹出一张: r_idx
            image_path = self.image_root / f'r_{idx}.png'
            gt_image = image_path_to_tensor(image_path).to(self.device)
            # ---------------------------------------
            if iter < warm_up:
                d_xyz, d_cholesky = 0.0, 0.0
            else:
                # tmp = image_name.split(".")[0]  # r_23
                # t = int(tmp.split("_")[1])  # 23
                fid = torch.tensor(idx / (self.image_length-1))
                time_input = fid.unsqueeze(0).expand(self.gaussian_model.get_xyz.shape[0], -1).cuda()
                d_xyz, d_cholesky = self.deform_model.step(self.gaussian_model.get_xyz.detach(), time_input)
            # ---------------------------------------
            loss, psnr = self.gaussian_model.train_iter(gt_image, d_xyz, d_cholesky)
            psnr_list.append(psnr)
            iter_list.append(iter)
            with torch.no_grad():
                if iter % 10 == 0:
                    progress_bar.set_postfix({f"Loss": f"{loss.item():.{7}f}", "PSNR": f"{psnr:.{4}f},"})
                    progress_bar.update(10)
                    
        end_time = time.time() - start_time
        progress_bar.close()
        test_start_time = time.time()
        psnrs, ms_ssims = self.test()
        test_end_time = (time.time() - test_start_time) / self.image_length

        self.logwriter.write(
            "Training Complete in {:.4f}s, Eval time:{:.8f}s, FPS:{:.4f}".format(end_time, test_end_time,
                                                                                 1 / test_end_time))
        torch.save(self.gaussian_model.state_dict(), self.log_dir / "gaussian_model.pth.tar")
        self.deform_model.save_weights(self.log_dir, self.iterations)
        np.save(self.log_dir / "training.npy",
                {"iterations": iter_list, "training_psnr": psnr_list, "training_time": end_time,
                 "psnr": psnrs, "ms-ssim": ms_ssims, "rendering_time": test_end_time,
                 "rendering_fps": 1 / test_end_time})
        # ----------------------------
        avg_psnr = torch.tensor(psnrs).mean().item()
        avg_ms_ssim = torch.tensor(ms_ssims).mean().item()
        avg_training_time = end_time
        avg_eval_time = test_end_time
        avg_eval_fps = 1 / test_end_time
        # ----------------------------
        return avg_psnr, avg_ms_ssim, avg_training_time, avg_eval_time, avg_eval_fps

    def test(self):
        psnrs, ms_ssims = [], []
        self.gaussian_model.eval()
        for idx in range(0, self.image_length):
            with torch.no_grad():
                fid = torch.tensor(idx / (self.image_length - 1))
                time_input = fid.unsqueeze(0).expand(self.gaussian_model.get_xyz.shape[0], -1).cuda()  # [num_points,1]
                d_xyz, d_cholesky = self.deform_model.step(self.gaussian_model.get_xyz.detach(), time_input)
                out = self.gaussian_model(d_xyz=d_xyz, d_cholesky=d_cholesky)
                image_path = self.image_root / f'r_{idx}.png'
                gt_image = image_path_to_tensor(image_path).to(self.device)
                mse_loss = F.mse_loss(out["render"].float(), gt_image.float())
                psnr = 10 * math.log10(1.0 / mse_loss.item())
                ms_ssim_value = ms_ssim(out["render"].float(), gt_image.float(), data_range=1, size_average=True).item()
                self.logwriter.write("Test PSNR:{:.4f}, MS_SSIM:{:.6f}".format(psnr, ms_ssim_value))
                if self.save_imgs:
                    transform = transforms.ToPILImage()
                    img = transform(out["render"].float().squeeze(0))
                    name = f'r_{idx}' + "_fitting.png"
                    img.save(str(self.log_dir / name))
                psnrs.append(psnr)
                ms_ssims.append(ms_ssim_value)
        return psnrs, ms_ssims


def image_path_to_tensor(image_path: Path):
    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0)  # [1, C, H, W]
    return img_tensor


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, default='./datasets/kodak/', help="Training dataset"
    )
    parser.add_argument(
        "--data_name", type=str, default='kodak', help="Training dataset"
    )
    parser.add_argument(
        "--iterations", type=int, default=50000, help="number of training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--model_name", type=str, default="GaussianImage_Cholesky",
        help="model selection: GaussianImage_Cholesky, GaussianImage_RS, 3DGS"
    )
    parser.add_argument(
        "--sh_degree", type=int, default=3, help="SH degree (default: %(default)s)"
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=50000,
        help="2D GS points (default: %(default)s)",
    )
    parser.add_argument("--model_path", type=str, default=None, help="Path to a checkpoint")
    parser.add_argument("--seed", type=float, default=1, help="Set random seed for reproducibility")
    parser.add_argument("--save_imgs", action="store_true", help="Save image")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

    logwriter = LogWriter(Path(f"./checkpoints/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}"))
    psnrs, ms_ssims, training_times, eval_times, eval_fpses = [], [], [], [], []
    if args.data_name == "synthetic":  # test-set
        image_length, start = 40, 0

    # warm-up: r_0 for 10k
    # deform-NN: r0 ~r199 for 50k
    trainer = DefTrainer2d(image_root=Path(args.dataset), num_points=args.num_points,
                           iterations=args.iterations, model_name=args.model_name, args=args,
                           model_path=args.model_path) 
    avg_psnr, avg_ms_ssim, avg_training_time, avg_eval_time, avg_eval_fps = trainer.train()
    image_h, image_w = 800, 800
    logwriter.write("Frames: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(
        image_h, image_w, avg_psnr, avg_ms_ssim, avg_training_time, avg_eval_time, avg_eval_fps))

if __name__ == "__main__":
    main(sys.argv[1:])