import torch
import pyiqa
import os
import csv
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse


class PairedDataset(Dataset):
    def __init__(
        self,
        gt_folders=None,
        hq_folders=None,
    ):
        super(PairedDataset, self).__init__()

        self.gt_list = [
            os.path.join(gt_folders, item) for item in os.listdir(gt_folders) if item.endswith(".png") or item.endswith(".jpg")
        ]
        self.hq_list = [
            os.path.join(hq_folders, item) for item in os.listdir(hq_folders) if item.endswith(".png") or item.endswith(".jpg")
        ]

        self.gt_list = sorted(self.gt_list)
        self.hq_list = sorted(self.hq_list)

        print("hq_len:{},gt_len:{}".format(
            len(self.hq_list), len(self.gt_list)))
        assert len(self.hq_list) == len(self.gt_list)

        self.img_preproc = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, index):

        gt_path = self.gt_list[index]
        gt_img = Image.open(gt_path).convert("RGB")
        gt_img = self.img_preproc(gt_img)

        hq_path = self.hq_list[index]
        hq_img = Image.open(hq_path).convert("RGB")
        hq_img = self.img_preproc(hq_img)

        return gt_img, hq_img

    def __len__(self):
        return len(self.gt_list)


def metric_image(hq_folder, gt_folder, metric_path, batch_size=1):
    metric_names = [
        'psnr_y',
        'ssim_y',
        "lpips",
        "musiq",
        'hyperiqa',
        'topiq_nr',
        'tres',
        # 'arniqa',
        # 'qalign',
    ]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")  # 使用GPU或CPU

    str_temp = ["name"] + metric_names
    with open(metric_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(str_temp)

    iqa_metric_dict = {}
    for name in metric_names:
        if name == 'psnr_y' or name == 'ssim_y':
            iqa_metric_dict[name] = pyiqa.create_metric(name.replace(
                '_y', ''), test_y_channel=True, color_space='ycbcr', device=device)
        else:
            iqa_metric_dict[name] = pyiqa.create_metric(name, device=device)

    dataset = PairedDataset(gt_folders=gt_folder, hq_folders=hq_folder)
    dataloader = DataLoader(dataset, num_workers=2,
                            batch_size=batch_size, shuffle=False)
    score_dict = {name: 0.0 for name in metric_names}
    for gt_img, hq_img in tqdm(dataloader, desc="Processing Images"):
        gt_img, hq_img = gt_img.float().to(device), hq_img.float().to(device)
        for metric in metric_names:
            score = iqa_metric_dict[metric](hq_img, gt_img)
            if score.ndimension() == 0:
                score_dict[metric] += score
            elif score.ndimension() == 1:
                for i in range(score.size(0)):
                    score_dict[metric] += score[i]
            elif score.ndimension() == 2:
                for i in range(score.size(0)):
                    score_dict[metric] += score[i, 0]
        result = [hq_folder] + [format(score_dict[metirc] / len(dataset), ".4f")
                                for metirc in metric_names]

    with open(metric_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hq_folder', type=str, default=None)
    parser.add_argument('--gt_folder', type=str, default=None)
    parser.add_argument('--metric_path', type=str, default='./metric.csv')
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    metric_image(hq_folder=args.hq_folder, gt_folder=args.gt_folder,
                 metric_path=args.metric_path, batch_size=args.batch_size)
