
from diffusers import DDPMScheduler
import torch.nn as nn
import torch.nn.functional as F
import torch


def make_ddpm_sched(pretrained_model_name_or_path=None):
    noise_scheduler = DDPMScheduler.from_pretrained(
        pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.cuda()
    return noise_scheduler


def get_x0_from_noise(sample, model_output, alphas_cumprod, timestep):
    alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
    beta_prod_t = 1 - alpha_prod_t
    pred_original_sample = (sample - beta_prod_t ** (0.5)
                            * model_output) / alpha_prod_t ** (0.5)
    return pred_original_sample


class TimeAdaptive(nn.Module):
    def __init__(self, time_list=[1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119, 129, 139, 149, 159, 169, 179, 189, 199, 209, 219, 229, 239, 249, 259, 269, 279, 289, 299, 309, 319, 329, 339, 349, 359, 369, 379, 389, 399, 409, 419, 429, 439, 449, 459, 469, 479, 489, 499, 509, 519, 529, 539, 549, 559, 569, 579, 589, 599, 609, 619, 629, 639, 649, 659, 669, 679, 689, 699, 709, 719, 729, 739, 749, 759, 769, 779, 789, 799, 809, 819, 829, 839, 849, 859, 869, 879, 889, 899, 909, 919, 929, 939, 949, 959, 969, 979, 989, 999],
                 channel=64, num_block=3, pool_output_size=(7, 7), tau=1):
        super(TimeAdaptive, self).__init__()

        self.register_buffer('time_list', torch.as_tensor(
            time_list, dtype=torch.float32))
        self.tau = tau
        time_output_len = len(time_list)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.SiLU(inplace=True)
        )
        self.blocks = self._make_blocks(channel, num_block)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.SiLU(inplace=True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(pool_output_size)
        self.fc = nn.Sequential(
            nn.Linear(
                channel * pool_output_size[0] * pool_output_size[1], 1000),
            nn.SiLU(inplace=True),
            nn.Linear(1000, time_output_len),
            nn.LayerNorm(len(time_list)),
        )

    def _make_blocks(self, channel, num_block):
        blocks = []
        for _ in range(num_block):
            blocks.append(ResNetBlock(in_channels=channel,
                          out_channels=channel, stride=1))
        return nn.Sequential(*blocks)

    def forward(self, x):

        x = F.interpolate(x, size=(128, 128))

        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        z = F.gumbel_softmax(x, tau=self.tau, hard=True)
        result = (self.time_list * z).sum(dim=1)
        return result


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.SiLU = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.SiLU(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.SiLU(out)
        return out
