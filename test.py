import argparse
import os
import glob
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from diffusers.utils.import_utils import is_xformers_available
from src.hero_sr import HeroSR
from src.wavelet_color_fix import adain_color_fix, wavelet_color_fix


def parse_args():
    parser = argparse.ArgumentParser(description="Hero-SR Evaluation Script")
    parser.add_argument('--sd_model_name_or_path', type=str, default='',
                        help='Path to the SD model')
    parser.add_argument('--pretrained_model_path', type=str, default='',
                        help='Path to the pretrained model')
    parser.add_argument('--input_dir', type=str, default='./figure',
                        help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, default='./result',
                        help='Directory to save processed images')
    parser.add_argument('--upscale', type=int, default=4,
                        help='upscale factor')
    parser.add_argument('--align_method', type=str, default='adain',
                        choices=['wavelet', 'adain', 'nofix'],
                        help='Color adjustment method')
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=96)
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--enable_xformers_memory_efficient_attention",
                        action="store_true", default=False)
    return parser.parse_args()


def process_image(model, input_image, args):
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    ori_width, ori_height = input_image.size
    rscale = args.upscale
    resize_flag = False
    if ori_width < args.process_size//rscale or ori_height < args.process_size//rscale:
        scale = (args.process_size//rscale)/min(ori_width, ori_height)
        input_image = input_image.resize((int(scale*ori_width), int(scale*ori_height)), Image.LANCZOS)
        resize_flag = True
    input_image = input_image.resize((input_image.size[0]*rscale, input_image.size[1]*rscale))
    new_width = input_image.width - input_image.width % 8
    new_height = input_image.height - input_image.height % 8
    input_image = input_image.resize((new_width, new_height), Image.LANCZOS)

    x = to_tensor(input_image).unsqueeze(0).to(
        'cuda' if torch.cuda.is_available() else 'cpu')
    x = x * 2 - 1
    with torch.no_grad():
        output_image = model(x, prompt=" ",)
    output_image = (output_image[0].cpu() * 0.5 + 0.5).clamp(0, 1)
    output_pil = to_pil(output_image)
    if args.align_method == 'wavelet':
        output_pil = wavelet_color_fix(output_pil, input_image)
    elif args.align_method == 'adain':
        output_pil = adain_color_fix(output_pil, input_image)
    elif args.align_method == 'nofix':
        pass
    if resize_flag:
        output_pil.resize((int(args.upscale*ori_width),int(args.upscale*ori_height)))
    return output_pil


def main(args):
    model = HeroSR(args=args)
    model.eval()
    model.requires_grad_(False)
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            model.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")
    img_paths = sorted(glob.glob(os.path.join(
        args.input_dir, "*.png")) + glob.glob(os.path.join(args.input_dir, "*.jpg")))
    for img_path in tqdm(img_paths):
        input_image = Image.open(img_path).convert('RGB')
        output_pil = process_image(model, input_image, args)
        bname = os.path.basename(img_path).replace('.jpg', '.png')
        save_path = os.path.join(args.output_dir, bname)
        output_pil.save(save_path)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
