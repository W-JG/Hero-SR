import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig
from transformers import AutoTokenizer, CLIPTextModel
from src.model import make_ddpm_sched, get_x0_from_noise, TimeAdaptive
from src.vaehook import VAEHook, perfcount


class HeroSR(torch.nn.Module):
    def __init__(self, args, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.device = device
        self.sd_model_name_or_path = args.sd_model_name_or_path
        self.pretrained_model_path = args.pretrained_model_path
        self.latent_tiled_size = args.latent_tiled_size
        self.latent_tiled_overlap = args.latent_tiled_overlap

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.sd_model_name_or_path, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.sd_model_name_or_path, subfolder="text_encoder"
        ).to(self.device)
        self.vae = AutoencoderKL.from_pretrained(
            self.sd_model_name_or_path, subfolder="vae"
        ).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(
            self.sd_model_name_or_path, subfolder="unet"
        ).to(self.device)
        self.text_encoder.requires_grad_(False)
        self.time_adaptive_model = TimeAdaptive().to(self.device)
        self.sched = make_ddpm_sched(self.sd_model_name_or_path)
        alphas_cumprod = self.sched.alphas_cumprod
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        if self.pretrained_model_path is not None:
            print(f"Initializing model from {self.pretrained_model_path}")
            sd = torch.load(self.pretrained_model_path, map_location='cpu')

            unet_lora_config = LoraConfig(
                r=sd["rank_unet"],
                init_lora_weights="gaussian",
                target_modules=sd["unet_lora_target_modules"],
            )
            self.unet.add_adapter(unet_lora_config)
            _sd_unet = self.unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            self.unet.load_state_dict(_sd_unet)

            vae_lora_config = LoraConfig(
                r=sd["rank_vae"],
                init_lora_weights="gaussian",
                target_modules=sd["vae_lora_target_modules"],
            )

            self.vae.add_adapter(vae_lora_config, adapter_name="vae")
            _sd_vae = self.vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            self.vae.load_state_dict(_sd_vae)
            self.time_adaptive_model.load_state_dict(
                sd["state_dict_time_adaptive"])

        self._init_tiled_vae(encoder_tile_size=args.vae_encoder_tiled_size,
                             decoder_tile_size=args.vae_decoder_tiled_size)

    @perfcount
    def forward(
        self,
        input_image,
        prompt=None,
    ):

        caption_tokens = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        caption_enc = self.text_encoder(caption_tokens).last_hidden_state
        self.timesteps = self.time_adaptive_model(input_image)

        latent_model_input = self.vae.encode(
            input_image).latent_dist.sample() * self.vae.config.scaling_factor

        _, _, h, w = latent_model_input.size()
        tile_size, tile_overlap = (
            self.latent_tiled_size, self.latent_tiled_overlap)
        if h * w <= tile_size * tile_size:
            print(f"[Tiled Latent]: the input size is tiny and unnecessary to tile.")
            model_pred = self.unet(
                latent_model_input, self.timesteps, encoder_hidden_states=caption_enc).sample
        else:
            print(
                f"[Tiled Latent]: the input size is {input_image.shape[-2]}x{input_image.shape[-1]}, need to tiled")
            tile_size = min(tile_size, min(h, w))
            tile_weights = self._gaussian_weights(
                tile_size, tile_size, 1).to(input_image.device)
            grid_rows = 0
            cur_x = 0
            while cur_x < latent_model_input.size(-1):
                cur_x = max(grid_rows * tile_size -
                            tile_overlap * grid_rows, 0)+tile_size
                grid_rows += 1

            grid_cols = 0
            cur_y = 0
            while cur_y < latent_model_input.size(-2):
                cur_y = max(grid_cols * tile_size -
                            tile_overlap * grid_cols, 0)+tile_size
                grid_cols += 1

            input_list = []
            noise_preds = []
            for row in range(grid_rows):
                noise_preds_row = []
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        ofs_x = max(row * tile_size-tile_overlap * row, 0)
                        ofs_y = max(col * tile_size-tile_overlap * col, 0)
                    if row == grid_rows-1:
                        ofs_x = w - tile_size
                    if col == grid_cols-1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    input_tile = latent_model_input[:, :,
                                                    input_start_y:input_end_y, input_start_x:input_end_x]
                    input_list.append(input_tile)

                    if len(input_list) == 1 or col == grid_cols-1:
                        input_list_t = torch.cat(input_list, dim=0)
                        model_pred = self.unet(
                            input_list_t, self.timesteps, encoder_hidden_states=caption_enc).sample
                        input_list = []
                    noise_preds.append(model_pred)

            noise_pred = torch.zeros(
                latent_model_input.shape, device=latent_model_input.device)
            contributors = torch.zeros(
                latent_model_input.shape, device=latent_model_input.device)
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        ofs_x = max(row * tile_size-tile_overlap * row, 0)
                        ofs_y = max(col * tile_size-tile_overlap * col, 0)
                    if row == grid_rows-1:
                        ofs_x = w - tile_size
                    if col == grid_cols-1:
                        ofs_y = h - tile_size
                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size
                    noise_pred[:, :, input_start_y:input_end_y,
                               input_start_x:input_end_x] += noise_preds[row*grid_cols + col] * tile_weights
                    contributors[:, :, input_start_y:input_end_y,
                                 input_start_x:input_end_x] += tile_weights
            noise_pred /= contributors
            model_pred = noise_pred

        x_denoised = get_x0_from_noise(latent_model_input.double(
        ), model_pred.double(), self.alphas_cumprod.double(), self.timesteps.long())
        output_image = (self.vae.decode(x_denoised.float() /
                        self.vae.config.scaling_factor).sample).clamp(-1, 1)
        return output_image

    def _init_tiled_vae(self,
                        encoder_tile_size=256,
                        decoder_tile_size=256,
                        fast_decoder=False,
                        fast_encoder=False,
                        color_fix=False,
                        vae_to_gpu=True):
        if not hasattr(self.vae.encoder, 'original_forward'):
            setattr(self.vae.encoder, 'original_forward',
                    self.vae.encoder.forward)
        if not hasattr(self.vae.decoder, 'original_forward'):
            setattr(self.vae.decoder, 'original_forward',
                    self.vae.decoder.forward)

        encoder = self.vae.encoder
        decoder = self.vae.decoder

        self.vae.encoder.forward = VAEHook(
            encoder, encoder_tile_size, is_decoder=False, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)
        self.vae.decoder.forward = VAEHook(
            decoder, decoder_tile_size, is_decoder=True, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generates a gaussian mask of weights for tile contributions"""
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width) /
                       (2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height) /
                       (2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights), (nbatches, self.unet.config.in_channels, 1, 1))
