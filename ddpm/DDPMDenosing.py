from glob import glob

import numpy as np
import torch
from PIL import Image
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from tqdm import tqdm


class DDPMDenosing():
    def __init__(self):
        self.model_name = "google/ddpm-celebahq-256"
        self.time_step = 50
        self.scheduler = DDPMScheduler.from_pretrained(self.model_name)
        self.pipeline = DDPMPipeline.from_pretrained(self.model_name)
        self.save_dir = "enhanced/"

        if torch.has_mps:
            self.device = "mps"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = UNet2DModel.from_pretrained(self.model_name).to(self.device)
        self.pipeline = self.pipeline.to(self.device)
        self.scheduler.set_timesteps(self.time_step)

    def pred_to_iamge(self, pred, sample_image_size, image_name):
        image = (pred / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = Image.fromarray((image * 255).round().astype("uint8"))
        image = image.resize(sample_image_size, Image.LANCZOS)
        image.save(f"{self.save_dir}ddpm/{image_name.join(image_name.split('/')[-1].split('.')[:-1])}.png")

    def denosing(self, image_path):
        for img in tqdm(glob(image_path + "/*.png")):
            sample_image = Image.open(img).convert("RGB")
            sample_image_size = sample_image.size
            sample_size = min(sample_image_size) if min(sample_image_size) < 512 else 512
            # sample_size = model.config.sample_size
            sample_image = torch.Tensor(
                np.array(
                    sample_image
                    .resize(
                        (sample_size, sample_size),
                        Image.LANCZOS))) \
                               .permute(2, 0, 1).unsqueeze(0) \
                               .to(self.device) / 255

            input = sample_image

            for t in self.scheduler.timesteps:
                with torch.no_grad():
                    noisy_residual = self.model(input, t).sample
            previous_noisy_sample = self.scheduler.step(noisy_residual, t, input).prev_sample
            input = previous_noisy_sample
            self.pred_to_iamge(input, sample_image_size, img)
