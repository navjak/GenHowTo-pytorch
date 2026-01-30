import os
import torch
import numpy as np
from PIL import Image
from diffusers import DDIMScheduler
import torchvision.transforms as transforms

from model import GenHowToModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_path):
    model = GenHowToModel()
    model.to(DEVICE) 

    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    print("Model loaded")
    
    del state_dict
    torch.cuda.empty_cache()

    model.eval()
    return model


def generate(model, img_path, prompt, steps=50, skip=2):
    scheduler = DDIMScheduler.from_pretrained("Manojb/stable-diffusion-2-1-base", subfolder="scheduler")
    scheduler.set_timesteps(steps)
    
    # preprocessing/transforms
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    src_tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    # encode src img -> z_src
    with torch.no_grad():
        z_src = model.vae.encode(src_tensor).latent_dist.sample() * model.vae.config.scaling_factor

    # start from noisy z_src instead of pure random noise as mentioned in paper
    # because we want to preserve structure and bg details of src image

    timesteps = scheduler.timesteps
    t_start = timesteps[skip]
    
    epsilon = torch.randn_like(z_src)
    
    # add noise
    z_t = scheduler.add_noise(z_src, epsilon, torch.tensor([t_start]))
    
    print(f"Starting Inference from step {skip}/{steps} (t={t_start.item()})...")
    
    # Sampling loop
    for i, t in enumerate(timesteps[skip:]):
        # expand for batch if needed (here batch=1)
        t_batch = torch.tensor([t], device=DEVICE)
        
        with torch.no_grad():
            pred_noise = model.predict_noise(z_src, z_t, [prompt], t_batch)
            
            step_output = scheduler.step(pred_noise, t, z_t) # step
            z_t = step_output.prev_sample

    # VAE decode
    with torch.no_grad():
        img_out = model.vae.decode(z_t / model.vae.config.scaling_factor).sample
        
    # post processing
    img_out = (img_out / 2 + 0.5).clamp(0, 1)
    img_out = img_out.cpu().permute(0, 2, 3, 1).numpy()[0]
    img_out = (img_out * 255).astype(np.uint8)
    
    return Image.fromarray(img_out)

def main():
    model_path = "./checkpoints/GHT_epoch_30.pth"
    test_img = "./data_dir/i_init/sample1.jpeg"
    
    try:
        model = load_model(model_path)
    except FileNotFoundError:
        print(f"ERROR - Model weights file not found.")
        return

    # prompt for image
    result = generate(model, test_img, "two apple halves on a cutting board")
    result.save("gen_img.png")
    print("Saved generated image")

if __name__ == "__main__":
    main()