import os
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from accelerate import Accelerator
import bitsandbytes as bnb

from data import GHTDataset
from model import GenHowToModel


# config details set for training on one sample
with open('config.json') as f:
    config = json.load(f)

DATA_ROOT = config["DATA_ROOT"]
OUTPUT_DIR = config["OUTPUT_DIR"]
LR = config["LR"]
EPOCHS = config["EPOCHS"]
# DEVICE = config["DEVICE"]
STEPS_PER_LOG = config["STEPS_PER_LOG"]
TRAIN_FINAL_STATE_MODEL = config["TRAIN_FINAL_STATE_MODEL"] # if true, final state model is trained


def main():
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device 
  
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Device: {device}")
    print(f"Root: {DATA_ROOT}")
    print(f"LR: {LR}")
    print(f"Epochs: {EPOCHS}")

    # handle data and create dataloader
    dataset = GHTDataset(root=DATA_ROOT, train_final_state_model=TRAIN_FINAL_STATE_MODEL)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"Dataset size: {len(dataset)}")

    # calling model
    model = GenHowToModel()
    model.train()

    # opt
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # opt = optim.AdamW(trainable_params, lr=LR)
    opt = bnb.optim.AdamW8bit(trainable_params, lr=LR)

    model, opt, loader = accelerator.prepare(model, opt, loader)

    # Training loop
    global_step = 0
    num_timesteps = 1000 

    for epoch in range(EPOCHS):
        if accelerator.is_main_process:
            print(f"\n Epoch: {epoch+1}/{EPOCHS}")

        for batch in loader:
            src = batch['src']
            tgt = batch['tgt']
            txt = batch['txt'] 

            # sample timesteps
            batch_size = src.shape[0]
            t = torch.randint(0, num_timesteps, (batch_size,), device=device).long()

            loss, pred = model(src, tgt, txt, t)
            
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()

            # Print loss
            if global_step % STEPS_PER_LOG == 0:
                    print(f"Step: {global_step:03d} , Epoch: {epoch+1} , Loss: {loss.item():.4f}")

            global_step += 1

        # save checkpoint every 3 epochs
        if (epoch + 1) % 3 == 0:
            unwrapped_model = accelerator.unwrap_model(model)
            save_path = f"{OUTPUT_DIR}/GHT_epoch_{epoch+1}.pth"
            torch.save(unwrapped_model.state_dict(), save_path)
            print(f"checkpoint saved : {save_path}")

if __name__ == "__main__":
    main()