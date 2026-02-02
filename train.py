import os
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data import GHTDataset
from model import GenHowToModel

# config details set for training on one sample
with open('config.json') as f:
    config = json.load(f)

DATA_ROOT = config["DATA_ROOT"]
OUTPUT_DIR = config["OUTPUT_DIR"]
LR = config["LR"]
EPOCHS = config["EPOCHS"]
STEPS_PER_LOG = config["STEPS_PER_LOG"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Root: {DATA_ROOT}")
    print(f"LR: {LR}")
    print(f"Epochs: {EPOCHS}")

    # handle data and create dataloader
    dataset = GHTDataset(root=DATA_ROOT, train_final_state_model=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    print(f"Dataset size: {len(dataset)}")

    # calling model
    model = GenHowToModel()
    model.to(DEVICE)
    model.train()

    # optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=LR) # opt

    # Trainign loop
    global_step = 0
    num_timesteps = model.scheduler.config.num_train_timesteps

    for epoch in range(EPOCHS):
        print(f"\n Epoch: {epoch+1}/{EPOCHS}")

        for batch in loader:
            src = batch['src'].to(DEVICE)
            tgt = batch['tgt'].to(DEVICE)
            txt = batch['txt'] # list of strings so no need to move

            # sample timesteps
            bs = src.shape[0]
            t = torch.randint(0, num_timesteps, (bs,), device=DEVICE).long()

            loss, pred = model(src, tgt, txt, t) # model returns (loss, pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss
            if global_step % STEPS_PER_LOG == 0:
                print(f"Step: {global_step:05d} , Epoch: {epoch+1} , Loss: {loss.item():.4f}")

            global_step += 1

        # save checkpoint every 3 epochs
        if (epoch + 1) % 3 == 0:
            save_path = f"{OUTPUT_DIR}/GHT_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"checkpoint saved : {save_path}")

if __name__ == "__main__":
    main()
