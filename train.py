import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import config
from dataset import smart_prepare_data, InMemoryVisualDynamicsDataset
from models import VisualDynamicsModel
from utils import set_seed, kld_loss, collect_empirical_z, visualize_empirical

def parse_args():
    parser = argparse.ArgumentParser(description="Future Frame Synthesis Training")
    
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--extract_root', type=str, default='/content/data')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    
    parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    
    set_seed(config.SEED)
    print(f"Epochs={args.epochs}, BS={args.batch_size}, LR={args.lr}")
    print(f"Data Root: {args.data_root}")
    print(f"Save Dir: {args.save_dir}")

    video_map = smart_prepare_data(zip_root=args.data_root, extract_root=args.extract_root)
    if not video_map: return

    train_ds = InMemoryVisualDynamicsDataset(video_map, config.VIDEO_CONFIG, augment=True, split_mode='train')
    test_ds = InMemoryVisualDynamicsDataset(video_map, config.VIDEO_CONFIG, augment=False, split_mode='test')

    if len(train_ds) == 0:
        print("ERROR: No pairs created.")
        return

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    viz_loader = DataLoader(test_ds, batch_size=16, shuffle=True)
    pool_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    model = VisualDynamicsModel(z_dim=config.Z_DIM).cuda()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    os.makedirs(args.save_dir, exist_ok=True)
    best_model_path = os.path.join(args.save_dir, "human_best_modular.pt")
    latest_model_path = os.path.join(args.save_dir, "human_latest_checkpoint.pt")

    if os.path.exists(latest_model_path):
        print(f"Resuming from saved model: {latest_model_path}")
        try:
            model.load_state_dict(torch.load(latest_model_path))
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Load failed: {e}.")
    else:
        print("Starting training.")

    max_beta = 0.0001
    best_loss = float('inf')

    for ep in range(1, config.EPOCHS + 1):
        model.train()
        total_rec = 0
        total_kl = 0

        if ep <= config.WARMUP_EPOCHS: 
            beta = max_beta * (ep / config.WARMUP_EPOCHS)
        else: 
            beta = max_beta

        loop = tqdm(train_loader, desc=f"Ep {ep}", leave=False)
        for batch in loop:
            im1 = batch['im1'].cuda()
            v = batch['v'].cuda()
            
            optimizer.zero_grad()
            v_pred, mu, lv = model(im1, v)

            diff = torch.abs(v)
            mask = (diff > 0.5).float()
            weights = 1.0 + (mask * 5.0)

            rec = torch.mean(torch.abs(v_pred - v) * weights)
            kl = kld_loss(mu, lv)
            loss = rec + beta * kl

            loss.backward()
            optimizer.step()

            total_rec += rec.item()
            total_kl += kl.item()
            loop.set_postfix(rec=rec.item(), kl=kl.item())

        avg_rec = total_rec / len(train_loader)
        current_loss = avg_rec + (total_kl / len(train_loader))
        
        print(f"E{ep} | Rec: {avg_rec:.4f} | KL: {total_kl/len(train_loader):.4f} | Loss: {current_loss:.4f}")
        scheduler.step(avg_rec)

        if current_loss < best_loss:
            best_loss = current_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best Model Saved. ({best_loss:.4f})")

        if ep % 5 == 0:
            z_pool = collect_empirical_z(model, pool_loader, num_samples=300)
            visualize_empirical(model, viz_loader, z_pool, ep)
            
            torch.save(model.state_dict(), latest_model_path)
            
            if ep % 25 == 0:
                backup_path = os.path.join(args.save_dir, f"human_backup_ep{ep}.pt")
                torch.save(model.state_dict(), backup_path)
                print(f"Backup Model Saved: {backup_path}")

if __name__ == "__main__":
    main()