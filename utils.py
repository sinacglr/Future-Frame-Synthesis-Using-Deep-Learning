import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import config

def kld_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def collect_empirical_z(model, loader, num_samples=500):
    model.eval()
    z_pool = []
    count = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in loader:
            im1 = batch['im1'].to(device)
            v = batch['v'].to(device)
            mu, _ = model.encode(im1, v)
            z_pool.append(mu.cpu())
            count += im1.size(0)
            if count >= num_samples: break
    return torch.cat(z_pool, dim=0)[:num_samples]

@torch.no_grad()
def visualize_empirical(model, loader, z_pool, epoch):
    model.eval()
    device = next(model.parameters()).device

    try:
        batch = next(iter(loader))
    except: return

    num_show = 10
    if len(batch['im1']) < num_show: num_show = len(batch['im1'])

    im1 = batch['im1'][:num_show].to(device)
    im2 = batch['im2'][:num_show].to(device)

    idx = torch.randint(0, z_pool.size(0), (num_show,))
    z_sampled = z_pool[idx].to(device)
    v_emp = model(im1, None, z_input=z_sampled)
    emp_img = (im1 + (v_emp / config.SIGNAL_GAIN)).clamp(0, 1)

    fig, axes = plt.subplots(num_show, 3, figsize=(10, 22))
    plt.suptitle(f"Epoch {epoch} | Input -> Target | Prediction", fontsize=14, y=1.005)

    for i in range(num_show):
        axes[i,0].imshow(im1[i].permute(1,2,0).cpu())
        axes[i,0].axis('off')
        if i==0: axes[i,0].set_title("Input")

        axes[i,1].imshow(im2[i].permute(1,2,0).cpu())
        axes[i,1].axis('off')
        if i==0: axes[i,1].set_title("Target")

        axes[i,2].imshow(emp_img[i].permute(1,2,0).cpu())
        axes[i,2].axis('off')
        if i==0: axes[i,2].set_title("Prediction (Empirical)")

    plt.tight_layout()
    plt.show()
    model.train()
