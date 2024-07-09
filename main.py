import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from fff.loss import fff_loss
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from architecture import transformer as T


class FreeFormFlowUpscaler(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root_dir, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        low_res = F.interpolate(img.unsqueeze(0), scale_factor=0.5, mode='bicubic').squeeze(0)
        return low_res, img


def psnr(img1, img2):
    return peak_signal_noise_ratio(img1.cpu().numpy(), img2.cpu().numpy())


def ssim(img1, img2):
    return structural_similarity(img1.cpu().numpy(), img2.cpu().numpy(), multichannel=True)


def train(model, train_loader, val_loader, optimizer, scheduler, scaler, beta, device, epochs):
    model.train()
    best_val_psnr = 0
    for epoch in range(epochs):
        train_loss = 0
        for batch in train_loader:
            low_res, high_res = batch
            low_res, high_res = low_res.to(device), high_res.to(device)

            optimizer.zero_grad()
            with autocast():
                output = model(low_res)
                loss = fff_loss(low_res, model.encoder, model.decoder, beta) + F.mse_loss(output, high_res)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_psnr, val_ssim = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val PSNR: {val_psnr:.2f}, Val SSIM: {val_ssim:.4f}")

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(model.state_dict(), 'best_fff_upscaler.pth')


def evaluate(model, val_loader, device):
    model.eval()
    psnr_values = []
    ssim_values = []
    with torch.no_grad():
        for low_res, high_res in val_loader:
            low_res, high_res = low_res.to(device), high_res.to(device)
            output = model(low_res)
            psnr_values.append(psnr(output, high_res))
            ssim_values.append(ssim(output, high_res))
    return np.mean(psnr_values), np.mean(ssim_values)


def upscale(model, low_res_image):
    model.eval()
    with torch.no_grad():
        return model(low_res_image)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Datentransformationen
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # Datensatz und DataLoader
    train_dataset = CustomDataset(root_dir='path/to/train/data', transform=transform)
    val_dataset = CustomDataset(root_dir='path/to/val/data', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Modell auswählen
    model = FreeFormFlowUpscaler(T.TransformerEncoder, T.TransformerDecoder).to(device)
    model.apply(init_weights)

    # Optimierer und Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    num_epochs = 50
    scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=num_epochs)
    scaler = GradScaler()

    # Training
    try:
        train(model, train_loader, val_loader, optimizer, scheduler, scaler, beta=1.0, device=device, epochs=num_epochs)
    except KeyboardInterrupt:
        print("Training interrupted. Saving current model state.")
    finally:
        torch.save(model.state_dict(), 'final_fff_upscaler.pth')

    # Beispiel für Upscaling
    test_image = torch.randn(1, 3, 32, 32).to(device)
    upscaled_image = upscale(model, test_image)
    print(f"Upscaled image shape: {upscaled_image.shape}")