import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from einops import rearrange
from fff.loss import fff_loss
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity



from PIL import Image
import os

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))

class SimpleConvEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.flatten(x)
        return self.fc(x)

class SimpleConvDecoder(nn.Module):
    def __init__(self, latent_dim=256, out_channels=3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.conv1 = ConvBlock(256, 128)
        self.conv2 = ConvBlock(128, 64)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.fc(x).view(-1, 256, 4, 4)
        x = self.upsample(self.conv1(x))
        x = self.upsample(self.conv2(x))
        x = self.upsample(self.conv3(x))
        return torch.tanh(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(channels)
        self.activation = nn.SiLU()

    def forward(self, x):
        residual = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)

class ResNetEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.res1 = ResidualBlock(64)
        self.conv2 = ConvBlock(64, 128)
        self.res2 = ResidualBlock(128)
        self.conv3 = ConvBlock(128, 256)
        self.res3 = ResidualBlock(256)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.conv2(x)
        x = self.res2(x)
        x = self.conv3(x)
        x = self.res3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

class ResNetDecoder(nn.Module):
    def __init__(self, latent_dim=256, out_channels=3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.conv1 = ConvBlock(256, 128)
        self.res1 = ResidualBlock(128)
        self.conv2 = ConvBlock(128, 64)
        self.res2 = ResidualBlock(64)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.fc(x).view(-1, 256, 4, 4)
        x = self.upsample(self.conv1(x))
        x = self.res1(x)
        x = self.upsample(self.conv2(x))
        x = self.res2(x)
        x = self.upsample(self.conv3(x))
        return torch.tanh(x)

class UNetEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.conv4 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * 2 * 2, latent_dim)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        x4 = self.conv4(self.pool(x3))
        x = self.flatten(self.pool(x4))
        return self.fc(x)

class UNetDecoder(nn.Module):
    def __init__(self, latent_dim=256, out_channels=3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 2 * 2)
        self.conv1 = ConvBlock(512, 256)
        self.conv2 = ConvBlock(256, 128)
        self.conv3 = ConvBlock(128, 64)
        self.conv4 = nn.Conv2d(64, out_channels, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.fc(x).view(-1, 512, 2, 2)
        x = self.upsample(self.conv1(x))
        x = self.upsample(self.conv2(x))
        x = self.upsample(self.conv3(x))
        x = self.upsample(self.conv4(x))
        return torch.tanh(x)

class TransformerEncoder(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, embed_dim=256, depth=6, num_heads=8, mlp_ratio=4., latent_dim=256):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, (32 // patch_size) ** 2, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * mlp_ratio), batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, latent_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.norm(x.mean(dim=1))
        return self.fc(x)

class TransformerDecoder(nn.Module):
    def __init__(self, latent_dim=256, embed_dim=256, depth=6, num_heads=8, mlp_ratio=4., patch_size=4, out_channels=3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, (32 // patch_size) ** 2, embed_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * mlp_ratio), batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.patch_size = patch_size
        self.output_proj = nn.Linear(embed_dim, patch_size ** 2 * out_channels)

    def forward(self, x):
        x = self.fc(x).unsqueeze(1)
        x = x + self.pos_embed
        x = self.transformer(x, x)
        x = self.norm(x)
        x = self.output_proj(x)
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
                      h=32//self.patch_size, w=32//self.patch_size, 
                      p1=self.patch_size, p2=self.patch_size)
        return torch.tanh(x)

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
    
class DIV2KDataset(Dataset):
    def __init__(self, hr_dir, crop_size=256, scale_factor=0.25):
        self.hr_dir = hr_dir
        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.image_files = [f for f in os.listdir(hr_dir) if f.endswith('.png')]
        self.transform = transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        hr_img = Image.open(os.path.join(self.hr_dir, img_name))
        
        hr_img = self.transform(hr_img)
        
        # Erzeuge das LR-Bild on-the-fly
        lr_img = F.interpolate(hr_img.unsqueeze(0), scale_factor=self.scale_factor, mode='bicubic').squeeze(0)

        return lr_img, hr_img
    
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
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val PSNR: {val_psnr:.2f}, Val SSIM: {val_ssim:.4f}")
        
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

def custom_collate(batch):
    lr_imgs, hr_imgs = zip(*batch)
    lr_imgs = torch.stack(lr_imgs)
    hr_imgs = torch.stack(hr_imgs)
    return lr_imgs, hr_imgs

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                                
    # Pfad zum DIV2K-Datensatz (nur HR benötigt)
    train_hr_dir = 'C:/Users/Admin/Desktop/DIV2K_train_HR'
    val_hr_dir = 'C:/Users/Admin/Desktop/DIV2K_valid_HR'

    # Datensatz und DataLoader
    train_dataset = DIV2KDataset(train_hr_dir, crop_size=256)
    val_dataset = DIV2KDataset(val_hr_dir, crop_size=256)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=custom_collate)
  
    # Modell auswählen
    model = FreeFormFlowUpscaler(SimpleConvEncoder(), SimpleConvDecoder()).to(device)
   # model.apply(init_weights)
    
    # Optimierer und Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    num_epochs = 50
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=num_epochs)

        # GradScaler für Mixed Precision Training
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
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