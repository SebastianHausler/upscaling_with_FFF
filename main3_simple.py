import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from fff.loss import fff_loss
from PIL import Image
import os
import numpy as np
import argparse

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))

class FreeFormFlow(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.channels = channels
        
        # Encoder (dimensionserhaltend)
        self.encoder = nn.Sequential(
            ConvBlock(channels, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 64),
            nn.Conv2d(64, channels, kernel_size=3, padding=1)
        )
        
        # Decoder (approximative Inverse des Encoders)
        self.decoder = nn.Sequential(
            ConvBlock(channels, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 64),
            nn.Conv2d(64, channels, kernel_size=3, padding=1)
        )
        
        # Latent-Verteilung (gleiche Dimensionalität wie Eingang)
        self.latent_distribution = torch.distributions.Normal(
            loc=torch.zeros(1),
            scale=torch.ones(1)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def to(self, device):
        super().to(device)
        self.latent_distribution.loc = self.latent_distribution.loc.to(device)
        self.latent_distribution.scale = self.latent_distribution.scale.to(device)
        return self

class Upscaler(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv1 = ConvBlock(3, 64)
        self.conv2 = ConvBlock(64, 64)
        self.conv3 = ConvBlock(64, 3)
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class DIV2KDataset(Dataset):
    def __init__(self, hr_dir, crop_size=256, scale_factor=0.5):
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
        
        lr_size = int(self.crop_size * self.scale_factor)
        lr_img = F.interpolate(hr_img.unsqueeze(0), size=(lr_size, lr_size), mode='bicubic', align_corners=False).squeeze(0)

        return lr_img, hr_img

def custom_collate(batch):
    lr_imgs, hr_imgs = zip(*batch)
    lr_imgs = torch.stack(lr_imgs)
    hr_imgs = torch.stack(hr_imgs)
    return lr_imgs, hr_imgs

def train(model, upscaler, train_loader, optimizer, device, beta):
    model.train()
    upscaler.train()
    train_loss = 0
    for batch in train_loader:
        low_res, high_res = batch
        low_res, high_res = low_res.to(device), high_res.to(device)
        
        optimizer.zero_grad()
        
        # FFF Loss
        fff_loss_value = fff_loss(
            low_res, 
            model.encode, 
            model.decode,
            model.latent_distribution, 
            beta,
            hutchinson_samples=1
        )
        
        # Upscaling Loss
        encoded = model.encode(low_res)
        decoded = model.decode(encoded)
        upscaled = upscaler(decoded)
        mse_loss = F.mse_loss(upscaled, high_res)
        
        total_loss = fff_loss_value.mean() + mse_loss
        total_loss.backward()
        optimizer.step()
        
        train_loss += total_loss.item()
    
    return train_loss / len(train_loader)

def upscale_image(model, upscaler, image_path, scale_factor, device):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        encoded = model.encode(img_tensor)
        decoded = model.decode(encoded)
        upscaled = upscaler(decoded)
    
    upscaled = upscaled.squeeze(0).cpu()
    upscaled = transforms.ToPILImage()(upscaled)
    return upscaled

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FFF Upscaler")
    parser.add_argument('--mode', choices=['train', 'upscale'], required=True)
    parser.add_argument('--scale', type=int, choices=[2, 4], default=2)
    parser.add_argument('--input', type=str, help='Input image path for upscaling')
    parser.add_argument('--output', type=str, help='Output image path for upscaling')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.mode == 'train':
        # Datensatz und DataLoader
        train_hr_dir = 'C:/Users/Admin/Desktop/DIV2K_train_HR'
        val_hr_dir = 'C:/Users/Admin/Desktop/DIV2K_valid_HR'
        train_dataset = DIV2KDataset(train_hr_dir, crop_size=256, scale_factor=1/args.scale)
        val_dataset = DIV2KDataset(val_hr_dir, crop_size=256, scale_factor=1/args.scale)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=custom_collate)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=custom_collate)

        # Modell, Upscaler, Optimierer und andere Hyperparameter
        model = FreeFormFlow().to(device)
        upscaler = Upscaler(scale_factor=args.scale).to(device)
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(upscaler.parameters()), lr=0.001, weight_decay=0.01)
        n_epochs = 50
        beta = 1.0

        # Training Loop
        for epoch in range(n_epochs):
            train_loss = train(model, upscaler, train_loader, optimizer, device, beta)
            print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}")

        torch.save({
            'fff_model': model.state_dict(),
            'upscaler': upscaler.state_dict()
        }, f'fff_upscaler_{args.scale}x.pth')

    elif args.mode == 'upscale':
        if not args.input or not args.output:
            print("Please provide input and output image paths for upscaling.")
            exit(1)

        model = FreeFormFlow().to(device)
        upscaler = Upscaler(scale_factor=args.scale).to(device)
        
        checkpoint = torch.load(f'fff_upscaler_{args.scale}x.pth')
        model.load_state_dict(checkpoint['fff_model'])
        upscaler.load_state_dict(checkpoint['upscaler'])

        upscaled_img = upscale_image(model, upscaler, args.input, args.scale, device)
        upscaled_img.save(args.output)
        print(f"Upscaled image saved to {args.output}")