import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ERA5Dataset(Dataset):
    def __init__(self, split="train", data_dir="processed_era5_t2m_downscaling"):
        self.lr = np.load(f"{data_dir}/{split}_lr.npy", mmap_mode="r")
        self.hr = np.load(f"{data_dir}/{split}_hr.npy", mmap_mode="r")
        print(f"Loaded {split}: {len(self.lr)} samples")

    def __len__(self):
        return len(self.lr)

    def __getitem__(self, idx):
        lr = torch.from_numpy(np.array(self.lr[idx], dtype=np.float32)).unsqueeze(0)
        hr = torch.from_numpy(np.array(self.hr[idx], dtype=np.float32)).unsqueeze(0)
        return lr, hr

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attn(x)

class DownscalingUNet(nn.Module):
    def __init__(self):
        super().__init__()

       
        self.enc1 = ConvBlock(1, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)

        self.pool = nn.MaxPool2d(2)

        
        self.bottleneck = ConvBlock(128, 256)
        self.attn = AttentionGate(256)

      
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)

        
        self.final_up = nn.Sequential(
            nn.Conv2d(32, 32 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            nn.Conv2d(32, 32 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU()
        )

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
   
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

    
        b = self.attn(self.bottleneck(self.pool(e3)))

    
        d3 = self.up3(b)
        e3 = nn.functional.interpolate(e3, size=d3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        e2 = nn.functional.interpolate(e2, size=d2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        e1 = nn.functional.interpolate(e1, size=d1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.final_up(d1)
        return self.out(out)


def train_model(model, train_loader, val_loader, epochs=50, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for lr_b, hr_b in train_loader:
            lr_b, hr_b = lr_b.to(device), hr_b.to(device)

            pred = model(lr_b)

            min_h = min(pred.shape[2], hr_b.shape[2])
            min_w = min(pred.shape[3], hr_b.shape[3])

            loss = criterion(pred[:, :, :min_h, :min_w],
                             hr_b[:, :, :min_h, :min_w])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for lr_b, hr_b in val_loader:
                lr_b, hr_b = lr_b.to(device), hr_b.to(device)
                pred = model(lr_b)

                min_h = min(pred.shape[2], hr_b.shape[2])
                min_w = min(pred.shape[3], hr_b.shape[3])

                val_loss += criterion(
                    pred[:, :, :min_h, :min_w],
                    hr_b[:, :, :min_h, :min_w]
                ).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_unet_downscaling.pth")

    return model



if __name__ == "__main__":
    print(" UNet-based Downscaling Model")

    train_set = ERA5Dataset(split="train")
    val_set   = ERA5Dataset(split="val")

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=8, shuffle=False)

    model = DownscalingUNet()

    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    model = train_model(model, train_loader, val_loader, epochs=50)

    print("训练完成!")