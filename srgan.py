import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, datasets, models
import torchvision.transforms.functional as TF
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim
from skimage.color import rgb2ycbcr
from PIL import Image
import torch.nn.functional as F


################################################################################
#0 TV Loss Module
#Total Variation Loss to reduce artifacts in generated images
#This loss penalizes large gradients in the image, encouraging smoothness.
class TVLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        batch_size = x.size(0)
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
        w_tv =torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        return self.weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.numel()

################################################################################
#1 Network Components

#1.1 Residual Block for the Generator
#Each block consists of two convolutional layers with BatchNorm and PReLU activation.
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )
    def forward(self, x):
        return x + self.block(x)

#1.2 Generator Network with Tanh in the Final Layer
#The generator uses a series of residual blocks, followed by upsampling layers.
#The final layer uses Tanh activation to constrain outputs to [-1, 1].
class Generator(nn.Module): 
    def __init__(self, num_residual_blocks=16):
        super(Generator, self).__init__()
        self.initial =nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )
        self.residuals = nn.Sequential(*[ResidualBlock(64) for _ in range(num_residual_blocks)])
        self.conv_mid = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU()
        )
        # Tanh activation
        self.final = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4),
            nn.Tanh()
        )
    def forward(self, x):

        out1 = self.initial(x)
        out = self.residuals(out1)
        out = self.conv_mid(out)
        out = out1 + out # Skip connection
        out = self.upsample(out)
        return self.final(out)

#1.3 Discriminator Network Components
#Each block consists of a convolutional layer followed by BatchNorm and LeakyReLU activation.
def disc_block(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True)
    )

#1.4 Discriminator Network
#The discriminator uses a series of convolutional layers to classify images as real or fake.
#The final layer outputs a single value (real or fake) using Sigmoid activation.
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        channels, height, width = input_shape
        self.model= nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            disc_block(64, 64, stride=2),
            disc_block(64, 128, stride=1),
            disc_block(128, 128, stride=2),
            disc_block(128, 256, stride=1),
            disc_block(256, 256, stride=2),
            disc_block(256, 512, stride=1),
            disc_block(512, 512, stride=2)
        )
        self.adv_layer = nn.Sequential(
            nn.Linear(512 * (height // 16) * (width // 16), 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
    def forward(self, img):
        out = self.model(img)
        out = out.view(out.size(0), -1)
        return self.adv_layer(out)

################################################################################
#2. Custom Dataset

class SRDataset(Dataset):
    def __init__(self, hr_dataset, hr_transform, lr_transform):
        self.hr_dataset = hr_dataset
        self.hr_transform = hr_transform
        self.lr_transform = lr_transform
    def __getitem__(self, index):
        img, _ = self.hr_dataset[index] # Expecting a PIL Image
        hr_img = self.hr_transform(img)
        lr_img = self.lr_transform(img)
        return lr_img, hr_img
    def __len__(self):
        return len(self.hr_dataset)

################################################################################
#3. Data Transforms & DataLoader
#Transforms for high-resolution and low-resolution images.
#Assuming original images are in [0,1] after ToTensor()
hr_transform =transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1) # Scale [0,1] -> [-1,1]
])
lr_transform = transforms.Compose([
    transforms.Resize((64, 64)), # 4x downsampling
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1)
])

#Use ImageFolder with transform=None so that SRDataset handles transformation.
dataset = datasets.ImageFolder(r"C:\Users\sfboz\Desktop\archive2", transform=None)
sr_dataset = SRDataset(dataset, hr_transform, lr_transform)
#For testing, we can limit the dataset size to a smaller subset.
#sr_dataset = Subset(sr_dataset, list(range(100)))
dataloader = DataLoader(sr_dataset, batch_size=16, shuffle=True)

################################################################################
#4. Loss Functions & VGG Feature Extractor
#Loss functions for the generator and discriminator.
criterion_GAN= nn.BCELoss() # Adversarial loss
criterion_content = nn.MSELoss() # For L2 and perceptual losses
tv_loss_fn = TVLoss(weight=1e-6) # TV Loss

#We'll use a pre-trained VGG19 for perceptual loss.
#VGG expects images in [0,1] normalized by ImageNet stats.
vgg_mean = [0.485, 0.456, 0.406]
vgg_std = [0.229, 0.224, 0.225]
vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg_feature_extractor = nn.Sequential(*list(vgg.features)[:35]).eval().to(device)
for param in vgg_feature_extractor.parameters():
    param.requires_grad = False

#Helper function to normalize images for VGG input
def normalize_for_vgg(x):
    # x is a tensor in [-1, 1]
    x = (x + 1) / 2.0 # now in [0,1]
    return TF.normalize(x, mean=vgg_mean, std=vgg_std)

################################################################################
#5. Models, Optimizers, Hyperparameters
PRETRAIN_EPOCHS =5
FINE_TUNE_EPOCHS = 3
lr_gen = 1e-4 # Learning rate for generator
lr_disc = 1e-4 # Learning rate for discriminator
adv_coeff = 1e-3 # Weight for adversarial loss
vgg_coeff = 0.006 # Weight for perceptual loss
tv_coeff = 1e-6 # Weight for TV loss

#Instantiate models
generator= Generator().to(device)
discriminator = Discriminator(input_shape=(3, 256, 256)).to(device)

#Initialize weights for generator and discriminator
optimizer_G = optim.Adam(generator.parameters(), lr=lr_gen)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_disc)

#Scheduler for fine-tuning (for generator only)
scheduler = optim.lr_scheduler.StepLR(optimizer_G, step_size=2, gamma=0.5)

################################################################################
#6. Two-Phase Training Loop
# Phase 1: Pretraining with L2 Loss Only
print("Pretraining Phase(L2 Loss Only)")
generator.train()
for epoch in range(PRETRAIN_EPOCHS):
    epoch_loss = 0.0
    for imgs_lr, imgs_hr in dataloader:
        imgs_lr= imgs_lr.to(device)
        imgs_hr = imgs_hr.to(device)
        
        output= generator(imgs_lr)
        loss = criterion_content(output, imgs_hr) # L2 loss
        
        optimizer_G.zero_grad()
        loss.backward()
        optimizer_G.step()
        epoch_loss +=loss.item()
    print(f"Pretrain Epoch {epoch+1}/{PRETRAIN_EPOCHS}, Loss: {epoch_loss/len(dataloader):.4f}")
    torch.save(generator.state_dict(), f'pretrain_generator_epoch_{epoch}.pth')

# Phase 2: Fine-Tuning with Combined Losses
#In this phase, we use the generator and discriminator together.
print("Fine-tuning Phase(combined losses)")
generator.train()
discriminator.train()
for epoch in range(FINE_TUNE_EPOCHS):
    epoch_loss_G = 0.0
    epoch_loss_D = 0.0
    for imgs_lr,imgs_hr in dataloader:
        imgs_lr = imgs_lr.to(device)
        imgs_hr = imgs_hr.to(device)
        
        # Create ground truth labels
        # For real images, we want the discriminator to output "real" (1)
        # For fake images (generated by the generator), we want it to output "fake" (0)
        valid = torch.ones((imgs_hr.size(0), 1), requires_grad=False).to(device)
        fake = torch.zeros((imgs_hr.size(0), 1), requires_grad=False).to(device)
        
        #### Train Discriminator ####
        optimizer_D.zero_grad()
        fake_hr = generator(imgs_lr)
        loss_real = criterion_GAN(discriminator(imgs_hr), valid)
        loss_fake = criterion_GAN(discriminator(fake_hr.detach()), fake)
        loss_D = (loss_real + loss_fake)/2
        loss_D.backward()
        optimizer_D.step()
        
        #### Train Generator ####
        optimizer_G.zero_grad()
        fake_hr = generator(imgs_lr)
        fake_prob = discriminator(fake_hr)
        
        #Compute perceptual loss using VGG features:
        #Convert from [-1,1] to VGG input format.
        features_fake = vgg_feature_extractor(normalize_for_vgg(fake_hr))
        features_real = vgg_feature_extractor(normalize_for_vgg(imgs_hr))
        loss_perceptual = criterion_content(features_fake, features_real.detach())
        

        loss_L2 = criterion_content(fake_hr, imgs_hr) # L2 (content) loss between generated and ground truth images, the pixel-wise loss
        loss_GAN = criterion_GAN(fake_prob, valid) # Adversarial loss (generator wants discriminator to output "real")
        loss_tv = tv_loss_fn(fake_hr) #Total Variation loss to reduce artifacts
        
        # Total generator loss: weighted sum
        loss_G = loss_L2 + vgg_coeff * loss_perceptual + adv_coeff * loss_GAN + tv_coeff * loss_tv
        loss_G.backward()
        optimizer_G.step()
        
        epoch_loss_G += loss_G.item()
        epoch_loss_D += loss_D.item()
    
    scheduler.step()
    print(f"Fine-Tune epoch {epoch+1}/{FINE_TUNE_EPOCHS}, Generator loss: {epoch_loss_G/len(dataloader):.4f}, Discriminator loss: {epoch_loss_D/len(dataloader):.4f}")
    torch.save(generator.state_dict(), f'fine_tuned_generator_epoch_{epoch}.pth')
    torch.save(discriminator.state_dict(), f'fine_tuned_discriminator_epoch_{epoch}.pth')

################################################################################
#7. Evaluation: Compare with Bicubic Upsampling

# Containers for per-image metrics
psnr_bic_list, ssim_bic_list = [], []
psnr_srgan_list, ssim_srgan_list = [], []

generator.eval()
with torch.no_grad():
    for imgs_lr, imgs_hr in dataloader:
        imgs_lr = imgs_lr.to(device) # LR in [-1,1]
        imgs_hr = imgs_hr.to(device) # HR in [-1,1]

        # 1) Generate SRGAN output
        fake_hr = generator(imgs_lr)
        fake_hr = torch.clamp(fake_hr, -1, 1)

        # 2) Bicubic upsampling (tensor)
        #    Note: scale_factor=4 because 64→256
        bic = F.interpolate(imgs_lr, scale_factor=4, mode='bicubic', align_corners=False)
        bic = torch.clamp(bic, -1, 1)

        # 3) Unnormalize back to [0,1] for metric computation
        hr_01 = (imgs_hr + 1) /2.0
        bic_01 = (bic+ 1) /2.0
        srgan_01= (fake_hr + 1) /2.0

        # 4) Move to CPU numpy
        hr_np = hr_01.cpu().permute(0,2,3,1).numpy()
        bic_np = bic_01.cpu().permute(0,2,3,1).numpy()
        srgan_np = srgan_01.cpu().permute(0,2,3,1).numpy()

        # 5) Compute metrics per image
        for gt, b, g in zip(hr_np,bic_np, srgan_np):
            # PSNR
            psnr_bic_list.append(compare_psnr(gt,b,data_range=1.0))
            psnr_srgan_list.append(compare_psnr(gt,g,data_range=1.0))
            # SSIM
            ssim_bic_list.append(compare_ssim(gt,b,data_range=1.0, channel_axis=-1))
            ssim_srgan_list.append(compare_ssim(gt,g,data_range=1.0, channel_axis=-1))

# 6) Summarize: mean ± std
def summarize(name, psnr_list, ssim_list):
    print(f"{name:<8} | PSNR: {np.mean(psnr_list):.2f} ± {np.std(psnr_list):.2f} " +
          f"| SSIM: {np.mean(ssim_list):.4f} ± {np.std(ssim_list):.4f}")

print("\nAverage Performance over Entire Dataset")
summarize("Bicubic", psnr_bic_list, ssim_bic_list)
summarize("SRGAN",   psnr_srgan_list, ssim_srgan_list)

#Unnormalize tensor for visualization (convert from [-1,1] to [0,1])
def unnormalize(tensor):
    return (tensor + 1)/2.0

#Set generator to evaluation mode and generate an output sample
generator.eval()
with torch.no_grad():
    # Get a batch of images from the dataloader
    imgs_lr, imgs_hr = next(iter(dataloader))
    imgs_lr = imgs_lr.to(device)
    imgs_hr = imgs_hr.to(device)
    fake_hr = generator(imgs_lr)
    fake_hr = torch.clamp(fake_hr, -1, 1)

#Convret outputs from [-1,1] to [0,1] for visualization and metric calculation
#Unnormalize the images for visualization
lr_img = unnormalize(imgs_lr[0].cpu()).numpy().transpose(1, 2, 0)
srgan_img = unnormalize(fake_hr[0].cpu()).numpy().transpose(1, 2, 0)
hr_img = unnormalize(imgs_hr[0].cpu()).numpy().transpose(1, 2, 0)

#Bicubic upsampling of the low-resolution image using OpenCV.
lr_img_cv = (lr_img * 255).astype(np.uint8)
bicubic_img_cv = cv2.resize(lr_img_cv, (256, 256), interpolation=cv2.INTER_CUBIC)
bicubic_img = bicubic_img_cv.astype(np.float32) /255.0

#Compute PSNR (both standard and on the Y channel)
psnr_srgan = compare_psnr(hr_img, srgan_img, data_range=1)
psnr_bicubic = compare_psnr(hr_img, bicubic_img, data_range=1)


#Compute SSIM (using full color images)
ssim_srgan = compare_ssim(hr_img, srgan_img, data_range=1, channel_axis=-1)
ssim_bicubic = compare_ssim(hr_img, bicubic_img, data_range=1, channel_axis=-1)

print("\nEvaluation Metrics:")
print(f"SRGAN - PSNR: {psnr_srgan:.2f} , SSIM: {ssim_srgan:.4f}")
print(f"Bicubic - PSNR: {psnr_bicubic:.2f} , SSIM: {ssim_bicubic:.4f}")

#Display the images for visual comparison
plt.figure(figsize=(12, 4))
plt.subplot(1, 4, 1)
plt.title("Low-Resolution Input")
plt.imshow(lr_img)
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title("Bicubic Upsampling")
plt.imshow(bicubic_img)
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title("SRGAN Output")
plt.imshow(srgan_img)
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title("High-Resolution GT")
plt.imshow(hr_img)
plt.axis('off')

plt.tight_layout()
plt.show()
import os

#Define output directories for the two types of images
output_dir_bicubic = r"C:\Users\sfboz\Desktop\outputs\bicubic"
output_dir_srgan = r"C:\Users\sfboz\Desktop\outputs\srgan"
os.makedirs(output_dir_bicubic, exist_ok=True)
os.makedirs(output_dir_srgan, exist_ok=True)

generator.eval()

# Loop over the dataset; here we use the underlying ImageFolder dataset from SRDataset
with torch.no_grad():
    for idx in range(len(sr_dataset)):
        # Retrieve the transformed images (low-res and high-res) from your custom dataset
        lr_tensor, hr_tensor=sr_dataset[idx]
        
        # Get the original file name from the ImageFolder dataset (assumes order is preserved)
        file_path, _ = dataset.samples[idx]
        filename = os.path.basename(file_path)
        name, ext =os.path.splitext(filename)
        
        # Add a batch dimension and move to device
        lr_tensor_batch = lr_tensor.unsqueeze(0).to(device)
        
        # Generate SRGAN output
        srgan_output = generator(lr_tensor_batch)
        srgan_output = torch.clamp(srgan_output, -1, 1)
        
        # Unnormalize and convert tensor to NumPy array for visualization/saving
        srgan_img_np = unnormalize(srgan_output[0].cpu()).numpy().transpose(1, 2, 0)
        lr_img_np = unnormalize(lr_tensor_batch[0].cpu()).numpy().transpose(1, 2, 0)
        
        # Bicubic upsampling of the low-resolution image using OpenCV
        lr_img_cv = (lr_img_np * 255).astype(np.uint8)
        bicubic_img_cv =cv2.resize(lr_img_cv, (256, 256), interpolation=cv2.INTER_CUBIC)
        bicubic_img = bicubic_img_cv.astype(np.float32) / 255.0
        
        # Prepare images for saving (convert to 8-bit and RGB to BGR if using cv2)
        srgan_save = (srgan_img_np * 255).astype(np.uint8)
        bicubic_save = (bicubic_img * 255).astype(np.uint8)
        srgan_save = cv2.cvtColor(srgan_save, cv2.COLOR_RGB2BGR)
        bicubic_save = cv2.cvtColor(bicubic_save, cv2.COLOR_RGB2BGR)
        
        # Build output file names
        srgan_filename = os.path.join(output_dir_srgan, f"{name}_srgan{ext}")
        bicubic_filename = os.path.join(output_dir_bicubic, f"{name}_bicubic{ext}")
        
        #Save the images
        cv2.imwrite(srgan_filename, srgan_save)
        cv2.imwrite(bicubic_filename, bicubic_save)
        
        print(f"Saved {srgan_filename} and {bicubic_filename}")
