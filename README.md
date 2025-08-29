# SRGAN-for-Image-Super-Resolution-PyTorch-with-Bicubic-SRCNN-Baselines
---

**1) Overview**

Super‑resolution (SR) enhances the spatial resolution of low‑quality images. While CNNs have advanced SR, large upscales (e.g., ×4) still struggle with fine textures and high‑frequency detail. In this project, we implement SRGAN and compare it to two baselines: bicubic interpolation and SRCNN. We report PSNR and SSIM, and emphasize perceptual quality enabled by adversarial training.

Implemented: SRGAN Generator/Discriminator from scratch in PyTorch

Baselines: Bicubic (OpenCV), SRCNN (adapted from open‐source code) ( https://github.com/AhmedIbrahimai/Super-Resolution-image-using-CNN-in-PyTorch )

Datasets: DIV2K (train/eval)

Metrics: PSNR, SSIM (skimage); perceptual comparison (visuals)<br>

---

**2) Repo Structure**

      ├── archive2/ # place dataset here
      │ ├── HR_images/ 
      │ └── LR_images/
      ├── outputs/
      │ ├── bicubic/
      │ └── srgan/
      ├── outputs_srcnn/
      ├── srcnn/
      │ └── images/
      │ │ ├── test/
      │ │ ├── train/
      │ │ └── val/
      ├── srcnn.py
      ├── srgan.py

---

**3) Setup & Requirements**

Python: 3.9+

PyTorch: 2.0+ (with CUDA if available)

Torchvision, opencv‑python, scikit‑image, Pillow, numpy, tqdm, matplotlib<br>

---

**4) Quick Start**


**4.1 Data prep**
    Place DIV2K under ./archive2. By default we use HR 128×128 crops and generate LR 32×32 pairs on‑the‑fly; values are mapped from [0,1] → [−1,1] to match Tanh.<br>


**4.2 Training**
    Phase A — Pretraining (MSE only, generator).
    Phase B — Fine‑tuning (GAN + perceptual + TV + pixel).<br>


---

**5) Architecture**

**5.1 Generator (deep residual network)**

Head: k9n64s1 conv + PReLU

16 residual blocks, each: k3n64s1 → BN → PReLU → k3n64s1 → BN → skip add

Tail: k3n64s1 conv → long skip from head features

Upsampling: two stages of k3n256s1 conv + PixelShuffle×2 + PReLU (→ 64 ch)

Output: k9n3s1 conv → Tanh (RGB in [−1,1])<br>


**5.2 Residual Block**

Learns residuals (high‑freq detail) w.r.t. its input via a skip connection, enabling deeper networks without vanishing gradients and focusing capacity on fine textures.<br>


**5.3 Discriminator**

Eight conv layers with alternating stride‑2 downsampling (e.g., k3n64s1, k3n64s2, k3n128s1, k3n128s2, …), then 2 FC layers → sigmoid (real/fake). Trained adversarially to push the generator toward realistic texture synthesis.<br>


**5.4 disc_block helper**

Conv → BN → LeakyReLU, the basic downsampling unit that grows channels while halving spatial dims.<br>


**5.5 Losses**

Pixel loss

Perceptual loss

Adversarial loss

Total Variation (TV)

Default weights used in this project: w_tv = 1e−6, w_vgg = 6e−3, w_adv = 1e−3.<br>

---

**6) Training Strategy**

**6.1 Phase A — Pretraining**

Train only the generator on MSE to learn stable upscaling (good but smooth results). Save checkpoints every epoch.<br>


**6.2 Phase B — Fine‑tuning**

Alternate D and G updates: update D on real HR and generated HR; update G on a weighted sum of pixel + perceptual + adversarial + TV losses. Over epochs, G learns sharper textures while staying close to ground truth.<br>

---

**7) Data Processing**

HR crops resized to 128×128, LR to 32×32 (for ×4)

Map tensors [0,1] → [−1,1]

Custom SRDataset wraps ImageFolder and returns aligned LR/HR pairs<br>

---

**8) User‑Adjustable Parameters**

            **Parameter	      | Purpose	                  | Default (example)**
            epochs_pretrain	| G warm‑up with MSE	      | 5
            epochs_finetune	| Full GAN fine‑tune	      | 3
            batch_size	      | Memory/grad smoothness	| 6
            res_blocks	      | Generator capacity	      | 16
            lr	            | Optimizer step	            | 1e−4
            scale	            | Upscaling factor	      | 4
            w_adv	            | Adversarial weight	      | 1e−3
            w_vgg	            | Perceptual weight	      | 6e−3
            w_tv	            | TV weight	                  | 1e−6

---

**9) Example Inputs and Outputs**
      
**9.1 Qualitative — Example Outputs (×4)**

In the four example outputs below, you can see how parameter choices directly shape the results:
- Example 1: The super-resolved image has an unnatural green cast compared to both the input and the ground truth. This color shift usually happens when the adversarial or perceptual loss weight is set too high, pulling the generator away from accurate color reproduction.<br>
      
- Example 2: Here the output looks blurry, and we can spot ringing artifacts around edges. That typically means the number of training epochs wasn’t well matched, either too few epochs for the generator to learn fine details, or too many, causing overfitting and oscillations.<br>
      
- Example 3: This image is overly sharp in places, with tiny bright spots and color oversaturation. In practice, such “too-sharp” artifacts arise when the adversarial loss is weighted too strongly, encouraging the network to hallucinate textures beyond what’s realistic.<br>
      
- Example 4: Finally, we have a well-balanced result with sharp edges, natural colors,and no visible artifacts. This shows how the right combination of learning rate, loss weights, and epoch count can produce the most realistic output.
These examples underscore that there’s no one size that fits all setting. Different images have different characteristics, so it’s important to understand our data and adjust hyperparametersaccordingly.<br>


Example 1:

<img width="475" height="299" alt="srgan" src="https://github.com/user-attachments/assets/dee2b24d-2ad1-4540-881d-30eaa8f4e9c0" /><br>


Example 2:

<img width="473" height="293" alt="srgan2" src="https://github.com/user-attachments/assets/2eb55478-11a9-4e7f-b84c-e95507bfeaf7" /><br>


Example 3:

<img width="507" height="309" alt="srgan3" src="https://github.com/user-attachments/assets/a3338944-8753-4ce2-a636-63ef853a1061" /><br>


Example 4:

<img width="480" height="298" alt="srgan4" src="https://github.com/user-attachments/assets/200399e7-dad1-4111-9976-8e5cd22dda91" /><br>



**9.2 Quantitative — PSNR / SSIM (DIV2K eval)**
   Results with Pretraining epoch = 5, Fine-tuning epoch = 3:
   
    Method	PSNR (dB)	SSIM
    Bicubic	33.20	0.8672
    SRCNN	24.7072	0.8082
    SRGAN	28.88	0.8316

Output of SRGAN model (Pretraining epoch = 5, Fine-tuning epoch = 3):

<img width="481" height="376" alt="srgan5" src="https://github.com/user-attachments/assets/5a853d8e-7f1a-4b81-b409-8b0ceb3a54fc" /><br>


Output of SRCNN model (epoch=3):

<img width="491" height="349" alt="srcnn" src="https://github.com/user-attachments/assets/c599f29f-ff46-4a0d-b2ec-0425ab1a8699" /><br>

---

**10) Conclusion**

SRGAN delivers sharper, more realistic ×4 super-resolution than bicubic and an improved SRCNN. In our tests it gained \~4 dB PSNR and +0.023 SSIM over SRCNN, but training was \~10× slower and required careful tuning. Use SRGAN when perceptual quality matters; choose simpler models when compute is limited.<br>

---

**11) Acknowledgments & References**

[1] Ledig, Christian, et al. "Photo-realistic single image super-resolution using a generative adversarial network." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017. 

[2] Dong, Chao, et al. "Learning a deep convolutional network for image super resolution." Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part IV 13. Springer International Publishing, 2014.

[3] https://github.com/AhmedIbrahimai/Super-Resolution-image-using-CNN-in-PyTorch 
