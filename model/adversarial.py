## Training Details
# For the network Gadv, we use a two-layer CNN,
# Gadv(I) = Conv3 ◦ Leaky ReLU ◦ Conv16(I).
# We list the hyper-parameters used for training the watermarking model. For our model, we set α_I1= 18.0, α_I2 = 0.01, αM = 0.3, 
# α_adv1 = 15.0, 
# α_adv2 = 1.0, 
# α_advW = 0.2, and num iter = 5. For the HiDDeN combined model and
# identity model, we set α_I1 = 6.0, 
# α_I2 = 0.01, αM = 1.0. The message size for our watermarking model is 120 instead of 30, due to the addition of the channel coding layer. We use the same network architecture as in HiDDeN. Namely, the input
# image Ico is first processed by 4 3 × 3 Conv-BN-ReLU blocks with 64 units per layer. This is then concatenated along the
# channel dimension with an H × W spatial repetition of the input message. The combined blocks are then passed to two
# additional Conv-BN-ReLU blocks to produce the encoded image. For the encoder, we symmetrically pad the input image
# and use ’VALID’ padding for all convolution operations to reduce boundary artifacts of the encoded image. The encoded
# image is clipped to [0, 1] before passing to the decoder. The decoder consists of seven 3 × 3 Conv-BN-ReLU layers of size,
# where the last two layers have stride 2. A global pooling operation followed by a fully-connected layer is used to produce
# the decoded message.
# For both our model and the combined model, the training warm-starts from a pre-trained HiDDeN identity model and
# stops at 250k iterations. We use ADAM with a learning rate of 1e − 3 for all models.
# For the channel model, we use a two fully connected layers with 512 units each, and train with BSC noise where the noise
# strength is uniformly sample from [0, 0.3].

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import os
import pickle

# Adversarial is the attack network
# The encoder, adversarial, and decoder networks are trained jointly
# The setup is as follows:
# Figure 3: Overview of proposed architecture. The input message X is first fed through the channel encoder to produce a redundant message X', which is then combined with the input image I_co to generate the encoded image Ien by the watermark encoder F_enc. The decoder F_dec
# produces a decoded message X'_dec, where it is further processed by the channel decoder to produce the final message X_dec. The attack
# network generates adversarial examples Iadv, which are fed to the image decoder to obtain X'_adv. F_enc, F_dec, are trained on a combination
# of the image loss LI which includes both proximity to the cover image I_co and perceptual quality as in Equation 1, the message loss LM
# as in Equation 2, and the message loss on the decoded adversarial message X'_adv as in Equation 4. The attack network G_adv is trained to
# minimize the adversarial loss L_adv as in Equation 3. The training updates G_adv and the F_enc, F_dec in an alternating fashion.
# Equation 3: 
# L_adv = α_adv1 * ||I_adv - I_en||^2 - α_adv2 * L_m(F_dec(I_adv); X)
# I_adv = G_adv(I_en), adversarial example
# L_m is the message loss which we set as L_2 loss in this paper

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttackNetwork(nn.Module):
    def __init__(self):
        super(AttackNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        return x

def train_attack_network(encoder, decoder, attack_network, dataloader, optimizer, alpha_adv1, alpha_adv2, device):
    attack_network.train()
    
    for cover_images, messages in dataloader:
        cover_images = cover_images.to(device)
        messages = messages.to(device)
        
        optimizer.zero_grad()
        
        # Generate encoded images
        encoded_images = encoder(cover_images, messages)
        
        # Generate adversarial images
        adv_images = attack_network(encoded_images)
        
        # Compute losses
        image_loss = F.mse_loss(adv_images, encoded_images)
        decoded_messages = decoder(adv_images)
        message_loss = F.mse_loss(decoded_messages, messages)
        
        adv_loss = alpha_adv1 * image_loss - alpha_adv2 * message_loss
        adv_loss.backward()
        
        optimizer.step()

# Define hyperparameters
alpha_adv1 = 15.0
alpha_adv2 = 1.0
learning_rate = 0.001

# # Instantiate models and optimizer
