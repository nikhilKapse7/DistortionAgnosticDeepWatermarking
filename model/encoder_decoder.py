import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder
from options import HiDDenConfiguration
from noise_layers.noiser import Noiser


import torch.nn as nn
import torch
from model.encoder import Encoder
from model.decoder import Decoder
from model.adversarial import AttackNetwork
from options import HiDDenConfiguration
from noise_layers.noiser import Noiser

class EncoderAttackDecoder(nn.Module):
    """
    Combines Encoder->Noiser->Decoder->Adversarial into a single pipeline.
    The input is the cover image and the watermark message. The module inserts the watermark into the image
    (obtaining encoded_image), then applies Noise layers (obtaining noised_image), then passes the noised_image
    to the Decoder which tries to recover the watermark (called decoded_message). The module outputs
    a four-tuple: (encoded_image, noised_image, decoded_message, adv_decoded_message)
    """
    def __init__(self, config: HiDDenConfiguration, noiser: Noiser):
        super(EncoderAttackDecoder, self).__init__()
        self.encoder = Encoder(config)
        self.noiser = noiser
        self.decoder = Decoder(config)
        self.attack_network = AttackNetwork()

    def forward(self, image, message):
        # Encode the image with the watermark
        encoded_image = self.encoder(image, message)
        
        # Apply noise to the encoded image
        noised_and_cover = self.noiser([encoded_image, image])
        noised_image = noised_and_cover[0]
        
        # Decode the noised image to recover the message
        decoded_message = self.decoder(noised_image)
        
        # Generate adversarial examples from the encoded image
        adv_images = self.attack_network(encoded_image)
        
        # Decode the adversarial images to recover the message
        adv_decoded_message = self.decoder(adv_images)
        
        return encoded_image, noised_image, decoded_message, adv_images, adv_decoded_message
