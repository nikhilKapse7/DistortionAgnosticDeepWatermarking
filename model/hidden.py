import numpy as np
import torch
import torch.nn as nn

from options import HiDDenConfiguration
from model.encoder_decoder import EncoderAttackDecoder
from vgg_loss import VGGLoss
from noise_layers.noiser import Noiser

class Hidden:
    def __init__(self, configuration: HiDDenConfiguration, device: torch.device, noiser: Noiser, tb_logger=None):
        super(Hidden, self).__init__()
        self.config = configuration
        self.device = device
        self.noiser = noiser
        self.tb_logger = tb_logger

        # Initialize the EncoderAttackDecoder
        self.encoder_decoder = EncoderAttackDecoder(configuration, noiser).to(device)

        # Optimizers for different parts of the model
        self.optimizer_enc = torch.optim.Adam(self.encoder_decoder.encoder.parameters(), lr=1e-3)
        self.optimizer_dec = torch.optim.Adam(self.encoder_decoder.decoder.parameters(), lr=1e-3)
        self.optimizer_adv = torch.optim.Adam(self.encoder_decoder.attack_network.parameters(), lr=1e-3)

        # Loss functions
        self.mse_loss = nn.MSELoss().to(device)
        if configuration.use_vgg:
            self.vgg_loss = VGGLoss(3, 1, False).to(device)
        else:
            self.vgg_loss = None

    # def __init__(self, configuration: HiDDenConfiguration, device: torch.device, noiser: Noiser, tb_logger):
    #     """
    #     :param configuration: Configuration for the net, such as the size of the input image, number of channels in the intermediate layers, etc.
    #     :param device: torch.device object, CPU or GPU
    #     :param noiser: Object representing stacked noise layers.
    #     :param tb_logger: Optional TensorboardX logger object, if specified -- enables Tensorboard logging
    #     """
    #     super(Hidden, self).__init__()

    #     self.encoder_decoder = EncoderAttackDecoder(configuration, noiser).to(device)

    #     self.optimizer = torch.optim.Adam(self.encoder_decoder.parameters())

    #     if configuration.use_vgg:
    #         self.vgg_loss = VGGLoss(3, 1, False)
    #         self.vgg_loss.to(device)
    #     else:
    #         self.vgg_loss = None

    #     self.config = configuration
    #     self.device = device

    #     self.mse_loss = nn.MSELoss().to(device)

    #     self.tb_logger = tb_logger
    #     if tb_logger is not None:
    #         from tensorboard_logger import TensorBoardLogger
    #         encoder_final = self.encoder_decoder.encoder._modules['final_layer']
    #         encoder_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/encoder_out'))
    #         decoder_final = self.encoder_decoder.decoder._modules['linear']
    #         decoder_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/decoder_out'))
    
    def train_on_batch(self, batch: list):
        """
        Trains the network on a single batch consisting of images and messages
        :param batch: batch of training data, in the form [images, messages]
        :return: dictionary of error metrics from Encoder, Decoder, and Discriminator on the current batch
        """
        images, messages = batch

        batch_size = images.shape[0]
        self.encoder_decoder.train()
        with torch.enable_grad():
            self.optimizer.zero_grad()

            encoded_images, noised_images, decoded_messages, adv_images, adv_decoded_messages = self.encoder_decoder(images, messages)

            g_loss_enc = self.mse_loss(encoded_images, images)
            g_loss_dec = self.mse_loss(decoded_messages, messages)
            g_loss_adv = self.mse_loss(adv_decoded_messages, messages)

            g_loss = self.config.adversarial_loss * g_loss_adv + self.config.encoder_loss * g_loss_enc \
                     + self.config.decoder_loss * g_loss_dec

            g_loss.backward()
            self.optimizer.step()

        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1])

        losses = {
            'loss           ': g_loss.item(),
            'encoder_mse    ': g_loss_enc.item(),
            'dec_mse        ': g_loss_dec.item(),
            'bitwise-error  ': bitwise_avg_err,
            'adversarial_mse': g_loss_adv.item()
        }
        return losses, (encoded_images, noised_images, decoded_messages, adv_images, adv_decoded_messages)

    def validate_on_batch(self, batch: list):
        """
        Runs validation on a single batch of data consisting of images and messages
        :param batch: batch of validation data, in form [images, messages]
        :return: dictionary of error metrics from Encoder, Decoder, and Discriminator on the current batch
        """
        # if TensorboardX logging is enabled, save some of the tensors.
        if self.tb_logger is not None:
            encoder_final = self.encoder_decoder.encoder._modules['final_layer']
            self.tb_logger.add_tensor('weights/encoder_out', encoder_final.weight)
            decoder_final = self.encoder_decoder.decoder._modules['linear']
            self.tb_logger.add_tensor('weights/decoder_out', decoder_final.weight)

        images, messages = batch

        batch_size = images.shape[0]

        self.encoder_decoder.eval()
        with torch.no_grad():
            encoded_images, noised_images, decoded_messages, adv_images, adv_decoded_messages = self.encoder_decoder(images, messages)

            g_loss_enc = self.mse_loss(encoded_images, images)
            g_loss_dec = self.mse_loss(decoded_messages, messages)
            g_loss_adv = self.mse_loss(adv_decoded_messages, messages)

            g_loss = self.config.adversarial_loss * g_loss_adv + self.config.encoder_loss * g_loss_enc \
                    + self.config.decoder_loss * g_loss_dec

        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1])

        losses = {
            'loss           ': g_loss.item(),
            'encoder_mse    ': g_loss_enc.item(),
            'dec_mse        ': g_loss_dec.item(),
            'bitwise-error  ': bitwise_avg_err,
            'adversarial_mse': g_loss_adv.item()
        }
        return losses, (encoded_images, noised_images, decoded_messages, adv_images, adv_decoded_messages)

    def to_stirng(self):
        # return '{}\n{}'.format(str(self.encoder_decoder), str(self.discriminator))
        return '{}'.format(str(self.encoder_decoder))
    
    def calculate_adversarial_loss(self, adv_images, encoded_images, decoded_messages, original_messages):
        """
        Calculate the adversarial loss as described in the paper.
        """
        image_loss = self.mse_loss(adv_images, encoded_images)
        message_loss = self.mse_loss(decoded_messages, original_messages)
        return self.config.adversarial_loss * (image_loss - message_loss)

    # def forward(self, images, messages):
        """
        Forward pass through the encoder, decoder, and adversarial network.
        """
        encoded_images = self.encoder_decoder.encoder(images, messages)
        noised_images = self.encoder_decoder.noiser(encoded_images)
        decoded_messages = self.encoder_decoder.decoder(noised_images)
        adv_images = self.encoder_decoder.attack_network(encoded_images)
        adv_decoded_messages = self.encoder_decoder.decoder(adv_images)

        return encoded_images, noised_images, decoded_messages, adv_images, adv_decoded_messages