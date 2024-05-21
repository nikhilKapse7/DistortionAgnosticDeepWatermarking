import torch
from collections import defaultdict
from options import *
from model.hidden import Hidden
from average_meter import AverageMeter
import utils


def calculate_adversarial_loss(adv_images, encoded_images, decoded_messages, original_messages, alpha_adv1, alpha_adv2):
    image_loss = torch.nn.functional.mse_loss(adv_images, encoded_images)
    message_loss = torch.nn.functional.mse_loss(decoded_messages, original_messages)
    return alpha_adv1 * image_loss - alpha_adv2 * message_loss

def train(model: Hidden, device: torch.device, config: HiDDenConfiguration, train_options: TrainingOptions, run_folder: str, tb_logger):
    """
    Trains the model with alternating updates between encoder/decoder and the attack network.
    """
    optimizer_enc = torch.optim.Adam(model.encoder_decoder.encoder.parameters(), lr=1e-3)
    optimizer_dec = torch.optim.Adam(model.encoder_decoder.decoder.parameters(), lr=1e-3)
    optimizer_adv = torch.optim.Adam(model.encoder_decoder.attack_network.parameters(), lr=1e-3)

    train_data, val_data = utils.get_data_loaders(config, train_options)

    for epoch in range(train_options.start_epoch, train_options.number_of_epochs + 1):
        for images, messages in train_data:
            images, messages = images.to(device), messages.to(device)

            # Compute Ien = Fenc(Ico, X)
            encoded_images = model.encoder_decoder.encoder(images, messages)

            # Iterate over adversarial training
            for _ in range(config.num_iter):
                # Compute Iadv = Gadv(Ien)
                adv_images = model.encoder_decoder.attack_network(encoded_images.detach())

                # Update Θadv
                optimizer_adv.zero_grad()
                loss_adv = calculate_adversarial_loss(adv_images, encoded_images, messages, messages, config.alpha_adv1, config.alpha_adv2)
                loss_adv.backward()
                optimizer_adv.step()

                # Update Θdec
                optimizer_dec.zero_grad()
                decoded_messages = model.encoder_decoder.decoder(adv_images)
                loss_dec = torch.nn.functional.mse_loss(decoded_messages, messages)
                (train_options.decoder_loss * loss_dec).backward()
                optimizer_dec.step()

                # Update Θenc
                optimizer_enc.zero_grad()
                loss_enc = torch.nn.functional.mse_loss(encoded_images, images)
                (train_options.encoder_loss * loss_enc).backward()
                optimizer_enc.step()

            print(f"Epoch {epoch}, Loss Encoder: {loss_enc.item()}, Loss Decoder: {loss_dec.item()}, Loss Adversarial: {loss_adv.item()}")
# def train(model: Hidden, train_options: TrainingOptions, device: torch.device):
#     """
#     Trains the model with alternating updates between encoder/decoder and the attack network.
#     """
#     optimizer_enc = torch.optim.Adam(model.encoder_decoder.encoder.parameters(), lr=1e-3)
#     optimizer_dec = torch.optim.Adam(model.encoder_decoder.decoder.parameters(), lr=1e-3)
#     optimizer_adv = torch.optim.Adam(model.encoder_decoder.attack_network.parameters(), lr=1e-3)

#     # Following pseudocode given in the paper
#     for epoch in range(train_options.start_epoch, train_options.number_of_epochs + 1):
#         for images, messages in model.train_dataloader:
#             images, messages = images.to(device), messages.to(device)

#             # Step 3: Compute Ien = Fenc(Ico, X)
#             encoded_images = model.encoder_decoder.encoder(images, messages)

#             # Steps 4-8: Iterate over adversarial training
#             for _ in range(model.config.num_iter):
#                 # Step 5: Compute Iadv = Gadv(Ien)
#                 adv_images = model.encoder_decoder.attack_network(encoded_images.detach())

#                 # Step 6: Update Θadv
#                 optimizer_adv.zero_grad()
#                 loss_adv = model.calculate_adversarial_loss(adv_images, encoded_images, messages)
#                 loss_adv.backward()
#                 optimizer_adv.step()

#                 # Step 7: Update Θdec
#                 optimizer_dec.zero_grad()
#                 decoded_messages = model.encoder_decoder.decoder(adv_images)
#                 loss_dec = torch.nn.functional.mse_loss(decoded_messages, messages)
#                 (train_options.decoder_loss * loss_dec).backward()
#                 optimizer_dec.step()

#                 # Step 8: Update Θenc
#                 optimizer_enc.zero_grad()
#                 loss_enc = torch.nn.functional.mse_loss(encoded_images, images)
#                 (train_options.encoder_loss * loss_enc).backward()
#                 optimizer_enc.step()

#             print(f"Epoch {epoch}, Loss Encoder: {loss_enc.item()}, Loss Decoder: {loss_dec.item()}, Loss Adversarial: {loss_adv.item()}")