import torch
import os
import copy
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import AutoTokenizer


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def evaluate_loss(net, device, criterion, dataloader, tokenizer):
    net.eval()

    mean_loss = 0
    count = 0

    with torch.no_grad():
        for it, (section_ones, section_two, labels) in enumerate(tqdm(dataloader)):
            encoded_pairs_1 = tokenizer(list(section_ones),
                                        padding='max_length',  # Pad to max_length
                                        truncation=True,  # Truncate to max_length
                                        max_length=128,
                                        return_tensors='pt')  # Return torch.Tensor objects

            encoded_pairs_2 = tokenizer(list(section_two),
                                        padding='max_length',  # Pad to max_length
                                        truncation=True,  # Truncate to max_length
                                        max_length=128,
                                        return_tensors='pt')  # Return torch.Tensor objects

            input_ids_1 = encoded_pairs_1['input_ids'].squeeze(0)  # tensor of token ids
            attn_masks_1 = encoded_pairs_1['attention_mask'].squeeze(
                0)  # binary tensor with "0" for padded values and "1" for the other values
            token_type_ids_1 = encoded_pairs_1['token_type_ids'].squeeze(
                0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

            input_ids_2 = encoded_pairs_2['input_ids'].squeeze(0)  # tensor of token ids
            attn_masks_2 = encoded_pairs_2['attention_mask'].squeeze(
                0)  # binary tensor with "0" for padded values and "1" for the other values
            token_type_ids_2 = encoded_pairs_2['token_type_ids'].squeeze(
                0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

            # Converting to cuda tensors
            input_ids_1, attn_masks_1, token_type_ids_1, labels = input_ids_1.to(device), attn_masks_1.to(
                device), token_type_ids_1.to(device), labels.to(device)
            input_ids_2, attn_masks_2, token_type_ids_2 = input_ids_2.to(device), attn_masks_2.to(
                device), token_type_ids_2.to(device)

            # Enables autocasting for the forward pass (model + loss)

            # Obtaining the logits from the model
            h_i, h_j, z_i, z_j = net(input_ids_1, attn_masks_1, token_type_ids_1, input_ids_2, attn_masks_2,
                                     token_type_ids_2)

            # Computing loss
            mean_loss += criterion(z_i, z_j)
            # seq, attn_masks, token_type_ids, labels = \
            #     seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
            # logits = net(seq, attn_masks, token_type_ids)
            # mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
            count += 1

    return mean_loss / count


def train_bert(net,
               criterion,
               opti,
               lr,
               lr_scheduler,
               train_loader,
               val_loader,
               epochs,
               iters_to_accumulate,
               device,
               bert_model):
    best_loss = np.Inf
    best_ep = 1
    nb_iterations = len(train_loader)
    print_every = nb_iterations // 5  # print the training loss 5 times per epoch
    iters = []
    train_losses = []
    val_losses = []

    scaler = GradScaler()
    tokenizer = AutoTokenizer.from_pretrained(bert_model)

    for ep in range(epochs):
        net.train()
        running_loss = 0.0
        for it, (section_ones, section_two, labels) in enumerate(tqdm(train_loader)):

            encoded_pairs_1 = tokenizer(list(section_ones),
                                          padding='max_length',  # Pad to max_length
                                          truncation=True,  # Truncate to max_length
                                          max_length=128,
                                          return_tensors='pt')  # Return torch.Tensor objects

            encoded_pairs_2 = tokenizer(list(section_two),
                                        padding='max_length',  # Pad to max_length
                                        truncation=True,  # Truncate to max_length
                                        max_length=128,
                                        return_tensors='pt')  # Return torch.Tensor objects

            input_ids_1 = encoded_pairs_1['input_ids'].squeeze(0)  # tensor of token ids
            attn_masks_1 = encoded_pairs_1['attention_mask'].squeeze(
                0)  # binary tensor with "0" for padded values and "1" for the other values
            token_type_ids_1 = encoded_pairs_1['token_type_ids'].squeeze(
                0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

            input_ids_2 = encoded_pairs_2['input_ids'].squeeze(0)  # tensor of token ids
            attn_masks_2 = encoded_pairs_2['attention_mask'].squeeze(
                0)  # binary tensor with "0" for padded values and "1" for the other values
            token_type_ids_2 = encoded_pairs_2['token_type_ids'].squeeze(
                0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

            # Converting to cuda tensors
            input_ids_1, attn_masks_1, token_type_ids_1, labels = input_ids_1.to(device), attn_masks_1.to(device), token_type_ids_1.to(device), labels.to(device)
            input_ids_2, attn_masks_2, token_type_ids_2 = input_ids_2.to(device), attn_masks_2.to(device), token_type_ids_2.to(device)

            # Enables autocasting for the forward pass (model + loss)
            with autocast():
                # Obtaining the logits from the model
                h_i, h_j, z_i, z_j = net(input_ids_1, attn_masks_1, token_type_ids_1, input_ids_2, attn_masks_2, token_type_ids_2)

                # Computing loss
                loss, acc = criterion(z_i, z_j)

                # loss = criterion(logits.squeeze(-1), labels.float())
                loss = loss / iters_to_accumulate  # Normalize the loss because it is averaged

            # Backpropagating the gradients
            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(loss).backward()

            if (it + 1) % iters_to_accumulate == 0:
                # Optimization step
                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, opti.step() is then called,
                # otherwise, opti.step() is skipped.
                scaler.step(opti)
                # Updates the scale for next iteration.
                scaler.update()
                # Adjust the learning rate based on the number of iterations.
                lr_scheduler.step()
                # Clear gradients
                opti.zero_grad()

            running_loss += loss.item()

            if (it + 1) % print_every == 0:  # Print training loss information
                print()
                print("Iteration {}/{} of epoch {} complete. Loss : {}. Accuracy: {}"
                      .format(it + 1, nb_iterations, ep + 1, running_loss / print_every, acc))

                running_loss = 0.0

        val_loss = evaluate_loss(net, device, criterion, val_loader, tokenizer)  # Compute validation loss
        print()
        print("Epoch {} complete! Validation Loss : {}".format(ep + 1, val_loss))

        if val_loss < best_loss:
            print("Best validation loss improved from {} to {}".format(best_loss, val_loss))
            print()
            net_copy = copy.deepcopy(net)  # save a copy of the model
            best_loss = val_loss
            best_ep = ep + 1

    # Saving the model
    if not os.path.exists('models'):
        os.makedirs('models')
    path_to_model = 'models/{}_lr_{}_val_loss_{}_ep_{}.pt'.format(bert_model, lr, round(best_loss.item(), 5), best_ep)
    torch.save(net_copy.state_dict(), path_to_model)
    print("The model has been saved in {}".format(path_to_model))
