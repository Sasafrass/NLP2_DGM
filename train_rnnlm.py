# Imports
# System imports
import os
import time
from datetime import datetime

# Numerical imports
import numpy as np

# Torch-y
import torch
import torch.optim as optim

# Own classes
from helpers import generate_text
from RNNLM import RNNLM

def train_rnnlm(config, train_data, valid_data, tokenizer):
    """
    Args:
        config    : Argparse object (turned dictionary) containing all parameters
        train_data: Fold with training data
        valid_data: Fold with validation data
        tokenizer : Object to tokenize words with
    """

    # Initialize the device which to run the model on
    if(torch.cuda.is_available()):
        device = torch.device("cuda")
        print('Device = CUDA')
    else:
        device = torch.device("cpu")
        print('Device = CPU')
        
    #Paths to save the model and optimizer to
    modelpath = config['model_path']
    optimpath = config['optim_path']

    # Initialize the model that we are going to use
    makeNew = config['new_model']

    #Load in model if necessary
    if(not makeNew and modelpath != ""):
        model = (torch.load(modelpath))
    else:
        model = RNNLM(config['vocab_size'],config['num_hidden'],config['num_layers'],device)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(),config['learning_rate'])

    #Load in the optimizer if necessary
    if(not makeNew and optimpath != ""):
        optimizer.load_state_dict(torch.load(optimpath))

    accs = []
    losses = []
    curr_accs = []
    curr_losses = []
    print_steps = []
    convergence = False
    conv_count = 0
    prev_loss = np.inf

    iteration = 0
    epoch = 0
    while(not convergence):
        print("Epoch: " + str(epoch))
        loss = 0
        accuracy = 0
        for step, (batch_inputs, batch_targets, lengths) in enumerate(train_data):
            # Only for time measurement of step through network
            t1 = time.time()
            optimizer.zero_grad()
            targets = batch_targets.to(device)
            out,_ = model.forward(batch_inputs)

            #Calculate loss and accuracy
            curr_loss = 0
            curr_acc = 0
            cor = 0

            seq_length = batch_inputs.shape[1]

            # Get average original lenghts of sentences to divide total loss by
            orig_lengths = torch.mean(torch.argmin(batch_inputs,dim=1).float())

            for i in range(seq_length):
                out_t = out[:,i,:]
                targets_t = targets[:,i]
                curr_loss += criterion.forward(out_t, targets_t)
                preds = (torch.argmax(out_t,dim=1)).long()
                cor += targets_t.eq(preds).sum().item()
            curr_acc = cor/(seq_length * targets.size()[0])
            loss += curr_loss.item()
            accuracy += curr_acc

            curr_loss.backward()
            
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_norm'])

            optimizer.step()
            
            curr_accs.append(curr_acc)
            curr_losses.append(curr_loss.item())

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config['batch_size']/float(t2-t1)
            if(iteration % config['print_every'] == 0):
                loss_std = np.std(curr_losses)
                print("[{}] Epoch {:04d}, Train Step {:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                    "Avg. Accuracy = {:.2f}, Avg. Loss = {:.3f}, Loss STD = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), epoch, iteration,
                        config['batch_size'], examples_per_second,
                        accuracy/config['print_every'], loss/config['print_every'], loss_std
                ))
                accs.append(accuracy/config['print_every'])
                losses.append(loss/config['print_every'])
                print_steps.append(iteration)
                if(np.abs(prev_loss/config['print_every'] - loss/config['print_every']) < 0.001):
                    conv_count += 1
                else:
                    conv_count = 0
                convergence = conv_count == 5
                prev_loss = loss
                accuracy = 0
                loss = 0
            #Generate text
            if iteration % config['sample_every'] == 0:
                model.eval()
                with torch.no_grad():
                    text = generate_text(model,device, tokenizer,config['sample_strat'],config['sample_temp'])
                    print(str(iteration), ": ", text)
                model.train()
                '''                if(modelpath != ""):
                    torch.save(model,modelpath)
                if(optimpath != ""):
                    torch.save(optimizer.state_dict(),optimpath)'''
            iteration += 1

        #TODO: Validate
        model.eval()
        iter = 0
        for step, (batch_inputs, batch_targets, lengths) in enumerate(valid_data):
            targets = batch_targets.to(device)
            out,_ = model.forward(batch_inputs)
            #Calculate loss and accuracy
            curr_loss = 0
            curr_acc = 0
            cor = 0

            seq_length = batch_inputs.shape[1]

            # Get average original lenghts of sentences to divide total loss by
            orig_lengths = torch.mean(torch.argmin(batch_inputs,dim=1).float())

            for i in range(seq_length):
                out_t = out[:,i,:]
                targets_t = targets[:,i]
                curr_loss += criterion.forward(out_t, targets_t)
                preds = (torch.argmax(out_t,dim=1)).long()
                cor += targets_t.eq(preds).sum().item()
            curr_acc = cor/(seq_length * targets.size()[0])
            loss += curr_loss.item()
            accuracy += curr_acc
            curr_accs.append(curr_acc)
            curr_losses.append(curr_loss.item())
            iter += 1
        loss_std = np.std(curr_losses)
        print("Epoch {:04d}, Validation, Avg. Accuracy = {:.2f}, Avg. Loss = {:.3f}, Loss STD = {:.3f}".format(
                epoch, accuracy/iter, loss/iter, loss_std))
        model.train()
        epoch += 1
        if(epoch == config['num_epochs'] or convergence):
            break

    print('Done training.')
    print(accs)
    print(losses)
    print(print_steps)