# Imports
# System imports
import os
import time

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
        model = RNNLM(config['vocab_size'],config['embedding_size'],config['num_hidden']).to(device)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0,reduction='sum')
    optimizer = optim.Adam(model.parameters(),config['learning_rate'])

    #Load in the optimizer if necessary
    if(not makeNew and optimpath != ""):
        optimizer.load_state_dict(torch.load(optimpath))

    losses = []

    for epoch in range(config['num_epochs']):
        print("Epoch: " + str(epoch))
        loss = 0
        model.train()
        for step, (batch_inputs, batch_targets, _) in enumerate(train_data):
            optimizer.zero_grad()
            curr_loss = calc_loss(model, criterion, batch_inputs, batch_targets, device)
            loss += curr_loss.item()
            curr_loss.backward()
            optimizer.step()

        print("Epoch {:04d}, Batch Size = {}, Avg. Loss = {:.3f}".format(epoch, config['batch_size'], loss/step))
        losses.append(loss/step)
        loss = 0

        #Generate text
        model.eval()
        with torch.no_grad():
            text = generate_text(model,device, tokenizer,config['sample_strat'],config['sample_temp'])
            print(text)

        '''                if(modelpath != ""):
            torch.save(model,modelpath)
        if(optimpath != ""):
            torch.save(optimizer.state_dict(),optimpath)'''

        for step, (batch_inputs, batch_targets, _) in enumerate(valid_data):
            curr_loss = calc_loss(model, criterion, batch_inputs, batch_targets, device)
            loss += curr_loss.item()

        print("Epoch {:04d}, Validation, Avg. Loss = {:.3f}".format(epoch, loss/step))

    print('Done training.')
    print(losses)

def calc_loss(model, criterion, batch_inputs, batch_targets, device):
    targets = batch_targets.to(device)
    out,_ = model(batch_inputs.to(device))

    seq_length = batch_inputs.shape[1]
    batch_size = batch_inputs.shape[0]
    curr_loss = criterion(out.view(batch_size*seq_length,-1),targets.view(-1))
    curr_loss /= batch_size
    return curr_loss

def calc_loss_old(model, criterion, batch_inputs, batch_targets, device):
    targets = batch_targets.to(device)
    out,_ = model(batch_inputs.to(device))

    curr_loss = 0
    seq_length = batch_inputs.shape[1]

    for i in range(seq_length):
        out_t = out[:,i,:]
        targets_t = targets[:,i]
        curr_loss += criterion(out_t, targets_t)
    return curr_loss