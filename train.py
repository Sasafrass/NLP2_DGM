# imports
import numpy as np
import pandas as pd
import tkinter

# Preprocessing
from preprocessing import AFFRDataset, get_data, padded_collate

# Get our own classes for models
from RNNLM import RNNLM
from train_rnnlm import train_rnnlm
from train_vae import train_VAE

# All things torch-y
import torch
from torch.utils.data import DataLoader

# To parse dem arguments
import argparse

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("You're running on:", device)

# For correct argument parsing
def str2bool(arg):
    if isinstance(arg, bool):
       return arg
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Get datasets
print("Preparing data and tokenizer...")
train_data, validation_data, test_data, tokenizer = get_data()

# Initialize argument parser
parser = argparse.ArgumentParser()

# Model selection, device selection
parser.add_argument('--model', type=str, default="dropout-vae",
                    help='Select model to use')
parser.add_argument('--device', type=str, default=device,
                    help='Select which device to use')

# Standard model parameters
parser.add_argument('--learning_rate', type=float, default=2e-3,
                    help='Learning rate')
parser.add_argument('--num_epochs', type=int, default=2,
                    help='Number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=64,
                    help='The batch size of our model')
parser.add_argument('--vocab_size', type=int, default=tokenizer.vocab_size,
                    help='Size of the vocabulary')
parser.add_argument('--learning_rate_decay', type=int, default=0.96,
                    help='Learning rate decay')

# GRU Parameters
parser.add_argument('--num_hidden', type=int, default=128,
                    help='Number of hidden units in selected LSTM model')
parser.add_argument('--num_layers', type=int, default=1,
                    help='Number of layers')

# VAE Parameters
parser.add_argument('--z_dim', type=int, default=13,
                    help='Latent space dimension')

parser.add_argument('--dropout', type=float, default=1,
                    help='Probability an input is dropped')
parser.add_argument('--word_dropout', type=str2bool, default=False,
                    help='Flag to use word dropout or not')

# Paths
parser.add_argument('--save_path', type=str, default="models",
                    help='Select where to save the model')
parser.add_argument('--load_path', type=str, default="models",
                    help='Select from where to load the model')
parser.add_argument('--model_name', type=str, default="test",
                    help='Define a model name')
parser.add_argument('--model_path', type=str, default="models/trump_model.txt",
                    help='Select from where to load the model')
parser.add_argument('--optim_path', type=str, default="models/trump_optim.txt",
                    help='Select from where to load the model')
parser.add_argument('--img_path', type=str, default="img",
                    help='Select from where to load the model')

# Model saving
parser.add_argument('--new_model', type=bool, default=True,
                    help='Select from where to load the model')

# Printing and sampling
parser.add_argument('--print_every', type=int, default=100,
                    help='Number of iterations before we print performance')

parser.add_argument('--sample_every', type=int, default=100,
                    help='Number of iterations after which we sample a new sequence')

parser.add_argument('--sample_strat', type=str, default='rand',
                    help='Select the sampling strategy to use')

parser.add_argument('--sample_temp', type=int, default=1.5,
                    help='Sampling temperature vs greedy sampling')

# Parse the arguments, get dictionary and add tokenizer
args = parser.parse_args()
config = vars(args)
config['tokenizer'] = tokenizer
print("Word dropout: ", config['word_dropout'])

# Make trainloaders
train_data = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, collate_fn=padded_collate)
valid_data = DataLoader(validation_data, batch_size=config['batch_size'], shuffle=False, collate_fn=padded_collate)
test_data  = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False, collate_fn=padded_collate)

# Run
print(config['model'])
if config['model'] in ("rnnlm", "RNNLM", "RNNlm", "rnnLM"):
    print("Training RNNLM now")
    train_rnnlm(config, train_data, validation_data, tokenizer) 
elif config['model'] in ("VAE", "Vae", "vae"):
    config['model'] = 'vae'
    print("Training VAE now")
    train_VAE(train_data, valid_data, test_data, config)
elif config['model'] in ("Dropout-VAE, dropout-vae, dropout-VAE, DROPOUT-VAE"):
    config['model'] = 'drop'
    print('Training Dropout-VAE now')
    train_VAE(train_data, valid_data, test_data, config)
elif config['model'] in ('FREEBITS-VAE, FreeBitsVAE, FreeBitsVae, FreeBits-VAE, FreeBits-Vae, FreeBits-vae, freebits-vae'):
    config['model'] = 'free'
    print("Training FreeBits-VAE now")
    train_VAE(train_data, valid_data, test_data, config)
elif config['model'] in ("SKIP-VAE", "Skip-VAE", "Skip-Vae", "Skip-vae", "skip-VAE", "skip-Vae", "skip-vae"):
    config['model'] = 'skip'
    print("Training Skip-VAE now")
    train_VAE(train_data, valid_data, test_data, config)
else:
    raise ValueError("Please choose SKIP-VAE, FreeBits-VAE, VAE or RNNLM")