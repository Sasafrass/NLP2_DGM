# Imports
# Get our Sentence VAE
from SentenceVAE import SentenceVAE
from SkipVAE import SkipVAE
from FreeBitsVAE import FreeBitsVAE

# torch-y
import torch
from torch.optim import Adam

# Own classes and helpers
from helpers import save_plot, save_model

#
def train_VAE(train_loader, 
                      valid_loader, 
                      test_loader,
                      config):
    """
    Function to train our Sentence VAE

    Args:
        train_loader: Loader for training data
        valid_loader: Loader for validation data
        test_loader : Loader for testing data
        config      : Dictionary containing all parameters
    """

    # Get necessary parameters from config
    model_type = config['model']
    device = config['device']
    epochs = config['num_epochs']
    zdim = config['z_dim']
    batch_size = config['batch_size']
    vocab_size = config['vocab_size']
    tokenizer  = config['tokenizer']

    # Instantiate model and make it CUTA
    if(model_type == 'vae'):
        model = SentenceVAE(vocab_size, config, z_dim=zdim) 
    elif(model_type == 'skip'):
        model = SkipVAE(vocab_size, config, z_dim=zdim) 
    elif(model_type == 'free'):
        model = FreeBitsVAE(vocab_size, config, z_dim=zdim) 
        
    print("Is this still cuda?: ", device)
    model = model.to(device)
    sample = model.sample(device=device, sampling_strat='rand', tokenizer = tokenizer)
    print(sample)

    # Optimizer and statistics
    optimizer = Adam(model.parameters())
    train_curve, val_curve = [], []

    for epoch in range(epochs):
        print('Epoch', epoch)
        elbos, KLs = run_epoch(model, (train_loader, valid_loader), optimizer, device) #TODO: Do something with the KLs
        train_elbo, val_elbo = elbos
        train_kl, val_kl = KLs
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print("[Epoch {}] train neg elbo: {} train KL: {}, val neg elbo: {} val kl: {}".format(epoch,train_elbo,train_kl,val_elbo,val_kl))
        sample = model.sample(device=device, sampling_strat='rand', tokenizer = tokenizer)
        print(sample)

    # Save ELBO plot and save the model
    save_plot(train_curve, val_curve, epoch, config)
    save_model(model, config)

def epoch_iter(model, data, optimizer, device):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    average_epoch_elbo = None
    total_elbo = 0
    total_KL = 0
    iterations = 0
    if(model.training):
        for step, (inputs, targets, lengths) in enumerate(data):
            optimizer.zero_grad()
            batch_elbo, batch_KL = model(inputs.to(device), targets.to(device), lengths, device)
            batch_elbo.backward()
            optimizer.step()
            iterations = step
            total_elbo += batch_elbo.detach()
            total_KL += torch.mean(batch_KL.detach())
    else:
        for step, (inputs, targets, lengths) in enumerate(data):
            with torch.no_grad():
                batch_elbo, batch_KL = model(inputs.to(device), targets.to(device), lengths, device)
                iterations = step
                total_elbo += batch_elbo.detach()
                total_KL += torch.mean(batch_KL.detach())
    average_epoch_elbo = total_elbo/iterations
    average_epoch_KL = total_KL/iterations
    return average_epoch_elbo, average_epoch_KL

def run_epoch(model, data, optimizer, device):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo, train_KL = epoch_iter(model, traindata, optimizer, device)

    model.eval()
    val_elbo, val_KL = epoch_iter(model, valdata, optimizer, device)
    
    return (train_elbo, val_elbo), (train_KL, val_KL)