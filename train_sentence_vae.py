# Imports
# Get our Sentence VAE
from SentenceVAE import SentenceVAE

# torch-y
import torch
from torch.optim import Adam

# Own classes and helpers
from helpers import save_plot, save_model

#
def train_sentenceVAE(train_loader, 
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
    device = config['device']
    epochs = config['num_epochs']
    zdim = config['z_dim']
    batch_size = config['batch_size']
    vocab_size = config['vocab_size']
    tokenizer  = config['tokenizer']

    # Instantiate model and make it CUTA
    model = SentenceVAE(vocab_size, config, z_dim=zdim) 
    print("Is this still cuda?: ", device)
    model = model.to(device)
    sample = model.sample(device=device, sampling_strat='rand', tokenizer = tokenizer)
    print(sample)

    # Optimizer and statistics
    optimizer = Adam(model.parameters())
    train_curve, val_curve = [], []

    for epoch in range(epochs):
        print('Epoch', epoch)
        elbos = run_epoch(model, (train_loader, valid_loader), optimizer, device)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train neg elbo: {train_elbo} val neg elbo: {val_elbo}")
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
    iterations = 0
    if(model.training):
        for step, (inputs, targets, lengths) in enumerate(data):
            optimizer.zero_grad()
            batch_elbo = model(inputs.to(device), targets.to(device), lengths, device)
            batch_elbo.backward()
            optimizer.step()
            iterations = step
            total_elbo += batch_elbo.detach()
    else:
        for step, (inputs, targets, lengths) in enumerate(data):
            with torch.no_grad():
                batch_elbo = model(inputs.to(device), targets.to(device), lengths, device)
                iterations = step
                total_elbo += batch_elbo.detach()
    average_epoch_elbo = total_elbo/iterations
    return average_epoch_elbo


def run_epoch(model, data, optimizer, device):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer, device)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer, device)
    
    return train_elbo, val_elbo