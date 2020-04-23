# Imports
# Get our Sentence VAE
from SentenceVAE import SentenceVAE
from SkipVAE import SkipVAE
from FreeBitsVAE import FreeBitsVAE
from DropoutVAE import DropoutVAE

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
    device = config['device']
    epochs = config['num_epochs']
    vocab_size = config['vocab_size']
    embed_size = config['embedding_size']
    hidden_size = config['num_hidden']
    zdim = config['z_dim']
    tokenizer  = config['tokenizer']

    # Instantiate model
    model = SentenceVAE(vocab_size, config, embed_size, hidden_size, zdim) 
    # if(model_type == 'vae'):
    #     model = SentenceVAE(vocab_size, config, z_dim=zdim) 
    # elif(model_type == 'skip'):
    #     model = SkipVAE(vocab_size, config, z_dim=zdim) 
    # elif(model_type == 'free'):
    #     model = FreeBitsVAE(vocab_size, config, z_dim=zdim) 
    # elif(model_type == 'drop'):
    #     model = DropoutVAE(vocab_size, config, z_dim=zdim)
        
    print("Is this still cuda?: ", device)
    model = model.to(device)
    # sample = model.sample(device=device, sampling_strat='rand', tokenizer = tokenizer)
    # print(sample)

    # Optimizer and statistics
    optimizer = Adam(model.parameters())
    train_curve, val_curve = [], []
    train_kl_curve, val_kl_curve = [], []
    print_telbo, print_velbo, print_tkl, print_vkl = [], [], [], []

    for epoch in range(epochs):
        print('Epoch', epoch)
        elbos, KLs = run_epoch(model, (train_loader, valid_loader), optimizer, device) #TODO: Do something with the KLs
        train_elbo, val_elbo = elbos
        train_kl, val_kl = KLs
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        train_kl_curve.append(train_kl)
        val_kl_curve.append(val_kl)
        print("[Epoch {}] train neg elbo: {} train KL: {}, val neg elbo: {} val kl: {}".format(epoch,train_elbo,train_kl,val_elbo,val_kl))
        sample = model.sample(device=device, sampling_strat='rand', tokenizer = tokenizer)
        print(sample)
        print_telbo.append(train_elbo.item())
        print_velbo.append(val_elbo.item())
        print_tkl.append(train_kl.item())
        print_vkl.append(val_kl.item())

    # Save ELBO and KL plot and save the model
    save_plot(train_curve, val_curve, epoch, config)
    save_plot(train_kl_curve, val_kl_curve, epoch, config, True)
    save_model(model, config)
    print("Train ELBO:")
    print(print_telbo)
    print("Validation ELBO:")
    print(print_velbo)
    print("Train KL:")
    print(print_tkl)
    print("Validation KL:")
    print(print_vkl)

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
            batch_elbo, batch_KL = model(inputs.to(device), targets.to(device), torch.tensor(lengths).to(device), device)
            batch_elbo.backward()
            optimizer.step()
            iterations = step
            total_elbo += batch_elbo.detach()
            total_KL += torch.mean(batch_KL.detach())
    else:
        for step, (inputs, targets, lengths) in enumerate(data):
            with torch.no_grad():
                batch_elbo, batch_KL = model(inputs.to(device), targets.to(device), torch.tensor(lengths).to(device), device)
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