# System stuff
import os

# Numerical imports
import numpy as np

# torch-y
import torch
import torch.nn.functional as F

# Visualisation
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# To save the elbo plot
def save_plot(train_curve, val_curve, epoch, config, KL=False):
    """
    Saves the training plot and validation plot in one figure

    Args:
        train_curve: List containing all training ELBO values
        val_curve  : List containing all validation ELBO values
        epoch      : Last epoch number, so we can get a range for plotting
    """

    # Checks whether the img folder exists - creates one if not
    if not os.path.exists(config['img_path']):
        os.makedirs(config['img_path'])

    # Get better style
    plt.style.use('ggplot')

    # Generate horizontal subplot grid
    fig, ax = plt.subplots(1,2, figsize = (14,4))
    x = np.arange(len(train_curve)) + 1

    # left plot - training
    ax[0].plot(x, train_curve, 'g-')
    if(KL):
        ax[0].set_title("Training KL")
        ax[0].set_xlabel("Number of epochs")
        ax[0].set_ylabel("KL")
    else:
        ax[0].set_title("Training ELBO")
        ax[0].set_xlabel("Number of epochs")
        ax[0].set_ylabel("ELBO")
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # right plot - validation
    ax[1].plot(x, val_curve)
    if(KL):
        ax[1].set_title("Validation KL")
        ax[1].set_xlabel("Number of epochs")
        ax[1].set_ylabel("KL")
    else:
        ax[1].set_title("Validation ELBO")
        ax[1].set_xlabel("Number of epochs")
        ax[1].set_ylabel("ELBO")
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    # Save, OS indifferent
    if(KL):
        name = "{}_{}_{}_{}_{}_KL.pdf".format(
            config['model'],
            config['num_epochs'],
            config['num_hidden'],
            config['dropout'],
            config['learning_rate']
        )
    else:
        name = "{}_{}_{}_{}_{}_ELBO.pdf".format(
            config['model'],
            config['num_epochs'],
            config['num_hidden'],
            config['dropout'],
            config['learning_rate']
        )
    fig.savefig(os.path.join(config['img_path'],name))


# Does some saving of the model
def save_model(model, config):
    """
    Function that saves a model to the desired location
    """

    # Check whether location exists - if not, create
    if not os.path.exists(config['save_path']):
        os.makedirs(config['save_path'])

    # Get OS indifferent path
    name = "{}_{}_{}_{}_{}.pth".format(
        config['model'],
        config['num_epochs'],
        config['num_hidden'],
        config['dropout'],
        config['learning_rate']
    )
    location = os.path.join(config['save_path'], name)
    torch.save(model.state_dict(), location)

# Function to generate text
def generate_text(model,device,tokenizer,sampling_strat='max',temperature=1, starting_text=[1],ksize=1):
    """
    Function allowing us to generate text for RNNLM

    Returns:
        Tokenized version of the decoding of generated text
    """
    assert sampling_strat in ('max', 'rand')
    
    # Start with encoded text
    start = np.array(starting_text)
    text = list(start) #This stores the eventual output
    current = torch.from_numpy(start).long().unsqueeze(dim=0) 

    #The initial step
    input = current.to(device)
    output,hidden = model.forward(input)
    current = output[0,-1,:].squeeze()
    if(sampling_strat == 'max'):
        guess = torch.argmax(current).unsqueeze(0)
    elif(sampling_strat == 'rand'):
        guess = torch.multinomial(F.softmax(temperature*current,dim=0),1)
    text.append(guess.item())
    input = guess.unsqueeze(0)

    #Now that we have an h and c, we can start the loop
    i = 0
    while(i < 100):
        output,hidden = model.forward(input,hidden)
        current = output.squeeze()
        if(sampling_strat == 'max'):
            guess = torch.argmax(current).unsqueeze(0)
        elif(sampling_strat == 'rand'):
            guess = torch.multinomial(F.softmax(temperature*current,dim=0),1)
        text.append(guess.item())
        input = guess.unsqueeze(0)
        i += 1
        if(guess.item() == 2):
            break

    return tokenizer.decode(text)