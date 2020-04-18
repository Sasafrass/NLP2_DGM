import argparse

import torch
import torch.nn as nn
import torch.nn.functional
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
import os 
import math
from scipy.stats import norm
import numpy as np 

from datasets.bmnist import bmnist


class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        # Initialize Linear Layers
        self.linear1 = nn.Linear(784, hidden_dim)
        self.linearmu = nn.Linear(hidden_dim, z_dim)
        self.linearsig = nn.Linear(hidden_dim, z_dim)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ht = torch.nn.functional.tanh(self.linear1(input)).to(device)
        mean, std = self.linearmu(ht), self.linearsig(ht).to(device)

        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        # Initialize linear layers
        self.linear1 = nn.Linear(z_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 784)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
        # First run one hidden layer, then get means output
        ht = torch.nn.functional.relu(self.linear1(input))
        mean = torch.sigmoid(self.linear2(ht))

        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        # Retrieve mean and std from encoder
        mean, std = self.encoder.forward(input)

        # Create Z using Reparametrization trick and interpreting it as logvar as in original paper
        z = torch.randn_like(std) * torch.exp(0.5 * std) + mean

        # Decode Z
        pred = self.decoder.forward(z)

        # Calculate kl divergence and 'loss'
        # KL closed form follows from the original paper 
        kl = torch.sum(1 + std - mean ** 2 - torch.exp(std)) * (-1/2)

        loss = torch.nn.functional.binary_cross_entropy(pred, input, reduction='sum')

        # Average over the batch size
        average_negative_elbo = (kl + loss) / input.shape[0]

        # Return the elbo
        #print("avg_neg_elbo: ", average_negative_elbo)
        return average_negative_elbo

    def sample(self, n_samples = None, smooth = False):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        # with torch.no_grad():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Get random z sample
        z = torch.randn(n_samples, self.z_dim).to(device)

        # Retrieve bernoulli means of forward pass 
        means = self.decoder.forward(z)

        # Sample from the means again with torch.bernoulli
        # If smooth, we use the actual mean values instead
        if not smooth:
            sampled_ims = torch.bernoulli(means)
            im_means = means
        if smooth:
            sampled_ims = means
            im_means = means

        return sampled_ims, im_means


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    # Use right device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    avg_elbo = 0

    # Main loop
    for imagebatch in data:

        # Reshape the imagebatch to a vector for each image 
        imagebatch = imagebatch.to(device)
        imagebatch = imagebatch.view(-1,784)

        if not model.training:
            # Not training
            elbo = model.forward(imagebatch)
        else:
            # Zero gradient buffer, calculate elbo
            optimizer.zero_grad()
            elbo = model.forward(imagebatch)

            # Backward the elbo and update with backprop
            elbo.backward()
            optimizer.step()

        # Update avg elbo 
        avg_elbo += elbo.item()
            
    average_epoch_elbo = avg_elbo / len(data)

    return average_epoch_elbo


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def main():
    data = bmnist()[:2]  # ignore test split
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = VAE(z_dim=ARGS.zdim).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------
        
        # number and square root of samples for rows
        num_samples = 25
        samples_sqrt = int(math.sqrt(num_samples))

        # Sample from the model
        samples, means = model.sample(n_samples = num_samples)
        samples = samples.view(25, 1, 28, 28)
        save_image(samples, "img/vae5/test-" + str(epoch) + ".png",
                nrow=samples_sqrt, normalize=True
        )

    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------
    

    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')

def plot_manifold(model):
    with torch.no_grad():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        num_row = 20
        grid = torch.linspace(0, 1, num_row)
        samples = [torch.erfinv(2 * torch.tensor([x, y]) - 1) * np.sqrt(2) for x in grid for y in grid]
        samples = torch.stack(samples).to(device)
        manifold = model.forward(samples).view(-1, 1, 28, 28)
        image = make_grid(manifold, nrow = num_row)
        plt.imsave("manifold.png", image.cpu().numpy().transpose(1,2, 0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()

    main()