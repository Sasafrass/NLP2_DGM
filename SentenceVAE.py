# Numerical / stats stuff
import scipy.stats
import numpy as np

# torch-y
import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

class Encoder(nn.Module):
    """
    Encoder module part of a VAE

    Returns:
        mean: Means of size z_dim for approximate posterior 
        std : Standard deviations of size z_dim for approximate posterior
    """

    def __init__(self, vocab_size, embed_size, hidden_dim, z_dim, bidir):
        super().__init__()

        #TODO: Implement word dropout
        #Should this embedding be the same as in the decoder?
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_dim, bidirectional=bidir) #TODO: Make this bidirectional
        self.mean_lin = nn.Linear(hidden_dim, z_dim)
        self.std_lin = nn.Linear(hidden_dim, z_dim)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim].
        """

        mean, std = None, None
        embedding = self.embed(input)
        
        #Push input through a non-linearity
        embedding = embedding.permute(1,0,2)
        out, hidden = self.gru(embedding)

        #Then transform to latent space
        mean = self.mean_lin(hidden)
        std = F.softplus(self.std_lin(hidden))

        return mean, std


class Decoder(nn.Module):
    """
    Decoder module for our VAE

    Args:
        Input : Input, usually in batch
        Hidden: Previous hidden state

    Returns:
        Out   : Output for current time step
        Hidden: Hidden state for current time step
    """

    def __init__(self, vocab_size, embed_size, z_dim, bidir, config):
        super().__init__()
        self.num_hidden = config.num_hidden
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.gru = nn.GRU(embed_size,z_dim,batch_first=True, bidirectional=bidir)
        self.linear = nn.Linear(z_dim, self.num_hidden)
        self.output = nn.Linear(z_dim,vocab_size)

    def forward(self, input, hidden=None):
        embedding = self.embed(input)
        
        if(hidden == None):
            out,hidden = self.gru.forward(embedding)
        else:
            out,hidden = self.gru.forward(embedding,hidden)
        out = self.output(out)

        return out, hidden

class SentenceVAE(nn.Module):
    """
    Full SentenceVAE model incorporating encoder and decoder

    Args:
        Input: Input from data loader, usually in batches
        Targets: Targets from data loader, usually in batches
        Lengths: Lengths of the original sentences

    Returns:
        average_negative_elbo: This is the average negative elbo 
    """

    def __init__(self, vocab_size, config, embed_size=464, hidden_dim=191, z_dim=13, bidir=False):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.z_dim = z_dim
        self.num_dirs = bidir + 1
        self.encoder = Encoder(vocab_size, embed_size, hidden_dim, z_dim, bidir)
        self.decoder = Decoder(vocab_size, embed_size, z_dim, bidir, config)

    def forward(self, input, targets, lengths, device):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        batch_size = input.shape[0]
        seq_len = input.shape[1]

        average_negative_elbo = None
        mean, std = self.encoder(input)
        
        #Reparameterization trick
        q_z = Normal(mean,std)
        sample_z = q_z.rsample()

        px_logits, _ = self.decoder(input,sample_z)
        p_x = Categorical(logits=px_logits)
        
        prior = Normal(torch.zeros(self.z_dim).to(device),torch.ones(self.z_dim).to(device))
        
        KLD = distributions.kl_divergence(q_z, prior)

        criterion =  nn.CrossEntropyLoss(ignore_index=0)
        recon_loss = criterion(p_x.logits.view(batch_size*seq_len,-1),targets.view(-1))
        average_negative_elbo = torch.sum(torch.mean(KLD,dim=0)) + recon_loss
        
        return average_negative_elbo


    def sample(self, tokenizer, device, sampling_strat='max', temperature=1, starting_text=[1]):
        """
        Function that allows us to sample a new sentence for the VAE
        """
        assert sampling_strat in ('max', 'rand')

        # Start with encoded text
        start = np.array(starting_text)
        text = list(start) #This stores the eventual output
        current = torch.from_numpy(start).long().view(1,-1)
        q_z = Normal(torch.zeros(self.z_dim),torch.ones(self.z_dim))
        sample_z = q_z.rsample().view(self.num_dirs,1,-1).to(device)

        #The initial step
        input = current.to(device)
        output,hidden = self.decoder(input, sample_z)
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
            output,hidden = self.decoder(input,hidden)
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

        # 
        return tokenizer.decode(text)