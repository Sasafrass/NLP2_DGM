import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

def perplexity(model, data, device):
    model.eval()
    total_per = 0
    for step, (input, targets, lenghts) in enumerate(data):
        input = input.to(device)
        targets = targets.to(device)
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        lenghts = torch.tensor(lenghts).to(device).float()
        mean, std = model.encoder(input)
        
        #Reparameterization trick
        q_z = Normal(mean,std)
        sample_z = q_z.rsample()

        h_0 = torch.tanh(model.upscale(sample_z)).unsqueeze(0)

        if(model.skip):
            z = model.z_lin(sample_z).unsqueeze(1)
            px_logits, _ = model.skip_decoder(input,h_0,z,device)
        else:
            px_logits, _ = model.decoder(input,h_0)

        criterion =  nn.CrossEntropyLoss(ignore_index=0,reduction='sum')
        perplexity = 0
        for i in range(batch_size):
            seq = px_logits[i,:,:]
            target = targets[i,:]
            perplexity += torch.exp(criterion(seq,target)/lenghts[i])
        perplexity /= batch_size
        total_per += perplexity.detach()
    return total_per/step