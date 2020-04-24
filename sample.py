import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal

import numpy as np

def sample(model,topic,tokenizer,device,temperature=1):
    topic = torch.tensor(tokenizer.encode(topic))
    text=[1]
    start = np.array(text)
    current = torch.from_numpy(start).long().view(1,-1)
    mean, std = model.encoder(topic.to(device).unsqueeze(0))
    q_z = Normal(mean,std)
    sample_z = q_z.rsample().view(1,1,-1).to(device)

    #The initial step
    input = current.to(device)
    h_0 = torch.tanh(model.upscale(sample_z))
    if(model.skip):
        z = model.z_lin(sample_z)	
        output,hidden = model.skip_decoder(input, h_0, z, device)
    else:
        output,hidden = model.decoder(input, h_0)
    current = output[0,-1,:].squeeze()
    guess = torch.multinomial(F.softmax(temperature*current,dim=0),1)
    text.append(guess.item())
    input = guess.unsqueeze(0)

    #Now that we have an h and c, we can start the loop
    i = 0
    while(i < 100):
        if(model.skip):
            output,hidden = model.skip_decoder(input,hidden,z,device)
        else:
            output,hidden = model.decoder(input,hidden)
        current = output.squeeze()
        guess = torch.multinomial(F.softmax(temperature*current,dim=0),1)
        text.append(guess.item())
        input = guess.unsqueeze(0)
        i += 1
        if(guess.item() == 2):
            break

    return tokenizer.decode(text)