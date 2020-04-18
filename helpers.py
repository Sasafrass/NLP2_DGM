# Numerical imports
import numpy

# torch-y
import torch.nn.functional as F

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