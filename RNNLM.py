import torch.nn as nn

class RNNLM(nn.Module):
    """
    Full RNNLM module including forward pass

    Args:
        x     : Input to our model
        hidden: Hidden state from previous timestep

    Returns:
        out   : Output for this timestep
        hidden: Hidden state for this timestep
    """

    # Initialize the GRU model
    def __init__(self, vocab_size, gru_num_hidden=256, gru_num_layers=2, device='cuda:0',dropout=0):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size,256)
        self.gru = nn.GRU(256,gru_num_hidden,gru_num_layers,dropout=dropout,batch_first=True)
        self.output = nn.Linear(gru_num_hidden,vocab_size)
        self.device = device
        self.to(device)

    # 
    def forward(self, x, hidden=None):
        input = x.to(self.device)
        embed = self.embed(input)
        if(hidden == None):
            out,hidden = self.gru.forward(embed)
        else:
            out,hidden = self.gru.forward(embed,hidden)
        out = self.output(out)

        return out, hidden