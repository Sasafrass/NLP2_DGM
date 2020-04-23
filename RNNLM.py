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
    def __init__(self, vocab_size, embed_size, hidden_dim):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.gru = nn.GRU(embed_size,hidden_dim,batch_first=True)
        self.output = nn.Linear(hidden_dim,vocab_size)

    def forward(self, input, hidden=None):
        embed = self.embed(input)

        if(hidden is None):
            out,hidden = self.gru.forward(embed)
        else:
            out,hidden = self.gru.forward(embed,hidden)

        out = self.output(out)

        return out, hidden