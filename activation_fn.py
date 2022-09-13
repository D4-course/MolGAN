import torch.nn.functional as F


def soft_gumbel(x):
    '''Sampled tensor of same shape as logits from the Gumbel-Softmax distribution. Returned value is
    they will be probability distributions that sum to 1 across dim.'''
    return F.gumbel_softmax(x, tau=1, hard=False)


def hard_gumbel(x):
    '''the returned samples will be one-hot,'''
    return F.gumbel_softmax(x, tau=1, hard=True)

