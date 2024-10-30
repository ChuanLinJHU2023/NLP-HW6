import torch


def hstack(v: torch.Tensor, n=None):
    """
    horizontally stack column vectors!!!!! We have n such vectors
    """
    result = torch.concat([v.unsqueeze(1)] * n, dim = 1) # unsqueeze(1) to get column vector
    assert result.shape == (len(v), n)
    return result


def vstack( v: torch.Tensor, n=None):
    """
    vertically stack row vectors!!!!! We have n such vectors
    """
    result = torch.concat([v.unsqueeze(0)] * n, dim = 0) # unsqueeze(0) to get row vector
    assert result.shape == (n, len(v))
    return result

a = torch.tensor([1,2,3])
print(hstack(a,4))
print(vstack(a,4))