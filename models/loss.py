import numpy as np
import torch


def decl_triplet_loss(tensor_A_pos, tensor_B_org, tensor_C_neg):
    dot_poduct1 = torch.bmm(tensor_A_pos, tensor_B_org.transpose(1, 2))
    dot_poduct2 = torch.bmm(tensor_B_org, tensor_C_neg.transpose(1, 2))
    pos = torch.exp(dot_poduct1)
    neg = torch.exp(dot_poduct2)
    denominator = pos+neg
    loss = torch.mean(-torch.log(torch.div(pos, denominator)))
    return loss
