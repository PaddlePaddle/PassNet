import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        tmp_3 = torch.nn.functional.relu(in_3, inplace = True);  in_3 = None
        conv2d = torch.conv2d(tmp_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  in_1 = in_0 = None
        tmp_5 = conv2d.view(12, 512, 4096);  conv2d = None
        tmp_6 = in_2.repeat(1, 1, 1);  in_2 = None
        einsum = torch.functional.einsum('bcn,bck->bnk', tmp_5, tmp_6);  tmp_6 = None
        tmp_8 = torch.nn.functional.softmax(einsum, dim = 2);  einsum = None
        tmp_9 = torch.nn.functional.normalize(tmp_8, dim = 1, p = 1);  tmp_8 = None
        einsum_1 = torch.functional.einsum('bcn,bnk->bck', tmp_5, tmp_9);  tmp_9 = None
        tmp_11 = torch.nn.functional.normalize(einsum_1, dim = 1, p = 2);  einsum_1 = None
        einsum_2 = torch.functional.einsum('bcn,bck->bnk', tmp_5, tmp_11);  tmp_11 = None
        tmp_13 = torch.nn.functional.softmax(einsum_2, dim = 2);  einsum_2 = None
        tmp_14 = torch.nn.functional.normalize(tmp_13, dim = 1, p = 1);  tmp_13 = None
        einsum_3 = torch.functional.einsum('bcn,bnk->bck', tmp_5, tmp_14);  tmp_14 = None
        tmp_16 = torch.nn.functional.normalize(einsum_3, dim = 1, p = 2);  einsum_3 = None
        einsum_4 = torch.functional.einsum('bcn,bck->bnk', tmp_5, tmp_16);  tmp_16 = None
        tmp_18 = torch.nn.functional.softmax(einsum_4, dim = 2);  einsum_4 = None
        tmp_19 = torch.nn.functional.normalize(tmp_18, dim = 1, p = 1)
        einsum_5 = torch.functional.einsum('bcn,bnk->bck', tmp_5, tmp_19);  tmp_5 = tmp_19 = None
        tmp_21 = torch.nn.functional.normalize(einsum_5, dim = 1, p = 2);  einsum_5 = None
        einsum_6 = torch.functional.einsum('bck,bnk->bcn', tmp_21, tmp_18);  tmp_21 = tmp_18 = None
        tmp_23 = einsum_6.view(12, 512, 64, 64);  einsum_6 = None
        tmp_24 = torch.nn.functional.relu(tmp_23, inplace = True);  tmp_23 = None
        return (tmp_24, tmp_3)
        