import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = torch.nn.functional.relu(in_3, inplace=True)
        tmp_4 = torch.conv2d(tmp_3, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_1 = tmp_0 = None
        tmp_5 = tmp_4.view(1, 512, 4096)
        tmp_4 = None
        tmp_6 = tmp_2.repeat(1, 1, 1)
        tmp_2 = None
        tmp_7 = torch.functional.einsum('bcn,bck->bnk', tmp_5, tmp_6)
        tmp_6 = None
        tmp_8 = torch.nn.functional.softmax(tmp_7, dim=2)
        tmp_7 = None
        tmp_9 = torch.nn.functional.normalize(tmp_8, dim=1, p=1)
        tmp_8 = None
        tmp_10 = torch.functional.einsum('bcn,bnk->bck', tmp_5, tmp_9)
        tmp_9 = None
        tmp_11 = torch.nn.functional.normalize(tmp_10, dim=1, p=2)
        tmp_10 = None
        tmp_12 = torch.functional.einsum('bcn,bck->bnk', tmp_5, tmp_11)
        tmp_11 = None
        tmp_13 = torch.nn.functional.softmax(tmp_12, dim=2)
        tmp_12 = None
        tmp_14 = torch.nn.functional.normalize(tmp_13, dim=1, p=1)
        tmp_13 = None
        tmp_15 = torch.functional.einsum('bcn,bnk->bck', tmp_5, tmp_14)
        tmp_14 = None
        tmp_16 = torch.nn.functional.normalize(tmp_15, dim=1, p=2)
        tmp_15 = None
        tmp_17 = torch.functional.einsum('bcn,bck->bnk', tmp_5, tmp_16)
        tmp_16 = None
        tmp_18 = torch.nn.functional.softmax(tmp_17, dim=2)
        tmp_17 = None
        tmp_19 = torch.nn.functional.normalize(tmp_18, dim=1, p=1)
        tmp_20 = torch.functional.einsum('bcn,bnk->bck', tmp_5, tmp_19)
        tmp_5 = tmp_19 = None
        tmp_21 = torch.nn.functional.normalize(tmp_20, dim=1, p=2)
        tmp_20 = None
        tmp_22 = torch.functional.einsum('bck,bnk->bcn', tmp_21, tmp_18)
        tmp_21 = tmp_18 = None
        tmp_23 = tmp_22.view(1, 512, 64, 64)
        tmp_22 = None
        tmp_24 = torch.nn.functional.relu(tmp_23, inplace=True)
        tmp_23 = None
        return (tmp_24, tmp_3)