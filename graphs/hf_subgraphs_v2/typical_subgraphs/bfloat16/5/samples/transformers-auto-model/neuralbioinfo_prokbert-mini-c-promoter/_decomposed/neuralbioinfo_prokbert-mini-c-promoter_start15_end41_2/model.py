import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6):
        linear = torch.nn.functional.linear(in_5, in_2, in_1);  in_5 = in_2 = in_1 = None
        tmp_4 = linear.view(4, -1, 6, 64);  linear = None
        tmp_5 = tmp_4.transpose(1, 2);  tmp_4 = None
        tmp_6 = in_4.transpose(-1, -2)
        matmul = torch.matmul(in_6, tmp_6);  tmp_6 = None
        tmp_8 = torch.arange(512, dtype = torch.int64, device = device(type='cuda', index=0))
        tmp_9 = tmp_8.view(-1, 1);  tmp_8 = None
        tmp_10 = torch.arange(512, dtype = torch.int64, device = device(type='cuda', index=0))
        tmp_11 = tmp_10.view(1, -1);  tmp_10 = None
        tmp_12 = tmp_9 - tmp_11;  tmp_9 = tmp_11 = None
        tmp_13 = tmp_12 + 2048;  tmp_12 = None
        tmp_14 = tmp_13 - 1;  tmp_13 = None
        tmp_15 = torch.nn.functional.embedding(tmp_14, in_0, None, None, 2.0, False, False);  tmp_14 = in_0 = None
        tmp_16 = tmp_15.to(dtype = torch.float32);  tmp_15 = None
        to_1 = tmp_16.to(torch.bfloat16)
        einsum = torch.functional.einsum('bhld,lrd->bhlr', in_6, to_1);  in_6 = to_1 = None
        to_2 = tmp_16.to(torch.bfloat16);  tmp_16 = None
        einsum_1 = torch.functional.einsum('bhrd,lrd->bhlr', in_4, to_2);  in_4 = to_2 = None
        tmp_19 = matmul + einsum;  matmul = einsum = None
        tmp_20 = tmp_19 + einsum_1;  tmp_19 = einsum_1 = None
        tmp_21 = tmp_20 / 8.0;  tmp_20 = None
        tmp_22 = tmp_21 + in_3;  tmp_21 = in_3 = None
        tmp_23 = torch.nn.functional.softmax(tmp_22, dim = -1);  tmp_22 = None
        tmp_24 = torch.nn.functional.dropout(tmp_23, 0.1, False, False);  tmp_23 = None
        matmul_1 = torch.matmul(tmp_24, tmp_5);  tmp_24 = tmp_5 = None
        tmp_26 = matmul_1.permute(0, 2, 1, 3);  matmul_1 = None
        tmp_27 = tmp_26.contiguous();  tmp_26 = None
        tmp_28 = tmp_27.view((4, 512, 384));  tmp_27 = None
        return (tmp_28,)
        