import torch

from torch import device

from torch import inf

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, in_0 : torch.Tensor):
        tmp_7 = torch.nn.functional.relu(in_0, inplace = True);  in_0 = None
        conv2d = torch.conv2d(tmp_7, w_4, w_3, (1, 1), (0, 0), (1, 1), 1)
        conv2d_1 = torch.conv2d(tmp_7, w_2, w_1, (1, 1), (0, 0), (1, 1), 1)
        conv2d_2 = torch.conv2d(tmp_7, w_6, w_5, (1, 1), (0, 0), (1, 1), 1)
        einsum = torch.functional.einsum('bchw,bciw->bwhi', conv2d, conv2d_1)
        tmp_12 = torch.tensor(-inf)
        tmp_13 = tmp_12.to(device(type='cuda', index=0));  tmp_12 = None
        tmp_14 = tmp_13.repeat(64);  tmp_13 = None
        tmp_15 = torch.diag(tmp_14, 0);  tmp_14 = None
        tmp_16 = einsum + tmp_15;  einsum = tmp_15 = None
        tmp_17 = tmp_16.transpose(1, 2);  tmp_16 = None
        einsum_1 = torch.functional.einsum('bchw,bchj->bhwj', conv2d, conv2d_1);  conv2d = conv2d_1 = None
        tmp_19 = torch.cat([tmp_17, einsum_1], dim = -1);  tmp_17 = einsum_1 = None
        tmp_20 = torch.nn.functional.softmax(tmp_19, dim = -1);  tmp_19 = None
        tmp_21 = tmp_20[(Ellipsis, slice(None, 64, None))]
        to_8 = tmp_21.to(torch.float16);  tmp_21 = None
        einsum_2 = torch.functional.einsum('bciw,bhwi->bchw', conv2d_2, to_8);  to_8 = None
        tmp_23 = tmp_20[(Ellipsis, slice(64, None, None))];  tmp_20 = None
        to_10 = tmp_23.to(torch.float16);  tmp_23 = None
        einsum_3 = torch.functional.einsum('bchj,bhwj->bchw', conv2d_2, to_10);  conv2d_2 = to_10 = None
        einsum_2 += einsum_3;  tmp_25 = einsum_2;  einsum_2 = einsum_3 = None
        tmp_26 = tmp_25 * w_0;  tmp_25 = None
        tmp_27 = tmp_26 + tmp_7;  tmp_26 = tmp_7 = None
        tmp_28 = tmp_27.contiguous();  tmp_27 = None
        conv2d_3 = torch.conv2d(tmp_28, w_4, w_3, (1, 1), (0, 0), (1, 1), 1);  w_4 = w_3 = None
        conv2d_4 = torch.conv2d(tmp_28, w_2, w_1, (1, 1), (0, 0), (1, 1), 1);  w_2 = w_1 = None
        conv2d_5 = torch.conv2d(tmp_28, w_6, w_5, (1, 1), (0, 0), (1, 1), 1);  w_6 = w_5 = None
        einsum_4 = torch.functional.einsum('bchw,bciw->bwhi', conv2d_3, conv2d_4)
        tmp_33 = torch.tensor(-inf)
        tmp_34 = tmp_33.to(device(type='cuda', index=0));  tmp_33 = None
        tmp_35 = tmp_34.repeat(64);  tmp_34 = None
        tmp_36 = torch.diag(tmp_35, 0);  tmp_35 = None
        tmp_37 = einsum_4 + tmp_36;  einsum_4 = tmp_36 = None
        tmp_38 = tmp_37.transpose(1, 2);  tmp_37 = None
        einsum_5 = torch.functional.einsum('bchw,bchj->bhwj', conv2d_3, conv2d_4);  conv2d_3 = conv2d_4 = None
        tmp_40 = torch.cat([tmp_38, einsum_5], dim = -1);  tmp_38 = einsum_5 = None
        tmp_41 = torch.nn.functional.softmax(tmp_40, dim = -1);  tmp_40 = None
        tmp_42 = tmp_41[(Ellipsis, slice(None, 64, None))]
        to_19 = tmp_42.to(torch.float16);  tmp_42 = None
        einsum_6 = torch.functional.einsum('bciw,bhwi->bchw', conv2d_5, to_19);  to_19 = None
        tmp_44 = tmp_41[(Ellipsis, slice(64, None, None))];  tmp_41 = None
        to_21 = tmp_44.to(torch.float16);  tmp_44 = None
        einsum_7 = torch.functional.einsum('bchj,bhwj->bchw', conv2d_5, to_21);  conv2d_5 = to_21 = None
        einsum_6 += einsum_7;  tmp_46 = einsum_6;  einsum_6 = einsum_7 = None
        tmp_47 = tmp_46 * w_0;  tmp_46 = w_0 = None
        tmp_48 = tmp_47 + tmp_28;  tmp_47 = tmp_28 = None
        tmp_49 = tmp_48.contiguous();  tmp_48 = None
        return (tmp_49,)
        