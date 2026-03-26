import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor):
        conv2d = torch.conv2d(in_0, w_7, w_6, (4, 4), (0, 0), (1, 1), 1);  in_0 = w_7 = w_6 = None
        tmp_10 = conv2d.flatten(2);  conv2d = None
        tmp_11 = tmp_10.transpose(1, 2);  tmp_10 = None
        tmp_12 = torch.nn.functional.layer_norm(tmp_11, (128,), w_5, w_4, 1e-05);  tmp_11 = w_5 = w_4 = None
        tmp_13 = torch.nn.functional.dropout(tmp_12, 0.0, False, False);  tmp_12 = None
        tmp_14 = torch.nn.functional.layer_norm(tmp_13, (128,), w_3, w_2, 1e-05);  w_3 = w_2 = None
        tmp_15 = tmp_14.view(1, 128, 128, 128);  tmp_14 = None
        tmp_16 = torch.nn.functional.pad(tmp_15, (0, 0, 0, 5, 0, 5), 'constant', None);  tmp_15 = None
        tmp_17 = torch.zeros((1, 133, 133), device = device(type='cuda', index=0))
        tmp_18 = tmp_17[(slice(None, None, None), slice(-5, None, None), slice(None, None, None))]
        tmp_19 = tmp_18.fill_(1);  tmp_18 = tmp_19 = None
        tmp_20 = tmp_17[(slice(None, None, None), slice(None, None, None), slice(-5, None, None))]
        tmp_21 = tmp_20.fill_(1);  tmp_20 = tmp_21 = None
        tmp_22 = tmp_16.reshape(1, 19, 7, 19, 7, 128);  tmp_16 = None
        tmp_23 = tmp_22.transpose(2, 3);  tmp_22 = None
        tmp_24 = tmp_17.reshape(1, 19, 7, 19, 7);  tmp_17 = None
        tmp_25 = tmp_24.transpose(2, 3);  tmp_24 = None
        tmp_26 = tmp_25.reshape(1, 361, 49);  tmp_25 = None
        tmp_27 = tmp_26.unsqueeze(2)
        tmp_28 = tmp_26.unsqueeze(3);  tmp_26 = None
        tmp_29 = tmp_27 - tmp_28;  tmp_27 = tmp_28 = None
        tmp_30 = tmp_29 != 0
        tmp_31 = tmp_29.masked_fill(tmp_30, -1000.0);  tmp_30 = None
        tmp_32 = tmp_29 == 0;  tmp_29 = None
        tmp_33 = tmp_31.masked_fill(tmp_32, 0.0);  tmp_31 = tmp_32 = None
        linear = torch.nn.functional.linear(tmp_23, w_1, w_0);  tmp_23 = w_1 = w_0 = None
        tmp_35 = linear.reshape(1, 361, 49, 3, 4, 32);  linear = None
        tmp_36 = tmp_35.permute(3, 0, 1, 4, 2, 5);  tmp_35 = None
        tmp_37 = tmp_36[0]
        tmp_38 = tmp_36[1]
        tmp_39 = tmp_36[2];  tmp_36 = None
        tmp_40 = tmp_38.transpose(-2, -1);  tmp_38 = None
        matmul = tmp_37 @ tmp_40;  tmp_37 = tmp_40 = None
        tmp_42 = matmul * 0.1767766952966369;  matmul = None
        tmp_43 = tmp_33.unsqueeze(2);  tmp_33 = None
        tmp_44 = tmp_42 + tmp_43;  tmp_42 = tmp_43 = None
        tmp_45 = tmp_44.softmax(dim = -1);  tmp_44 = None
        tmp_46 = torch.nn.functional.dropout(tmp_45, 0.0, False, False);  tmp_45 = None
        to_3 = tmp_46.to(torch.float16);  tmp_46 = None
        matmul_1 = to_3 @ tmp_39;  to_3 = tmp_39 = None
        tmp_48 = matmul_1.transpose(2, 3);  matmul_1 = None
        tmp_49 = tmp_48.reshape(1, 19, 19, 7, 7, 128);  tmp_48 = None
        tmp_50 = tmp_49.transpose(2, 3);  tmp_49 = None
        tmp_51 = tmp_50.reshape(1, 133, 133, 128);  tmp_50 = None
        tmp_52 = tmp_51[(slice(None, None, None), slice(None, 128, None), slice(None, 128, None), slice(None, None, None))];  tmp_51 = None
        tmp_53 = tmp_52.contiguous();  tmp_52 = None
        tmp_54 = tmp_53.reshape(1, 16384, 128);  tmp_53 = None
        return (tmp_13, tmp_54)
        