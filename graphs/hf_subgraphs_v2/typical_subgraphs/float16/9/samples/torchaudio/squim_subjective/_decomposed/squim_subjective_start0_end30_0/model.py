import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, w_14 : torch.Tensor, w_15 : torch.Tensor, w_16 : torch.Tensor, w_17 : torch.Tensor, in_1 : torch.Tensor):
        tmp_20 = in_0[(slice(None, None, None), slice(None, 80000, None))];  in_0 = None
        tmp_21 = in_1.unsqueeze(1);  in_1 = None
        conv1d = torch.conv1d(tmp_21, w_9, None, (5,), (0,), (1,), 1);  tmp_21 = w_9 = None
        tmp_23 = torch.nn.functional.group_norm(conv1d, 512, w_11, w_10, 1e-05);  conv1d = w_11 = w_10 = None
        tmp_24 = torch.nn.functional.gelu(tmp_23);  tmp_23 = None
        conv1d_1 = torch.conv1d(tmp_24, w_12, None, (2,), (0,), (1,), 1);  tmp_24 = w_12 = None
        tmp_26 = torch.nn.functional.gelu(conv1d_1);  conv1d_1 = None
        conv1d_2 = torch.conv1d(tmp_26, w_13, None, (2,), (0,), (1,), 1);  tmp_26 = w_13 = None
        tmp_28 = torch.nn.functional.gelu(conv1d_2);  conv1d_2 = None
        conv1d_3 = torch.conv1d(tmp_28, w_14, None, (2,), (0,), (1,), 1);  tmp_28 = w_14 = None
        tmp_30 = torch.nn.functional.gelu(conv1d_3);  conv1d_3 = None
        conv1d_4 = torch.conv1d(tmp_30, w_15, None, (2,), (0,), (1,), 1);  tmp_30 = w_15 = None
        tmp_32 = torch.nn.functional.gelu(conv1d_4);  conv1d_4 = None
        conv1d_5 = torch.conv1d(tmp_32, w_16, None, (2,), (0,), (1,), 1);  tmp_32 = w_16 = None
        tmp_34 = torch.nn.functional.gelu(conv1d_5);  conv1d_5 = None
        conv1d_6 = torch.conv1d(tmp_34, w_17, None, (2,), (0,), (1,), 1);  tmp_34 = w_17 = None
        tmp_36 = torch.nn.functional.gelu(conv1d_6);  conv1d_6 = None
        tmp_37 = tmp_36.transpose(1, 2);  tmp_36 = None
        tmp_38 = torch.nn.functional.layer_norm(tmp_37, (512,), w_1, w_0, 1e-05);  tmp_37 = w_1 = w_0 = None
        linear = torch.nn.functional.linear(tmp_38, w_3, w_2);  tmp_38 = w_3 = w_2 = None
        tmp_40 = torch.nn.functional.dropout(linear, 0.1, False, False);  linear = None
        tmp_41 = tmp_40.transpose(-2, -1)
        tmp_42 = torch._weight_norm(w_7, w_6, 2);  w_7 = w_6 = None
        conv1d_7 = torch.conv1d(tmp_41, tmp_42, w_8, (1,), (64,), (1,), 16);  tmp_41 = tmp_42 = w_8 = None
        tmp_44 = conv1d_7[(Ellipsis, slice(None, -1, None))];  conv1d_7 = None
        tmp_45 = torch.nn.functional.gelu(tmp_44);  tmp_44 = None
        tmp_46 = tmp_45.transpose(-2, -1);  tmp_45 = None
        tmp_47 = tmp_40 + tmp_46;  tmp_40 = tmp_46 = None
        tmp_48 = torch.nn.functional.layer_norm(tmp_47, (768,), w_5, w_4, 1e-05);  tmp_47 = w_5 = w_4 = None
        tmp_49 = torch.nn.functional.dropout(tmp_48, 0.1, False, False);  tmp_48 = None
        return (tmp_20, tmp_49)
        