import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor):
        tmp_15 = in_0[(slice(None, None, None), None)];  in_0 = None
        conv1d = torch.conv1d(tmp_15, w_5, None, (4,), (0,), (1,), 1);  tmp_15 = w_5 = None
        tmp_17 = torch.nn.functional.group_norm(conv1d, 32, w_7, w_6, 1e-05);  conv1d = w_7 = w_6 = None
        tmp_18 = torch.nn.functional.gelu(tmp_17);  tmp_17 = None
        conv1d_1 = torch.conv1d(tmp_18, w_8, None, (4,), (0,), (1,), 1);  tmp_18 = w_8 = None
        tmp_20 = torch.nn.functional.gelu(conv1d_1);  conv1d_1 = None
        conv1d_2 = torch.conv1d(tmp_20, w_9, None, (4,), (0,), (1,), 1);  tmp_20 = w_9 = None
        tmp_22 = torch.nn.functional.gelu(conv1d_2);  conv1d_2 = None
        tmp_23 = tmp_22.transpose(1, 2);  tmp_22 = None
        tmp_24 = torch.nn.functional.layer_norm(tmp_23, (32,), w_11, w_10, 1e-05);  tmp_23 = w_11 = w_10 = None
        linear = torch.nn.functional.linear(tmp_24, w_13, w_12);  tmp_24 = w_13 = w_12 = None
        tmp_26 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_27 = tmp_26.transpose(1, 2)
        tmp_28 = torch._weight_norm(w_3, w_2, 2);  w_3 = w_2 = None
        conv1d_3 = torch.conv1d(tmp_27, tmp_28, w_4, (1,), (8,), (1,), 2);  tmp_27 = tmp_28 = w_4 = None
        tmp_30 = conv1d_3[(slice(None, None, None), slice(None, None, None), slice(None, -1, None))];  conv1d_3 = None
        tmp_31 = torch.nn.functional.gelu(tmp_30);  tmp_30 = None
        tmp_32 = tmp_31.transpose(1, 2);  tmp_31 = None
        tmp_33 = tmp_26 + tmp_32;  tmp_26 = tmp_32 = None
        tmp_34 = torch.nn.functional.layer_norm(tmp_33, (16,), w_1, w_0, 1e-05);  tmp_33 = w_1 = w_0 = None
        tmp_35 = torch.nn.functional.dropout(tmp_34, 0.1, False, False);  tmp_34 = None
        tmp_36 = torch.rand([]);  tmp_36 = None
        return (tmp_35,)
        