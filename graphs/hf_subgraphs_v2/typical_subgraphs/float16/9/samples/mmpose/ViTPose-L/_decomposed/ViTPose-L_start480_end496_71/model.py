import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, w_14 : torch.Tensor, w_15 : torch.Tensor, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_16 = torch.nn.functional.gelu(in_0, approximate = 'none');  in_0 = None
        tmp_17 = torch.nn.functional.dropout(tmp_16, 0.0, False, False);  tmp_16 = None
        linear = torch.nn.functional.linear(tmp_17, w_1, w_0);  tmp_17 = w_1 = w_0 = None
        tmp_19 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_20 = in_1 + tmp_19;  in_1 = tmp_19 = None
        tmp_21 = torch.nn.functional.layer_norm(tmp_20, (1024,), w_3, w_2, 1e-06);  tmp_20 = w_3 = w_2 = None
        tmp_22 = tmp_21[(slice(None, None, None), slice(0, None, None))];  tmp_21 = None
        tmp_23 = tmp_22.reshape(1, 16, 12, -1);  tmp_22 = None
        tmp_24 = tmp_23.permute(0, 3, 1, 2);  tmp_23 = None
        tmp_25 = torch.conv_transpose2d(tmp_24, w_4, None, (2, 2), (1, 1), (0, 0), 1, (1, 1));  tmp_24 = w_4 = None
        tmp_26 = torch.nn.functional.batch_norm(tmp_25, w_5, w_6, w_8, w_7, False, 0.1, 1e-05);  tmp_25 = w_5 = w_6 = w_8 = w_7 = None
        tmp_27 = torch.nn.functional.relu(tmp_26, inplace = True);  tmp_26 = None
        tmp_28 = torch.conv_transpose2d(tmp_27, w_9, None, (2, 2), (1, 1), (0, 0), 1, (1, 1));  tmp_27 = w_9 = None
        tmp_29 = torch.nn.functional.batch_norm(tmp_28, w_10, w_11, w_13, w_12, False, 0.1, 1e-05);  tmp_28 = w_10 = w_11 = w_13 = w_12 = None
        tmp_30 = torch.nn.functional.relu(tmp_29, inplace = True);  tmp_29 = None
        conv2d = torch.conv2d(tmp_30, w_15, w_14, (1, 1), (0, 0), (1, 1), 1);  tmp_30 = w_15 = w_14 = None
        return (conv2d,)
        