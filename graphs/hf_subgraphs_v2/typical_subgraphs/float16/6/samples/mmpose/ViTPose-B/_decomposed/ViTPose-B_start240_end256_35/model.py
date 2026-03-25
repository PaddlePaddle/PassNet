import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor, in_14 : torch.Tensor, in_15 : torch.Tensor, in_16 : torch.Tensor, in_17 : torch.Tensor):
        tmp_16 = torch.nn.functional.gelu(in_16, approximate = 'none');  in_16 = None
        tmp_17 = torch.nn.functional.dropout(tmp_16, 0.0, False, False);  tmp_16 = None
        linear = torch.nn.functional.linear(tmp_17, in_1, in_0);  tmp_17 = in_1 = in_0 = None
        tmp_19 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_20 = in_17 + tmp_19;  in_17 = tmp_19 = None
        tmp_21 = torch.nn.functional.layer_norm(tmp_20, (768,), in_3, in_2, 1e-06);  tmp_20 = in_3 = in_2 = None
        tmp_22 = tmp_21[(slice(None, None, None), slice(0, None, None))];  tmp_21 = None
        tmp_23 = tmp_22.reshape(16, 16, 12, -1);  tmp_22 = None
        tmp_24 = tmp_23.permute(0, 3, 1, 2);  tmp_23 = None
        tmp_25 = torch.conv_transpose2d(tmp_24, in_4, None, (2, 2), (1, 1), (0, 0), 1, (1, 1));  tmp_24 = in_4 = None
        tmp_26 = torch.nn.functional.batch_norm(tmp_25, in_5, in_6, in_8, in_7, False, 0.1, 1e-05);  tmp_25 = in_5 = in_6 = in_8 = in_7 = None
        tmp_27 = torch.nn.functional.relu(tmp_26, inplace = True);  tmp_26 = None
        tmp_28 = torch.conv_transpose2d(tmp_27, in_9, None, (2, 2), (1, 1), (0, 0), 1, (1, 1));  tmp_27 = in_9 = None
        tmp_29 = torch.nn.functional.batch_norm(tmp_28, in_10, in_11, in_13, in_12, False, 0.1, 1e-05);  tmp_28 = in_10 = in_11 = in_13 = in_12 = None
        tmp_30 = torch.nn.functional.relu(tmp_29, inplace = True);  tmp_29 = None
        conv2d = torch.conv2d(tmp_30, in_15, in_14, (1, 1), (0, 0), (1, 1), 1);  tmp_30 = in_15 = in_14 = None
        return (conv2d,)
        