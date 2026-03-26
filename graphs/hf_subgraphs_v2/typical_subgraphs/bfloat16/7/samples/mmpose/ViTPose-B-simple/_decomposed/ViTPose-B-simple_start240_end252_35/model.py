import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor):
        tmp_6 = torch.nn.functional.gelu(in_6, approximate = 'none');  in_6 = None
        tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False);  tmp_6 = None
        linear = torch.nn.functional.linear(tmp_7, in_1, in_0);  tmp_7 = in_1 = in_0 = None
        tmp_9 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_10 = in_7 + tmp_9;  in_7 = tmp_9 = None
        tmp_11 = torch.nn.functional.layer_norm(tmp_10, (768,), in_3, in_2, 1e-06);  tmp_10 = in_3 = in_2 = None
        tmp_12 = tmp_11[(slice(None, None, None), slice(0, None, None))];  tmp_11 = None
        tmp_13 = tmp_12.reshape(32, 16, 12, -1);  tmp_12 = None
        tmp_14 = tmp_13.permute(0, 3, 1, 2);  tmp_13 = None
        tmp_15 = torch.nn.functional.relu(tmp_14);  tmp_14 = None
        tmp_16 = torch.nn.functional.interpolate(tmp_15, None, 4.0, 'bilinear', False);  tmp_15 = None
        conv2d = torch.conv2d(tmp_16, in_5, in_4, (1, 1), (1, 1), (1, 1), 1);  tmp_16 = in_5 = in_4 = None
        return (conv2d,)
        