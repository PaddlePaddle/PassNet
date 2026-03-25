import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor):
        conv2d = torch.conv2d(in_5, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  in_5 = in_1 = in_0 = None
        tmp_6 = torch.nn.functional.dropout(conv2d, 0.0, False, False);  conv2d = None
        tmp_7 = in_2.unsqueeze(-1);  in_2 = None
        tmp_8 = tmp_7.unsqueeze(-1);  tmp_7 = None
        tmp_9 = tmp_8 * tmp_6;  tmp_8 = tmp_6 = None
        tmp_10 = in_6 + tmp_9;  in_6 = tmp_9 = None
        tmp_11 = tmp_10.flatten(2);  tmp_10 = None
        tmp_12 = tmp_11.transpose(1, 2);  tmp_11 = None
        tmp_13 = torch.nn.functional.layer_norm(tmp_12, (16,), in_4, in_3, 1e-06);  tmp_12 = in_4 = in_3 = None
        tmp_14 = tmp_13.view(512, 56, 56, 16);  tmp_13 = None
        tmp_15 = tmp_14.permute(0, 3, 1, 2);  tmp_14 = None
        return (tmp_15,)
        