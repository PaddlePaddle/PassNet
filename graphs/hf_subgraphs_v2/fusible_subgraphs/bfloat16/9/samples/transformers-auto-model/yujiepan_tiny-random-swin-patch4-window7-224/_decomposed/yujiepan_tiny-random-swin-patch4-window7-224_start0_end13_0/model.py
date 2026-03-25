import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor):
        conv2d = torch.conv2d(in_0, in_4, in_3, (4, 4), (0, 0), (1, 1), 1);  in_0 = in_4 = in_3 = None
        tmp_8 = conv2d.flatten(2);  conv2d = None
        tmp_9 = tmp_8.transpose(1, 2);  tmp_8 = None
        tmp_10 = torch.nn.functional.layer_norm(tmp_9, (8,), in_2, in_1, 1e-05);  tmp_9 = in_2 = in_1 = None
        tmp_11 = torch.nn.functional.dropout(tmp_10, 0.0, False, False);  tmp_10 = None
        tmp_12 = torch.nn.functional.layer_norm(tmp_11, (8,), in_6, in_5, 1e-05);  in_6 = in_5 = None
        tmp_13 = tmp_12.view(1, 56, 56, 8);  tmp_12 = None
        tmp_14 = torch.nn.functional.pad(tmp_13, (0, 0, 0, 0, 0, 0), 'constant', None);  tmp_13 = None
        tmp_15 = tmp_14.view(1, 8, 7, 8, 7, 8);  tmp_14 = None
        tmp_16 = tmp_15.permute(0, 1, 3, 2, 4, 5);  tmp_15 = None
        tmp_17 = tmp_16.contiguous();  tmp_16 = None
        tmp_18 = tmp_17.view(-1, 7, 7, 8);  tmp_17 = None
        tmp_19 = tmp_18.view(-1, 49, 8);  tmp_18 = None
        return (tmp_11, tmp_19)
        