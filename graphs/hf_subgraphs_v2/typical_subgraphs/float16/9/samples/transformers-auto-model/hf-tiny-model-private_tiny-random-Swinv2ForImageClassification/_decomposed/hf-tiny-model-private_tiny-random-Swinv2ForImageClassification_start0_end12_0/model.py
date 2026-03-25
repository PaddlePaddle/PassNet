import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor):
        conv2d = torch.conv2d(in_0, w_3, w_2, (2, 2), (0, 0), (1, 1), 1);  in_0 = w_3 = w_2 = None
        tmp_6 = conv2d.flatten(2);  conv2d = None
        tmp_7 = tmp_6.transpose(1, 2);  tmp_6 = None
        tmp_8 = torch.nn.functional.layer_norm(tmp_7, (16,), w_1, w_0, 1e-05);  tmp_7 = w_1 = w_0 = None
        tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False);  tmp_8 = None
        tmp_10 = tmp_9.view(1, 16, 16, 16)
        tmp_11 = torch.nn.functional.pad(tmp_10, (0, 0, 0, 0, 0, 0), 'constant', None);  tmp_10 = None
        tmp_12 = tmp_11.view(1, 8, 2, 8, 2, 16);  tmp_11 = None
        tmp_13 = tmp_12.permute(0, 1, 3, 2, 4, 5);  tmp_12 = None
        tmp_14 = tmp_13.contiguous();  tmp_13 = None
        tmp_15 = tmp_14.view(-1, 2, 2, 16);  tmp_14 = None
        tmp_16 = tmp_15.view(-1, 4, 16);  tmp_15 = None
        return (tmp_9, tmp_16)
        