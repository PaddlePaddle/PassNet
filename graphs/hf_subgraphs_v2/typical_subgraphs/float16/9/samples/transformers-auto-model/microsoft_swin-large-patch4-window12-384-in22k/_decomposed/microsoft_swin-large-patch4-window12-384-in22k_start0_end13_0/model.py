import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor):
        conv2d = torch.conv2d(in_0, w_3, w_2, (4, 4), (0, 0), (1, 1), 1);  in_0 = w_3 = w_2 = None
        tmp_8 = conv2d.flatten(2);  conv2d = None
        tmp_9 = tmp_8.transpose(1, 2);  tmp_8 = None
        tmp_10 = torch.nn.functional.layer_norm(tmp_9, (192,), w_1, w_0, 1e-05);  tmp_9 = w_1 = w_0 = None
        tmp_11 = torch.nn.functional.dropout(tmp_10, 0.0, False, False);  tmp_10 = None
        tmp_12 = torch.nn.functional.layer_norm(tmp_11, (192,), w_5, w_4, 1e-05);  w_5 = w_4 = None
        tmp_13 = tmp_12.view(1, 96, 96, 192);  tmp_12 = None
        tmp_14 = torch.nn.functional.pad(tmp_13, (0, 0, 0, 0, 0, 0), 'constant', None);  tmp_13 = None
        tmp_15 = tmp_14.view(1, 8, 12, 8, 12, 192);  tmp_14 = None
        tmp_16 = tmp_15.permute(0, 1, 3, 2, 4, 5);  tmp_15 = None
        tmp_17 = tmp_16.contiguous();  tmp_16 = None
        tmp_18 = tmp_17.view(-1, 12, 12, 192);  tmp_17 = None
        tmp_19 = tmp_18.view(-1, 144, 192);  tmp_18 = None
        return (tmp_11, tmp_19)
        