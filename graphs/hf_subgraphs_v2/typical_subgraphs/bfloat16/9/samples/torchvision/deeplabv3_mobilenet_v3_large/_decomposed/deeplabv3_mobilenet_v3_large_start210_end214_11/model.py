import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, in_0 : torch.Tensor):
        tmp_2 = torch.nn.functional.relu(in_0, inplace = False);  in_0 = None
        tmp_3 = torch.nn.functional.dropout(tmp_2, 0.1, False, False);  tmp_2 = None
        conv2d = torch.conv2d(tmp_3, w_1, w_0, (1, 1), (0, 0), (1, 1), 1);  tmp_3 = w_1 = w_0 = None
        tmp_5 = torch.nn.functional.interpolate(conv2d, size = (224, 224), mode = 'bilinear', align_corners = False);  conv2d = None
        return (tmp_5,)
        