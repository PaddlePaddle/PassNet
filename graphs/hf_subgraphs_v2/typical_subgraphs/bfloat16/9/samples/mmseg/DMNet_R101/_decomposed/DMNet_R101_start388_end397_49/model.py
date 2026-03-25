import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_4 = torch.nn.functional.relu(in_5, inplace = True);  in_5 = None
        tmp_5 = tmp_4.view(1, 512, 64, 64);  tmp_4 = None
        tmp_6 = in_0.view(512, 1, 7, 7);  in_0 = None
        tmp_7 = torch.nn.functional.pad(tmp_5, (3, 3, 3, 3), 'constant', 0);  tmp_5 = None
        conv2d = torch.conv2d(input = tmp_7, weight = tmp_6, groups = 512);  tmp_7 = tmp_6 = None
        tmp_9 = conv2d.view(1, 512, 64, 64);  conv2d = None
        tmp_10 = torch.nn.functional.batch_norm(tmp_9, w_0, w_1, w_3, w_2, False, 0.1, 1e-05);  tmp_9 = w_0 = w_1 = w_3 = w_2 = None
        tmp_11 = torch.nn.functional.relu(tmp_10, inplace = False);  tmp_10 = None
        tmp_12 = torch.cat([in_1, in_3, in_4, in_2, tmp_11], dim = 1);  in_1 = in_3 = in_4 = in_2 = tmp_11 = None
        return (tmp_12,)
        