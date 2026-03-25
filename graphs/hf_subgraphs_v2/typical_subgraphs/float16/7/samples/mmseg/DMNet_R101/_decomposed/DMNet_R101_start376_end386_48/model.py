import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8):
        tmp_6 = torch.nn.functional.relu(in_8, inplace = True);  in_8 = None
        tmp_7 = tmp_6.view(1, 512, 64, 64);  tmp_6 = None
        tmp_8 = in_6.view(512, 1, 5, 5);  in_6 = None
        tmp_9 = torch.nn.functional.pad(tmp_7, (2, 2, 2, 2), 'constant', 0);  tmp_7 = None
        conv2d = torch.conv2d(input = tmp_9, weight = tmp_8, groups = 512);  tmp_9 = tmp_8 = None
        tmp_11 = conv2d.view(1, 512, 64, 64);  conv2d = None
        tmp_12 = torch.nn.functional.batch_norm(tmp_11, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  tmp_11 = in_0 = in_1 = in_3 = in_2 = None
        tmp_13 = torch.nn.functional.relu(tmp_12, inplace = False);  tmp_12 = None
        tmp_14 = torch.nn.functional.adaptive_avg_pool2d(in_7, 7);  in_7 = None
        conv2d_1 = torch.conv2d(tmp_14, in_5, in_4, (1, 1), (0, 0), (1, 1), 1);  tmp_14 = in_5 = in_4 = None
        return (conv2d_1, tmp_13)
        