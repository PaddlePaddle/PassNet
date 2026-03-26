import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18):
        in_18 += in_17;  in_19 = in_18;  in_18 = in_17 = None
        tmp_18 = torch.nn.functional.relu(in_19, inplace = True);  in_19 = None
        tmp_19 = torch.conv_transpose2d(tmp_18, in_0, None, (2, 2), (1, 1), (0, 0), 1, (1, 1));  tmp_18 = in_0 = None
        tmp_20 = torch.nn.functional.batch_norm(tmp_19, in_1, in_2, in_4, in_3, False, 0.1, 1e-05);  tmp_19 = in_1 = in_2 = in_4 = in_3 = None
        tmp_21 = torch.nn.functional.relu(tmp_20, inplace = True);  tmp_20 = None
        tmp_22 = torch.conv_transpose2d(tmp_21, in_5, None, (2, 2), (1, 1), (0, 0), 1, (1, 1));  tmp_21 = in_5 = None
        tmp_23 = torch.nn.functional.batch_norm(tmp_22, in_6, in_7, in_9, in_8, False, 0.1, 1e-05);  tmp_22 = in_6 = in_7 = in_9 = in_8 = None
        tmp_24 = torch.nn.functional.relu(tmp_23, inplace = True);  tmp_23 = None
        tmp_25 = torch.conv_transpose2d(tmp_24, in_10, None, (2, 2), (1, 1), (0, 0), 1, (1, 1));  tmp_24 = in_10 = None
        tmp_26 = torch.nn.functional.batch_norm(tmp_25, in_11, in_12, in_14, in_13, False, 0.1, 1e-05);  tmp_25 = in_11 = in_12 = in_14 = in_13 = None
        tmp_27 = torch.nn.functional.relu(tmp_26, inplace = True);  tmp_26 = None
        conv2d = torch.conv2d(tmp_27, in_16, in_15, (1, 1), (0, 0), (1, 1), 1);  tmp_27 = in_16 = in_15 = None
        return (conv2d,)
        