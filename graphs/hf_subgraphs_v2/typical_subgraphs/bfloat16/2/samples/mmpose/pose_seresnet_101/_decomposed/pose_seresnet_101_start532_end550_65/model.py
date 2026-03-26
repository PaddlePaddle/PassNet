import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22):
        tmp_21 = torch.nn.functional.adaptive_avg_pool2d(in_22, 1)
        conv2d = torch.conv2d(tmp_21, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  tmp_21 = in_1 = in_0 = None
        tmp_23 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        conv2d_1 = torch.conv2d(tmp_23, in_3, in_2, (1, 1), (0, 0), (1, 1), 1);  tmp_23 = in_3 = in_2 = None
        tmp_25 = torch.sigmoid(conv2d_1);  conv2d_1 = None
        tmp_26 = in_22 * tmp_25;  in_22 = tmp_25 = None
        tmp_26 += in_21;  tmp_27 = tmp_26;  tmp_26 = in_21 = None
        tmp_28 = torch.nn.functional.relu(tmp_27, inplace = True);  tmp_27 = None
        tmp_29 = torch.conv_transpose2d(tmp_28, in_4, None, (2, 2), (1, 1), (0, 0), 1, (1, 1));  tmp_28 = in_4 = None
        tmp_30 = torch.nn.functional.batch_norm(tmp_29, in_5, in_6, in_8, in_7, False, 0.1, 1e-05);  tmp_29 = in_5 = in_6 = in_8 = in_7 = None
        tmp_31 = torch.nn.functional.relu(tmp_30, inplace = True);  tmp_30 = None
        tmp_32 = torch.conv_transpose2d(tmp_31, in_9, None, (2, 2), (1, 1), (0, 0), 1, (1, 1));  tmp_31 = in_9 = None
        tmp_33 = torch.nn.functional.batch_norm(tmp_32, in_10, in_11, in_13, in_12, False, 0.1, 1e-05);  tmp_32 = in_10 = in_11 = in_13 = in_12 = None
        tmp_34 = torch.nn.functional.relu(tmp_33, inplace = True);  tmp_33 = None
        tmp_35 = torch.conv_transpose2d(tmp_34, in_14, None, (2, 2), (1, 1), (0, 0), 1, (1, 1));  tmp_34 = in_14 = None
        tmp_36 = torch.nn.functional.batch_norm(tmp_35, in_15, in_16, in_18, in_17, False, 0.1, 1e-05);  tmp_35 = in_15 = in_16 = in_18 = in_17 = None
        tmp_37 = torch.nn.functional.relu(tmp_36, inplace = True);  tmp_36 = None
        conv2d_2 = torch.conv2d(tmp_37, in_20, in_19, (1, 1), (0, 0), (1, 1), 1);  tmp_37 = in_20 = in_19 = None
        return (conv2d_2,)
        