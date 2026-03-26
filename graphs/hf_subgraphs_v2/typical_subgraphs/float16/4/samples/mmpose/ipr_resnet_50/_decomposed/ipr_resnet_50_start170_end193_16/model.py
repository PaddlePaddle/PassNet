import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20):
        in_20 += in_19;  in_21 = in_20;  in_20 = in_19 = None
        tmp_20 = torch.nn.functional.relu(in_21, inplace = True);  in_21 = None
        tmp_21 = torch.conv_transpose2d(tmp_20, in_0, None, (2, 2), (1, 1), (0, 0), 1, (1, 1));  tmp_20 = in_0 = None
        tmp_22 = torch.nn.functional.batch_norm(tmp_21, in_1, in_2, in_4, in_3, False, 0.1, 1e-05);  tmp_21 = in_1 = in_2 = in_4 = in_3 = None
        tmp_23 = torch.nn.functional.relu(tmp_22, inplace = True);  tmp_22 = None
        tmp_24 = torch.conv_transpose2d(tmp_23, in_5, None, (2, 2), (1, 1), (0, 0), 1, (1, 1));  tmp_23 = in_5 = None
        tmp_25 = torch.nn.functional.batch_norm(tmp_24, in_6, in_7, in_9, in_8, False, 0.1, 1e-05);  tmp_24 = in_6 = in_7 = in_9 = in_8 = None
        tmp_26 = torch.nn.functional.relu(tmp_25, inplace = True);  tmp_25 = None
        tmp_27 = torch.conv_transpose2d(tmp_26, in_10, None, (2, 2), (1, 1), (0, 0), 1, (1, 1));  tmp_26 = in_10 = None
        tmp_28 = torch.nn.functional.batch_norm(tmp_27, in_11, in_12, in_14, in_13, False, 0.1, 1e-05);  tmp_27 = in_11 = in_12 = in_14 = in_13 = None
        tmp_29 = torch.nn.functional.relu(tmp_28, inplace = True);  tmp_28 = None
        conv2d = torch.conv2d(tmp_29, in_16, in_15, (1, 1), (0, 0), (1, 1), 1);  tmp_29 = in_16 = in_15 = None
        tmp_31 = conv2d * 1.0;  conv2d = None
        tmp_32 = tmp_31.reshape(-1, 17, 4096);  tmp_31 = None
        tmp_33 = torch.nn.functional.softmax(tmp_32, dim = 2);  tmp_32 = None
        tmp_34 = tmp_33.reshape(-1, 17, 64, 64);  tmp_33 = None
        tmp_35 = tmp_34.mul(in_17);  in_17 = None
        tmp_36 = tmp_35.reshape(32, 17, -1);  tmp_35 = None
        tmp_37 = torch.sum(tmp_36, dim = 2, keepdim = True);  tmp_36 = None
        tmp_38 = tmp_34.mul(in_18);  in_18 = None
        tmp_39 = tmp_38.reshape(32, 17, -1);  tmp_38 = None
        tmp_40 = torch.sum(tmp_39, dim = 2, keepdim = True);  tmp_39 = None
        tmp_41 = torch.cat([tmp_37, tmp_40], dim = -1);  tmp_37 = tmp_40 = None
        return (tmp_34, tmp_41)
        