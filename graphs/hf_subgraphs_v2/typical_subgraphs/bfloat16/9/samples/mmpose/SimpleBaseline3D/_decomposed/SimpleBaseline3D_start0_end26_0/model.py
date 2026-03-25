import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, w_14 : torch.Tensor, w_15 : torch.Tensor, w_16 : torch.Tensor, w_17 : torch.Tensor, w_18 : torch.Tensor, w_19 : torch.Tensor, w_20 : torch.Tensor, w_21 : torch.Tensor, w_22 : torch.Tensor, w_23 : torch.Tensor, w_24 : torch.Tensor, w_25 : torch.Tensor, w_26 : torch.Tensor):
        conv1d = torch.conv1d(in_0, w_4, None, (1,), (0,), (1,), 1);  in_0 = w_4 = None
        tmp_29 = torch.nn.functional.batch_norm(conv1d, w_0, w_1, w_3, w_2, False, 0.1, 1e-05);  conv1d = w_0 = w_1 = w_3 = w_2 = None
        tmp_30 = torch.nn.functional.relu(tmp_29, inplace = True);  tmp_29 = None
        tmp_31 = torch.nn.functional.dropout(tmp_30, 0.5, False, False);  tmp_30 = None
        conv1d_1 = torch.conv1d(tmp_31, w_9, None, (1,), (0,), (1,), 1);  w_9 = None
        tmp_33 = torch.nn.functional.batch_norm(conv1d_1, w_5, w_6, w_8, w_7, False, 0.1, 1e-05);  conv1d_1 = w_5 = w_6 = w_8 = w_7 = None
        tmp_34 = torch.nn.functional.relu(tmp_33, inplace = True);  tmp_33 = None
        tmp_35 = torch.nn.functional.dropout(tmp_34, 0.5, False, False);  tmp_34 = None
        conv1d_2 = torch.conv1d(tmp_35, w_14, None, (1,), (0,), (1,), 1);  tmp_35 = w_14 = None
        tmp_37 = torch.nn.functional.batch_norm(conv1d_2, w_10, w_11, w_13, w_12, False, 0.1, 1e-05);  conv1d_2 = w_10 = w_11 = w_13 = w_12 = None
        tmp_38 = torch.nn.functional.relu(tmp_37, inplace = True);  tmp_37 = None
        tmp_39 = torch.nn.functional.dropout(tmp_38, 0.5, False, False);  tmp_38 = None
        tmp_40 = tmp_31[(slice(None, None, None), slice(None, None, None), slice(0, 1, None))];  tmp_31 = None
        tmp_41 = tmp_39 + tmp_40;  tmp_39 = tmp_40 = None
        conv1d_3 = torch.conv1d(tmp_41, w_19, None, (1,), (0,), (1,), 1);  w_19 = None
        tmp_43 = torch.nn.functional.batch_norm(conv1d_3, w_15, w_16, w_18, w_17, False, 0.1, 1e-05);  conv1d_3 = w_15 = w_16 = w_18 = w_17 = None
        tmp_44 = torch.nn.functional.relu(tmp_43, inplace = True);  tmp_43 = None
        tmp_45 = torch.nn.functional.dropout(tmp_44, 0.5, False, False);  tmp_44 = None
        conv1d_4 = torch.conv1d(tmp_45, w_24, None, (1,), (0,), (1,), 1);  tmp_45 = w_24 = None
        tmp_47 = torch.nn.functional.batch_norm(conv1d_4, w_20, w_21, w_23, w_22, False, 0.1, 1e-05);  conv1d_4 = w_20 = w_21 = w_23 = w_22 = None
        tmp_48 = torch.nn.functional.relu(tmp_47, inplace = True);  tmp_47 = None
        tmp_49 = torch.nn.functional.dropout(tmp_48, 0.5, False, False);  tmp_48 = None
        tmp_50 = tmp_41[(slice(None, None, None), slice(None, None, None), slice(0, 1, None))];  tmp_41 = None
        tmp_51 = tmp_49 + tmp_50;  tmp_49 = tmp_50 = None
        conv1d_5 = torch.conv1d(tmp_51, w_26, w_25, (1,), (0,), (1,), 1);  tmp_51 = w_26 = w_25 = None
        tmp_53 = conv1d_5.reshape(-1, 16, 3);  conv1d_5 = None
        return (tmp_53,)
        