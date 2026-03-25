import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, w_14 : torch.Tensor, w_15 : torch.Tensor, w_16 : torch.Tensor, w_17 : torch.Tensor, w_18 : torch.Tensor, w_19 : torch.Tensor, w_20 : torch.Tensor, w_21 : torch.Tensor, w_22 : torch.Tensor, w_23 : torch.Tensor, w_24 : torch.Tensor, w_25 : torch.Tensor, w_26 : torch.Tensor, w_27 : torch.Tensor, w_28 : torch.Tensor, w_29 : torch.Tensor, w_30 : torch.Tensor, w_31 : torch.Tensor, w_32 : torch.Tensor, w_33 : torch.Tensor, in_1 : torch.Tensor):
        tmp_36 = in_0[1]
        tmp_37 = in_0[0];  in_0 = None
        tmp_38 = w_5.index_select(-2, tmp_37);  w_5 = tmp_37 = None
        tmp_39 = tmp_36.view((-1, 1));  tmp_36 = None
        tmp_40 = tmp_39.expand_as(tmp_38);  tmp_39 = None
        tmp_41 = tmp_38.new_zeros((1000, 128))
        tmp_42 = tmp_41.scatter_add_(0, tmp_40, tmp_38);  tmp_41 = tmp_40 = tmp_38 = None
        tmp_43 = tmp_42 + w_4;  tmp_42 = w_4 = None
        linear = torch.nn.functional.linear(tmp_43, w_1, w_0);  w_1 = w_0 = None
        tmp_45 = tmp_43 + linear;  tmp_43 = linear = None
        linear_1 = torch.nn.functional.linear(in_1, w_33, w_32);  in_1 = w_33 = w_32 = None
        tmp_47 = torch.nn.functional.dropout(linear_1, p = 0.0, training = False);  linear_1 = None
        tmp_48 = tmp_45 + tmp_47;  tmp_45 = None
        linear_2 = torch.nn.functional.linear(tmp_47, w_3, w_2);  tmp_47 = w_3 = w_2 = None
        tmp_50 = tmp_48 + linear_2;  tmp_48 = linear_2 = None
        tmp_51 = tmp_50.relu_();  tmp_50 = None
        linear_3 = torch.nn.functional.linear(tmp_51, w_7, w_6);  tmp_51 = w_7 = w_6 = None
        tmp_53 = torch.nn.functional.relu(linear_3, inplace = False);  linear_3 = None
        tmp_54 = torch.nn.functional.batch_norm(tmp_53, w_16, w_17, w_19, w_18, False, 0.1, 1e-05);  tmp_53 = w_16 = w_17 = w_19 = w_18 = None
        tmp_55 = torch.nn.functional.dropout(tmp_54, p = 0.0, training = False);  tmp_54 = None
        linear_4 = torch.nn.functional.linear(tmp_55, w_9, w_8);  tmp_55 = w_9 = w_8 = None
        tmp_57 = torch.nn.functional.relu(linear_4, inplace = False);  linear_4 = None
        tmp_58 = torch.nn.functional.batch_norm(tmp_57, w_20, w_21, w_23, w_22, False, 0.1, 1e-05);  tmp_57 = w_20 = w_21 = w_23 = w_22 = None
        tmp_59 = torch.nn.functional.dropout(tmp_58, p = 0.0, training = False);  tmp_58 = None
        linear_5 = torch.nn.functional.linear(tmp_59, w_11, w_10);  tmp_59 = w_11 = w_10 = None
        tmp_61 = torch.nn.functional.relu(linear_5, inplace = False);  linear_5 = None
        tmp_62 = torch.nn.functional.batch_norm(tmp_61, w_24, w_25, w_27, w_26, False, 0.1, 1e-05);  tmp_61 = w_24 = w_25 = w_27 = w_26 = None
        tmp_63 = torch.nn.functional.dropout(tmp_62, p = 0.0, training = False);  tmp_62 = None
        linear_6 = torch.nn.functional.linear(tmp_63, w_13, w_12);  tmp_63 = w_13 = w_12 = None
        tmp_65 = torch.nn.functional.relu(linear_6, inplace = False);  linear_6 = None
        tmp_66 = torch.nn.functional.batch_norm(tmp_65, w_28, w_29, w_31, w_30, False, 0.1, 1e-05);  tmp_65 = w_28 = w_29 = w_31 = w_30 = None
        tmp_67 = torch.nn.functional.dropout(tmp_66, p = 0.0, training = False);  tmp_66 = None
        linear_7 = torch.nn.functional.linear(tmp_67, w_15, w_14);  tmp_67 = w_15 = w_14 = None
        tmp_69 = torch.nn.functional.dropout(linear_7, p = 0.0, training = False);  linear_7 = None
        return (tmp_69,)
        