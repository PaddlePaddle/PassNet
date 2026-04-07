class Program_weight_tensor_meta_attention_scores:
	name = "in_0"
	original_name = "attention_scores"
	shape = [2, 12, 1024, 1024]
	dtype = "torch.bfloat16"
	device = "cuda:0"
	mean = -0.002
	std = 0.099


class Program_weight_tensor_meta_extended_attention_mask_2:
	name = "in_1"
	original_name = "extended_attention_mask_2"
	shape = [2, 1, 1, 1024]
	dtype = "torch.bfloat16"
	device = "cuda:0"
	mean = 0.000
	std = 0.100


class Program_weight_tensor_meta_value_layer:
	name = "in_2"
	original_name = "value_layer"
	shape = [2, 12, 1024, 64]
	dtype = "torch.bfloat16"
	device = "cuda:0"
	mean = 0.005
	std = 0.112
