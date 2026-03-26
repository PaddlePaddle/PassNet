class Program_weight_tensor_meta_key_layer:
	name = "in_0"
	original_name = "key_layer"
	shape = [24, 1, 256, 32]
	dtype = "torch.bfloat16"
	device = "cuda:0"
	mean = 0.003
	std = 0.023


class Program_weight_tensor_meta_query_layer:
	name = "in_1"
	original_name = "query_layer"
	shape = [24, 1, 16384, 32]
	dtype = "torch.bfloat16"
	device = "cuda:0"
	mean = -0.005
	std = 0.017


class Program_weight_tensor_meta_value_layer:
	name = "in_2"
	original_name = "value_layer"
	shape = [24, 1, 256, 32]
	dtype = "torch.bfloat16"
	device = "cuda:0"
	mean = -0.005
	std = 0.022
