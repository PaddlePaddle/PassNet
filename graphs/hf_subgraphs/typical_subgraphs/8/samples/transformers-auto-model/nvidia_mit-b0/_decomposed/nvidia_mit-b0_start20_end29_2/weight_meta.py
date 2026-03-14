class Program_weight_tensor_meta_key_layer:
	name = "in_0"
	original_name = "key_layer"
	shape = [64, 1, 256, 32]
	dtype = "torch.float32"
	device = "cuda:0"
	mean = 0.002
	std = 0.020


class Program_weight_tensor_meta_query_layer:
	name = "in_1"
	original_name = "query_layer"
	shape = [64, 1, 16384, 32]
	dtype = "torch.float32"
	device = "cuda:0"
	mean = -0.001
	std = 0.024


class Program_weight_tensor_meta_value_layer:
	name = "in_2"
	original_name = "value_layer"
	shape = [64, 1, 256, 32]
	dtype = "torch.float32"
	device = "cuda:0"
	mean = -0.004
	std = 0.020
