from neuralop.data.datasets import Burgers1dTimeDataset
from torch.utils.data import DataLoader

batch_size = 1
test_batch_size = batch_size

def load_data():
	burgers_dataset = Burgers1dTimeDataset(
		root_dir='./data',
		n_train=16,
		n_tests=[16],
		batch_size=batch_size,
		test_batch_sizes=[test_batch_size],
		train_resolution=16,
		test_resolutions=[16],
		temporal_subsample=1,
		spatial_subsample=1,
	)

	train_loader = DataLoader(
		burgers_dataset.train_db, batch_size=batch_size, shuffle=True
	)

	test_loaders = {
		16: DataLoader(
			burgers_dataset.test_dbs[16], batch_size=test_batch_size, shuffle=False
		)
	}

	data_processor = burgers_dataset.data_processor

	return train_loader, test_loaders, data_processor 
