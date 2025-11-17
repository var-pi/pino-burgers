from load_data import load_data
from neuralop.models import FNO
import torch
import matplotlib.pyplot as plt

def main():
	train_loader, test_loaders, data_processor = load_data()
	model = FNO.from_checkpoint(save_folder='./params/', save_name='pino')
	model.eval()

	test_loader = test_loaders[16]
	it = iter(test_loader)
	next(it)
	next(it)
	sample = data_processor.preprocess(next(it))  # take first batch

	x = sample['x'][0:1]  # take first example in batch
	y_true = data_processor.out_normalizer.inverse_transform(sample['y'][0:1])

	with torch.no_grad():
		y_pred = model(x)
		y_pred = data_processor.out_normalizer.inverse_transform(y_pred)
	
	fig, [ax_true, ax_pred, ax_dif] = plt.subplots(ncols=3)
	im_true = ax_true.imshow(y_true.squeeze())
	plt.colorbar(im_true, ax=ax_true, fraction=0.046, pad=0.04)
	im_pred = ax_pred.imshow(y_pred.squeeze())
	plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)
	im_dif = ax_dif.imshow(y_true.squeeze()-y_pred.squeeze(), cmap='bwr')
	plt.colorbar(im_dif, ax=ax_dif, fraction=0.046, pad=0.04)

	plt.show()		
	
if __name__ == "__main__":
    main()
