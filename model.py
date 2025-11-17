from neuralop.models import FNO
from neuralop.training import Trainer
from load_data import load_data
from burgers_res import burgers_res
import torch

def main():
	train_loader, test_loaders, data_processor = load_data()

	model = FNO(
		n_modes=(8,8),
		hidden_channels=16,
        in_channels=1,
        out_channels=1,
	)

	optimizer = torch.optim.Adam(model.parameters())	

	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

	mse = lambda output, **sample: torch.nn.MSELoss()(output, sample['y'].float())
	p_loss = lambda output, **sample: torch.mean(burgers_res(output)**2)
	def ic_loss(output, **sample):
		u0_pred = data_processor.out_normalizer.inverse_transform(output)[:, :, 0, :]
		u0_true = data_processor.in_normalizer.inverse_transform(sample['x'])[:, :, 0, :]
		return torch.mean((u0_pred - u0_true)**2)

	# train_loss = lambda o, **s: p_loss(o, **s)+ic_loss(o, **s)
	train_loss = lambda o, **s: ic_loss(o, **s) + mse(o, **s)
	eval_losses = {'mse': mse, 'ic_loss': train_loss}

	trainer = Trainer(
        model=model,
        n_epochs=100,
        data_processor=data_processor,
        verbose=True
    )
	
	trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=False,
        training_loss=train_loss,
        eval_losses=eval_losses,
    )

	model.save_checkpoint(save_folder='./params', save_name='pino')

if __name__ == "__main__":
	main()
