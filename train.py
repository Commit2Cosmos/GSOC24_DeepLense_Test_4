from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from utils import DiffusionLensDataset, DataLoader
import lightning as L
import torch


if __name__ == '__main__':

    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels=1,
        flash_attn = True
    )


    diffusion = GaussianDiffusion(
        model,
        image_size = 144,
        timesteps = 1000,           # number of steps
        sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    )


    class LitDiffusion(L.LightningModule):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.diffusion = diffusion

        def forward(self, x):
            return self.diffusion(x)

        def training_step(self, batch):
            loss = self.diffusion(batch)
            self.log('train_loss', loss)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters())
        
    p = "./lightning_logs/version_0"
    diff_model = LitDiffusion.load_from_checkpoint(f"{p}/checkpoints/epoch=0-step=10000.ckpt", map_location=torch.device('cpu'))
    trainer = L.Trainer(max_epochs=20)

    ds = DiffusionLensDataset("./data")
    dl = DataLoader(ds, batch_size=4)

    trainer.fit(diff_model, dl)