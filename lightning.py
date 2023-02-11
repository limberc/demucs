import warnings

import hydra
import torch
import torch.nn as nn
from hydra.utils import to_absolute_path
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from torch.utils.data import DataLoader

import demucs.augment as augment
from demucs import states
from demucs.apply import apply_model
from demucs.evaluate import new_sdr
from demucs.svd import svd_penalty
from demucs.train import get_datasets, get_model, get_optimizer

torch.set_float32_matmul_precision('medium')
warnings.filterwarnings("ignore", category=PossibleUserWarning)


class GetLoss(nn.Module):
    def __init__(self, loss_n, quantizer, diffq):
        super().__init__()
        self.loss_n = loss_n
        self.quantizer = quantizer
        self.diffq = diffq
        if loss_n == 'l1':
            self.loss = nn.L1Loss(reduction='none')
        elif loss_n == 'mse':
            self.loss = nn.MSELoss(reduction='none')
        else:
            raise ValueError(f"Invalid loss {loss_n}")

    def forward(self, estimate, sources):
        loss = self.loss(estimate, sources)
        dims = tuple(range(2, sources.dim()))
        if self.loss_n == 'l1':
            loss = loss.mean(dims)
        elif self.loss_n == 'mse':
            loss = loss.mean(dims) ** 0.5
        loss = loss.mean(0)
        ms = 0
        if self.quantizer is not None:
            ms = self.quantizer.model_size()
        if self.diffq:
            loss += self.diffq * ms
        return loss, ms


def get_augument(args):
    augments = [augment.Shift(shift=int(args.dset.samplerate * args.dset.shift),
                              same=args.augment.shift_same)]
    if args.augment.flip:
        augments += [augment.FlipChannels(), augment.FlipSign()]
    for aug in ['scale', 'remix']:
        kw = getattr(args.augment, aug)
        if kw.proba:
            augments.append(getattr(augment, aug.capitalize())(**kw))
    return torch.nn.Sequential(*augments)


class DemucsDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.num_worker = args.misc.num_workers
        self.train_set, self.val_set = get_datasets(self.args)

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=self.num_worker, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, shuffle=False, num_workers=self.num_worker, drop_last=True)


class DemucsModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters({
            'model_name': args.model
        })
        self.model = get_model(args)
        self.batch_size = args.batch_size
        self.optimizer = get_optimizer(self.model, args)
        # self.averager = EMA()
        # self.emas = {'batch': [], 'epoch': []}
        # for kind in self.emas.keys():
        #     decays = getattr(args.ema, kind)
        #     device = self.device if kind == 'batch' else 'cpu'
        #     if decays:
        #         for decay in decays:
        #             self.emas[kind].append(ModelEMA(self.model, decay, device=device))
        self.augument = get_augument(args)
        self.split = args.test.split
        self.quantizer = states.get_quantizer(self.model, args.quant, self.optimizer)
        # self.loss = GetLoss(args.optim.loss, args.weights, self.quantizer, args.quant.diffq)
        # I have disabled the weights argument in GetLoss.
        self.loss = GetLoss(args.optim.loss, self.quantizer, args.quant.diffq)
        self.svd = args.svd

    def training_step(self, batch, batch_idx):
        sources = self.augument(batch)
        mix = sources.sum(dim=1)
        estimate = self.model(mix)  # same shape as sources.
        if hasattr(self.model, 'transform_target'):
            sources = self.model.transform_target(mix, sources)
        assert estimate.shape == sources.shape, (estimate.shape, sources.shape)
        loss, ms = self.loss(estimate, sources)
        loss = loss.sum()
        del estimate
        if self.svd.penalty > 0:
            kw = dict(self.svd)
            kw.pop('penalty')
            penalty_val = svd_penalty(self.model, **kw)
            loss += self.svd.penalty * penalty_val
            self.log('train_penalty', penalty_val, logger=True, on_step=True, on_epoch=True)
        # for k, source in enumerate(self.model.sources):
        #     loss_val = loss.data[k].item()
        #     self.log(f'train_reco_{source}', loss_val, logger=True, on_step=True, on_epoch=True)
        loss = loss.sum()
        self.log('train_loss', loss, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mix = batch[:, 0]
        sources = batch[:, 1:]
        estimate = apply_model(self.model, mix, split=self.split, overlap=0)
        assert estimate.shape == sources.shape, (estimate.shape, sources.shape)
        loss, ms = self.loss(estimate, sources)

        nsdrs = new_sdr(sources, estimate.detach()).mean(0)
        del estimate

        total = 0
        for source, nsdr in zip(self.model.sources, nsdrs):
            self.log(f'nsdr_{source}', nsdr, on_step=True, on_epoch=True)
            total += nsdr
        nsdr = total

        # for k, source in enumerate(self.model.sources):
        #     loss_val = loss.data[k].item()
        #     self.log(f'val_reco_{source}', loss_val, logger=True, on_step=True, on_epoch=True)
        loss = loss.sum()
        self.log('val_loss', loss, logger=True, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return self.optimizer


@hydra.main(version_base="1.3", config_path="./conf", config_name="config.yaml")
def main(args):
    global __file__
    __file__ = to_absolute_path(__file__)
    for attr in ["musdb", "wav", "metadata"]:
        val = getattr(args.dset, attr)
        if val is not None:
            setattr(args.dset, attr, to_absolute_path(val))
    hydra_args = hydra.utils.instantiate(args)
    # logger = WandbLogger('baseline', project="SourceSep")
    dm = DemucsDataModule(hydra_args)
    model = DemucsModule(hydra_args)
    trainer = Trainer(
        devices=-1,
        accelerator='auto',
        max_epochs=args.epochs,
        benchmark=True,
        precision=16,
        strategy='ddp',
    )
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    main()
