import argparse
import os
import shutil

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from detcon.datasets.s2c_data_module import S2cDataModule
from detcon.models import DetConB
import datetime

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def main(cfg_path: str, cfg: DictConfig) -> None:
    pl.seed_everything(0, workers=True)
    module = DetConB(**cfg.module)
    [print((k, v))  for k, v in cfg.module.items()]

    meta_df = pd.read_csv("/gpfs/scratch1/shared/ramaudruz/s2c_un/ssl4eo_s2_l1c_full_extract_metadata.csv")
    temp_var = meta_df['patch_id'].astype(str)
    meta_df['patch_id'] = temp_var.map(lambda x: (7 - len(x)) * '0' + x)
    meta_df['file_name'] = meta_df['patch_id'] + '/' + meta_df['timestamp']

    datamodule = S2cDataModule(
        batch_size=cfg['datamodule']['batch_size'],
        meta_df=meta_df,
        num_workers=cfg['datamodule']['num_workers']
    )

    module.n_iterations = cfg['trainer']['max_epochs'] * len(datamodule)

    timestamp = datetime.datetime.now().__str__().split('.')[0][:-3].replace(' ', '_').replace(':', '-')
    print(f'Timestamp: {timestamp}')
    checkpoint_dir = f'/gpfs/work5/0/prjs0790/data/run_outputs/checkpoints/odin/run_{timestamp}/'
    os.makedirs(checkpoint_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='ckp-{epoch:02d}',
        save_top_k=-1,
        verbose=True,
        every_n_epochs=1
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    callbacks = [checkpoint_callback, lr_monitor]

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=WandbLogger(log_model="all"),
        accumulate_grad_batches=3,
        gradient_clip_val=1,
        **cfg.trainer
    )
    trainer.fit(model=module, datamodule=datamodule)
    shutil.copyfile(cfg_path, os.path.join(checkpoint_dir, "config.yaml"))

class FakeArgs:
    cfg = 'conf/pretrain_s2c.yaml'


if __name__ == "__main__":

    args = FakeArgs()

    cfg = OmegaConf.load(args.cfg)
    main(args.cfg, cfg)
