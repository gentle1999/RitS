# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import logging
from pathlib import Path

import torch
import hydra
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from megalodon.data.batch_preprocessor import BatchPreProcessor
from megalodon.data.ts_batch_preprocessor import TsBatchPreProcessor
from megalodon.data.molecule_datamodule import MoleculeDataModule
from megalodon.data.statistics import Statistics
from megalodon.metrics.molecule_evaluation_callback import MoleculeEvaluationCallback
from megalodon.metrics.conformer_evaluation_callback import ConformerEvaluationCallback
from megalodon.metrics.ts_evaluation_callback import TransitionStatesEvaluationCallback
from megalodon.models.module import Graph3DInterpolantModel


@hydra.main(version_base=None, config_path="conf", config_name=None)
def main(cfg: DictConfig) -> None:
    """
    This is the main function conducting data loading and model training.
    """
    logging.info("\n\n************** Experiment Configuration ***********")
    pl.seed_everything(cfg.train.seed)
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")
    cfg.outdir = os.path.join(cfg.outdir, cfg.run_name)
    os.makedirs(cfg.outdir, exist_ok=True)
    os.makedirs(os.path.join(cfg.outdir, 'checkpoints'), exist_ok=True)
    loss_fn = None
    eval_type = OmegaConf.select(cfg, "evaluation.type", default=None)
    model_name = OmegaConf.select(cfg, "dynamics.model_name", default=None)
    is_ts_run = (eval_type == "transition_states") or (model_name == "megav3ts")

    if is_ts_run:
        ts_ratio = OmegaConf.select(cfg, "data.ts_ratio", default=1.0)
        logging.info(f"TS ratio (curriculum learning): {ts_ratio}")
        batch_preprocessor = TsBatchPreProcessor(
            aug_rotations=cfg.data.aug_rotations,
            scale_coords=cfg.data.scale_coords,
            ts_ratio=ts_ratio
        )
    else:
        batch_preprocessor = BatchPreProcessor(aug_rotations=cfg.data.aug_rotations,
                                           scale_coords=cfg.data.scale_coords)

    resume_from = OmegaConf.select(cfg, "resume", default=None)
    finetune_from = OmegaConf.select(cfg, "finetune_from", default=None)
    # ``init_from_ckpt`` is the name used by the upstream fork; retain it as
    # an alias for the local ``finetune_from`` option.
    init_from_ckpt = OmegaConf.select(cfg, "init_from_ckpt", default=None)
    if finetune_from and init_from_ckpt:
        raise ValueError("finetune_from and init_from_ckpt are mutually exclusive")
    finetune_from = finetune_from or init_from_ckpt
    if resume_from and finetune_from:
        raise ValueError("resume and finetune_from are mutually exclusive")

    if resume_from:
        if os.path.isdir(resume_from):
            resume_from = f"{resume_from}/last.ckpt"
            ckpt = "last"
        else:
            ckpt = resume_from
        pl_module = Graph3DInterpolantModel.load_from_checkpoint(resume_from,
                                                                 loss_fn=loss_fn,
                                                                 loss_params=cfg.loss,
                                                                 interpolant_params=cfg.interpolant,
                                                                 sampling_params=cfg.sample,
                                                                 batch_preprocessor=batch_preprocessor,
                                                                 weights_only=False)
    elif finetune_from:
        if os.path.isdir(finetune_from):
            finetune_from = f"{finetune_from}/last.ckpt"
        pl_module = Graph3DInterpolantModel.load_from_checkpoint(
            finetune_from,
            loss_fn=loss_fn,
            optimizer_params=cfg.optimizer,
            lr_scheduler_params=cfg.lr_scheduler,
            dynamics_params=cfg.dynamics,
            loss_params=cfg.loss,
            interpolant_params=cfg.interpolant,
            sampling_params=cfg.sample,
            self_cond_params=OmegaConf.select(cfg, "self_conditioning", default=None),
            ema=OmegaConf.select(cfg, "ema", default=True),
            batch_preprocessor=batch_preprocessor,
            weights_only=False,
        )
        # Fine-tuning starts a fresh Lightning run with new optimizer/scheduler state.
        ckpt = None
    else:
        pl_module = Graph3DInterpolantModel(
            loss_params=cfg.loss,
            optimizer_params=cfg.optimizer,
            lr_scheduler_params=cfg.lr_scheduler,
            dynamics_params=cfg.dynamics,
            interpolant_params=cfg.interpolant,
            sampling_params=cfg.sample,
            self_cond_params=OmegaConf.select(cfg, "self_conditioning", default=None),
            ema=OmegaConf.select(cfg, "ema", default=True),
            loss_fn=loss_fn,
            batch_preprocessor=batch_preprocessor
        )
        ckpt = None
    wandb_resume = cfg.wandb_params.resume if "resume" in cfg.wandb_params else "allow"
    wandb_logger = pl.loggers.WandbLogger(
        save_dir=cfg.outdir,
        project=cfg.wandb_params.project,
        group=cfg.wandb_params.group,
        name=cfg.run_name,
        id=cfg.run_name,
        resume=wandb_resume,
        mode=cfg.wandb_params.mode,
    )
    tensorboard_logger = TensorBoardLogger(
        save_dir=cfg.outdir,
        name="tensorboard",
        version="",
    )
    loggers = [wandb_logger, tensorboard_logger]
    logger_hparams = OmegaConf.to_container(cfg, resolve=True)
    for logger in loggers:
        # Lightning adds the module hparams later; keep this mutable so resumed
        # runs can merge both sets without OmegaConf struct-key errors.
        logger.log_hyperparams(logger_hparams)

    datamodule = MoleculeDataModule(cfg.data.dataset_root,
                                    cfg.data.processed_folder,
                                    cfg.data.batch_size,
                                    cfg.data.data_loader_type,
                                    cfg.data.inference_batch_size,
                                    validation_data_loader_type=OmegaConf.select(
                                        cfg, "data.validation_data_loader_type", default=None
                                    ))

    lr_monitor = LearningRateMonitor(logging_interval="step")

    last_checkpoint_callback = ModelCheckpoint(
        dirpath=Path(cfg.outdir, 'checkpoints'),
        save_last=True,
        save_on_train_epoch_end=True,
    )
    periodic_checkpoint_callback = None
    checkpoint_every_n_train_steps = OmegaConf.select(
        cfg, "train.checkpoint_every_n_train_steps", default=0
    )
    if checkpoint_every_n_train_steps > 0:
        periodic_checkpoint_callback = ModelCheckpoint(
            dirpath=Path(cfg.outdir, 'checkpoints'),
            save_top_k=-1,
            every_n_train_steps=checkpoint_every_n_train_steps,
            save_on_train_epoch_end=False,
            filename="periodic-{epoch}-{step}",
        )
    best_checkpoint_callback = ModelCheckpoint(
        dirpath=Path(cfg.outdir, 'checkpoints'),
        save_top_k=5,
        monitor=cfg.train.checkpoint_monitor,
        mode=cfg.train.checkpoint_monitor_mode,
        filename="best-{epoch}-{step}",
    )
    # Additional checkpoint based on train loss as backup (saves best 5 by train loss epoch avg)
    train_loss_checkpoint_callback = ModelCheckpoint(
        dirpath=Path(cfg.outdir, 'checkpoints'),
        save_top_k=5,
        monitor="train/loss_epoch",
        mode="min",
        save_on_train_epoch_end=True,
        filename="best_train-{epoch}-{step}",
    )
    evaluation_callback = None
    if eval_type == "molecules":
        energy_metrics_args = OmegaConf.to_container(cfg.evaluation.energy_metrics_args,
                                                 resolve=True) if cfg.evaluation.energy_metrics_args is not None else None
        statistics = Statistics.load_statistics(
            statistics_dir=f"{cfg.data.dataset_root}/{cfg.data.processed_folder}",
            split_name="train")
        evaluation_callback = MoleculeEvaluationCallback(
            n_graphs=cfg.evaluation.n_molecules,
            batch_size=cfg.evaluation.batch_size,
            timesteps=cfg.evaluation.timesteps,
            train_smiles=datamodule.train_dataset.smiles,
            statistics=statistics,
            compute_2D_metrics=cfg.evaluation.compute_2D_metrics,
            compute_3D_metrics=cfg.evaluation.compute_3D_metrics,
            compute_train_data_metrics=cfg.evaluation.compute_train_data_metrics,
            compute_energy_metrics=cfg.evaluation.compute_energy_metrics,
            energy_metrics_args=energy_metrics_args,
            scale_coords=cfg.evaluation.scale_coords,
            preserve_aromatic=OmegaConf.select(cfg.evaluation, "preserve_aromatic", default=True)
        )

    elif eval_type == "conformers":
        energy_metrics_args = OmegaConf.to_container(cfg.evaluation.energy_metrics_args,
                                                 resolve=True) if cfg.evaluation.energy_metrics_args is not None else None
        statistics = Statistics.load_statistics(
            statistics_dir=f"{cfg.data.dataset_root}/{cfg.data.processed_folder}",
            split_name="train")
        evaluation_callback = ConformerEvaluationCallback(
            statistics=statistics,
            max_molecules=cfg.evaluation.max_molecules,
            timesteps=cfg.evaluation.timesteps,
            compute_3D_metrics=cfg.evaluation.compute_3D_metrics,
            compute_energy_metrics=cfg.evaluation.compute_energy_metrics,
            energy_metrics_args=energy_metrics_args,
            scale_coords=cfg.evaluation.scale_coords,
            compute_stereo_metrics=cfg.evaluation.compute_stereo_metrics
        )
    elif eval_type == "transition_states":
        evaluation_callback = TransitionStatesEvaluationCallback(
            max_molecules=cfg.evaluation.max_molecules,
            timesteps=cfg.evaluation.timesteps,
            scale_coords=cfg.evaluation.scale_coords,
            save_dir=cfg.evaluation.save_dir
        )
    elif eval_type is not None:
        raise NotImplementedError

    if 'num_nodes' in cfg.train:
        num_nodes = cfg.train.num_nodes
    else:
        num_nodes = 1

    callbacks = [
        lr_monitor,
        last_checkpoint_callback,
        best_checkpoint_callback,
        train_loss_checkpoint_callback,
    ]
    if periodic_checkpoint_callback is not None:
        callbacks.append(periodic_checkpoint_callback)
    early_stopping_patience = OmegaConf.select(
        cfg, "train.early_stopping.patience", default=None
    )
    if early_stopping_patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor=OmegaConf.select(
                    cfg,
                    "train.early_stopping.monitor",
                    default=cfg.train.checkpoint_monitor,
                ),
                mode=OmegaConf.select(
                    cfg,
                    "train.early_stopping.mode",
                    default=cfg.train.checkpoint_monitor_mode,
                ),
                min_delta=OmegaConf.select(
                    cfg, "train.early_stopping.min_delta", default=0.0
                ),
                patience=early_stopping_patience,
                check_finite=True,
                verbose=True,
            )
        )
    if evaluation_callback is not None:
        callbacks.insert(1, evaluation_callback)

    trainer = pl.Trainer(
        max_epochs=cfg.train.n_epochs,
        logger=loggers,
        callbacks=callbacks,
        enable_progress_bar=cfg.train.enable_progress_bar,
        accelerator='gpu',
        devices=cfg.train.gpus,
        num_nodes=num_nodes,
        strategy=('ddp' if cfg.train.gpus > 1 else 'auto'),
        check_val_every_n_epoch=cfg.train.val_freq,
        gradient_clip_val=cfg.train.gradient_clip_value,
        log_every_n_steps=cfg.train.log_freq,  # for train steps
        max_steps=OmegaConf.select(cfg, "train.max_steps", default=-1),
        limit_train_batches=OmegaConf.select(cfg, "train.limit_train_batches", default=1.0),
        limit_val_batches=OmegaConf.select(cfg, "train.limit_val_batches", default=1.0),
        num_sanity_val_steps=OmegaConf.select(cfg, "train.num_sanity_val_steps", default=0),
    )

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    trainer.fit(model=pl_module, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=ckpt)

    best_model_path = best_checkpoint_callback.best_model_path
    if best_model_path:
        best_alias = Path(cfg.outdir, "checkpoints", "best.ckpt")
        alias_tmp = best_alias.with_suffix(".ckpt.tmp")
        alias_tmp.unlink(missing_ok=True)
        alias_tmp.symlink_to(Path(best_model_path).name)
        alias_tmp.replace(best_alias)
        logging.info(f"Best validation checkpoint: {best_model_path}")


if __name__ == "__main__":
    main()
