import os
import torch
import lightning as L

from torch import nn
from torch.utils.data import DataLoader
from datetime import timedelta

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from zclip import ZClipLightningCallback

from ..nn.model import StateEmbeddingModel
from ..data import H5adSentenceDataset, VCIDatasetSentenceCollator
from ..train.callbacks import LogLR, ProfilerCallback, ResumeCallback, EMACallback, PerfProfilerCallback
from ..utils import get_latest_checkpoint, get_embedding_cfg, get_dataset_cfg


def get_embeddings(cfg):
    # Load in ESM2 embeddings and special tokens
    all_pe = torch.load(get_embedding_cfg(cfg).all_embeddings, weights_only=False)
    if isinstance(all_pe, dict):
        all_pe = torch.vstack(list(all_pe.values()))

    all_pe = all_pe.cuda()
    return all_pe


def main(cfg):
    print(f"Starting training with Embedding {cfg.embeddings.current} and dataset {cfg.dataset.current}")
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["NCCL_LAUNCH_TIMEOUT"] = str(cfg.experiment.ddp_timeout)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    TOTAL_N_CELL = cfg.dataset.num_cells
    EPOCH_LENGTH = int(TOTAL_N_CELL // cfg.model.batch_size // 24)
    # ? not sure why this needs to be included but seems empirical?? no clue why this is 6
    warmup_steps = EPOCH_LENGTH * 6

    train_dataset_sentence_collator = VCIDatasetSentenceCollator(cfg, is_train=True)
    # validation should not do augmentations
    val_dataset_sentence_collator = VCIDatasetSentenceCollator(cfg, is_train=False)

    generator = torch.Generator()
    generator.manual_seed(cfg.dataset.seed)

    if get_dataset_cfg(cfg).ds_type == "h5ad":
        DatasetClass = H5adSentenceDataset
    else:
        raise ValueError(f"Unknown dataset type: {get_dataset_cfg(cfg).ds_type}")

    # Training dataloader
    train_dataset = DatasetClass(cfg)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.model.batch_size,
        shuffle=True,
        collate_fn=train_dataset_sentence_collator,
        num_workers=cfg.dataset.num_train_workers,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=4,
        generator=generator,
    )

    val_dataset = DatasetClass(cfg, test=True)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.model.batch_size,
        shuffle=True,
        collate_fn=val_dataset_sentence_collator,
        num_workers=cfg.dataset.num_val_workers,
        persistent_workers=True,
        generator=generator,
    )

    model = StateEmbeddingModel(
        token_dim=get_embedding_cfg(cfg).size,
        d_model=cfg.model.emsize,
        nhead=cfg.model.nhead,
        d_hid=cfg.model.d_hid,
        nlayers=cfg.model.nlayers,
        output_dim=cfg.model.output_dim,
        dropout=cfg.model.dropout,
        warmup_steps=warmup_steps,
        compiled=False,
        max_lr=cfg.optimizer.max_lr,
        emb_size=get_embedding_cfg(cfg).size,
        collater=val_dataset_sentence_collator,
        cfg=cfg,
    )
    model = model.cuda()
    all_pe = get_embeddings(cfg)
    all_pe.requires_grad = False
    model.pe_embedding = nn.Embedding.from_pretrained(all_pe)

    model = model.train()

    run_name, chk = get_latest_checkpoint(cfg)
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=cfg.experiment.checkpoint.every_n_train_steps,
        dirpath=os.path.join(cfg.experiment.checkpoint.path, cfg.experiment.name),
        filename=f"{run_name}" + "-{epoch}-{step}",
        save_last=True,
        save_top_k=cfg.experiment.checkpoint.save_top_k,
        monitor=cfg.experiment.checkpoint.monitor,
    )

    if cfg.wandb.enable:
        exp_logger = WandbLogger(project=cfg.wandb.project, name=cfg.experiment.name)
        exp_logger.watch(model, log_freq=1000)
    else:
        exp_logger = None

    callbacks = [checkpoint_callback, LogLR(100), ResumeCallback(cfg), PerfProfilerCallback()]

    if getattr(cfg.model, "ema", False):
        ema_decay = getattr(cfg.model, "ema_decay", 0.999)
        callbacks.append(EMACallback(decay=ema_decay))

    if getattr(cfg.optimizer, "zclip", False):
        zclip_cb = ZClipLightningCallback(
            mode="zscore", alpha=0.97, z_thresh=2.5, clip_option="adaptive_scaling", max_grad_norm=1.0, clip_factor=1.0
        )
        callbacks.append(zclip_cb)

    max_steps = -1
    if cfg.experiment.profile.enable_profiler:
        callbacks.append(ProfilerCallback(cfg=cfg))
        max_steps = cfg.experiment.profile.max_steps

    val_interval = int(cfg.experiment.val_check_interval * cfg.experiment.num_gpus_per_node * cfg.experiment.num_nodes)
    trainer = L.Trainer(
        max_epochs=cfg.experiment.num_epochs,
        max_steps=max_steps,
        callbacks=callbacks,
        devices=cfg.experiment.num_gpus_per_node,
        num_nodes=cfg.experiment.num_nodes,
        # Accumulation
        gradient_clip_val=None if cfg.optimizer.zclip else cfg.optimizer.max_grad_norm,
        accumulate_grad_batches=cfg.optimizer.gradient_accumulation_steps,
        precision="bf16-mixed",
        strategy=DDPStrategy(
            process_group_backend="nccl",
            find_unused_parameters=False,
            timeout=timedelta(seconds=cfg.experiment.get("ddp_timeout", 3600)),
        ),
        val_check_interval=val_interval,
        # Logging
        logger=exp_logger,
        fast_dev_run=False,
        limit_val_batches=cfg.experiment.limit_val_batches,
    )

    if chk:
        print(f"******** Loading chkpoint {run_name} {chk}...")
    else:
        print(f"******** Initialized fresh {run_name}...")

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=chk)

    trainer.save_checkpoint(os.path.join(cfg.experiment.checkpoint.path, f"{run_name}_final.pt"))
