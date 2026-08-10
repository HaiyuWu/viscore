import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from hydra.utils import instantiate
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from masking import MaskSampleTransform
from module import ARPredictor, Embedder, MLP
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.weight
    use_recon = bool(cfg.loss.cls_recon.enabled)

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch, return_masked=use_recon)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()

    # With CLS-recon on, the regularizer sees both clean and masked embeddings.
    # `reg_masked` (default true) disables the concat to isolate the effect.
    if use_recon and cfg.loss.cls_recon.get("reg_masked", True):
        reg_input = torch.cat([emb, output["masked_emb"]], dim=0)
    else:
        reg_input = emb
    output["reg_loss"] = self.regularizer(reg_input.transpose(0, 1))
    total = output["pred_loss"] + lambd * output["reg_loss"]

    if use_recon:
        if cfg.loss.cls_recon.get("patch_token", False):
            # Patch-token recon: MSE between clean and masked patch projections,
            # restricted to MASKED positions.
            clean_patches = output["patch_emb"].detach()       # (B, T, N, D)
            masked_patches = output["masked_patch_emb"]        # (B, T, N, D)
            mask_flat = batch["mask"].view(batch["mask"].size(0), batch["mask"].size(1), -1).bool()  # (B, T, N)
            diff = (clean_patches - masked_patches).pow(2).mean(dim=-1)  # (B, T, N)
            output["patch_recon_loss"] = diff[mask_flat].mean()
            total = total + cfg.loss.cls_recon.weight * output["patch_recon_loss"]
        else:
            # CLS recon: masked-CLS projection should reconstruct clean-CLS.
            output["cls_recon_loss"] = (output["emb"].detach() - output["masked_emb"]).pow(2).mean()
            total = total + cfg.loss.cls_recon.weight * output["cls_recon_loss"]

    output["loss"] = total

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    # Pinned to stable-worldmodel 0.0.6, whose data backend is HDF5. Fail loudly rather than
    # silently ignore an unsupported backend -- see docs/SWM_MIGRATION.md.
    fmt = cfg.data.get("format", "hdf5")
    if fmt != "hdf5":
        raise NotImplementedError(
            f"data.format={fmt!r} is not supported on stable-worldmodel 0.0.6 "
            "(HDF5 only). See docs/SWM_MIGRATION.md."
        )
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    if cfg.loss.cls_recon.enabled:
        transforms.append(MaskSampleTransform(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            min_ratio=cfg.loss.cls_recon.min_ratio,
            max_ratio=cfg.loss.cls_recon.max_ratio,
        ))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=cfg.loss.cls_recon.enabled,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    # Asymmetric patch-token projectors for the patch-token recon mode.
    patch_proj_clean = None
    patch_proj_masked = None
    if cfg.loss.cls_recon.enabled and cfg.loss.cls_recon.get("patch_token", False):
        patch_proj_clean = torch.nn.LayerNorm(hidden_dim, elementwise_affine=False)
        patch_proj_masked = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 2048), torch.nn.BatchNorm1d(2048), torch.nn.GELU(),
            torch.nn.Linear(2048, 2048),       torch.nn.BatchNorm1d(2048), torch.nn.GELU(),
            torch.nn.Linear(2048, hidden_dim), torch.nn.BatchNorm1d(hidden_dim),
        )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
        patch_proj_clean=patch_proj_clean,
        patch_proj_masked=patch_proj_masked,
    )

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        regularizer = instantiate(cfg.loss.regularizer),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        # save_dir anchors Lightning's default ModelCheckpoint under STABLEWM_HOME
        # (s3), i.e. $DATA_HOME/<project>/<run>/checkpoints/, instead of polluting
        # the cwd (project dir) with a ./lewm-visreg/ tree. The checkpoints there are
        # redundant with the Manager's lewm_weights.ckpt (same full-state ckpt);
        # this just keeps every artifact on s3 alongside the run dir.
        logger = WandbLogger(save_dir=str(swm.data.utils.get_cache_dir()), **cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()
    return


if __name__ == "__main__":
    run()
