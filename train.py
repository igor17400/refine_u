"""
Unified training script with Hydra configuration.

Usage (from inside refine_u/):
    python -m train                                            # default (traditional GCN)
    python -m train experiment=refine_u                        # best UNet config
    python -m train experiment=gsrn_baseline                   # GSRN baseline
    python -m train model.gcn_type=unet training.epochs=100    # inline overrides
"""

import hydra
from omegaconf import DictConfig, OmegaConf

import wandb
from refine_u.data import load_data
from refine_u.pipelines import run_defend, run_gsrn, run_refine_u
from refine_u.utils import build_run_name


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    run_name = build_run_name(cfg)

    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    else:
        wandb.init(mode="disabled")

    lr_matrices, hr_matrices, lr_matrices_test = load_data(
        cfg.data.dir, cfg.data.lr_dim, cfg.data.hr_dim
    )
    print(
        f"Data loaded: {lr_matrices.shape[0]} train, {lr_matrices_test.shape[0]} test"
    )

    if cfg.model.name == "defend":
        run_defend(cfg, lr_matrices, hr_matrices, lr_matrices_test)
    elif cfg.model.name == "gsrn":
        run_gsrn(cfg, lr_matrices, hr_matrices, lr_matrices_test)
    else:
        run_refine_u(cfg, lr_matrices, hr_matrices, lr_matrices_test)

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
