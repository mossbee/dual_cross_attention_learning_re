"""
Train script for FGVC (e.g., CUB-200-2011).

Purpose:
- Parse args, load config, build datasets/loaders, instantiate DCAL model,
  and launch training via `Trainer` and `FGVCEngine`.
"""


def main() -> None:
    """
    Entry point for FGVC training.
    """
    # Minimal CLI skeleton: parse args and load config
    import argparse
    from dcal.utils.config import load_config
    from dcal.utils.logging import setup_logger
    from dcal.data.loaders import build_fgvc_loaders
    from dcal.engine.trainer import Trainer

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = setup_logger(cfg)
    log.info("Building FGVC loaders")
    _ = build_fgvc_loaders(cfg)
    log.info("Initializing trainer")
    _ = Trainer(cfg)
    log.info("FGVC training pipeline initialized (implementation placeholder)")


if __name__ == "__main__":
    main()


