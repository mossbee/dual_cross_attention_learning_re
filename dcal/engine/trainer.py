"""
Generic training harness.

Purpose:
- Handle training epochs, optimization steps, checkpointing, logging, and
  dispatch to task-specific engines for loss computation and metrics.

Design:
- `Trainer` consumes a config, model, optimizer, schedulers, dataloaders, and
  a task engine (`fgvc_engine` or `reid_engine`).

I/O:
- Trainer.train() and Trainer.validate() methods.
"""

from typing import Any, Dict, Optional
import time
import torch

from dcal.utils.logging import setup_logger
from dcal.utils.checkpoint import save_checkpoint
from dcal.engine.evaluator import top1_accuracy


class Trainer:
    """
    Skeleton training harness.
    """

    def __init__(
        self,
        cfg: Any,
        engine: Any,
        train_loader: Any,
        val_loader: Optional[Any] = None,
        query_loader: Optional[Any] = None,
        gallery_loader: Optional[Any] = None,
    ) -> None:
        self.cfg = cfg
        self.engine = engine
        self.model = engine.model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.query_loader = query_loader
        self.gallery_loader = gallery_loader

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        tr = cfg["train"]
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt_name = tr.get("optimizer", "adamw").lower()
        if opt_name == "sgd":
            self.optimizer = torch.optim.SGD(
                params, lr=tr.get("lr", 0.01), momentum=0.9, weight_decay=tr.get("weight_decay", 1e-4)
            )
        else:
            self.optimizer = torch.optim.AdamW(params, lr=tr.get("lr", 5e-4), weight_decay=tr.get("weight_decay", 0.05))

        if tr.get("cosine_decay", True):
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=tr.get("epochs", 100))
        else:
            self.scheduler = None

        self.logger = setup_logger(cfg)

    def train(self) -> Dict[str, Any]:
        epochs = self.cfg["train"].get("epochs", 1)
        log_interval = 10
        best_acc = 0.0
        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            t0 = time.time()
            for i, batch in enumerate(self.train_loader, start=1):
                # Move tensors to device
                if "images" in batch:
                    batch["images"] = batch["images"].to(self.device, non_blocking=True)
                if "labels" in batch:
                    batch["labels"] = batch["labels"].to(self.device, non_blocking=True)
                if "pids" in batch:
                    batch["pids"] = batch["pids"].to(self.device, non_blocking=True)
                if "camids" in batch:
                    batch["camids"] = batch["camids"].to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)
                out = self.engine.step(batch)
                loss = out["loss"]
                loss.backward()
                self.optimizer.step()
                epoch_loss += float(loss.detach().cpu())

                if i % log_interval == 0:
                    msg = f"Epoch {epoch} Iter {i} Loss {epoch_loss / i:.4f}"
                    if "acc1" in out:
                        msg += f" Acc@1 {float(out['acc1']):.4f}"
                    self.logger.info(msg)

            if self.scheduler is not None:
                self.scheduler.step()

            # Validation (FGVC only by default)
            if self.val_loader is not None and self.cfg.get("task") == "fgvc":
                acc1 = self.validate()
                if acc1 > best_acc:
                    best_acc = acc1
                    save_checkpoint(
                        {
                            "model": self.model.state_dict(),
                            "optimizer": self.optimizer.state_dict(),
                            "epoch": epoch,
                            "best_acc": best_acc,
                        },
                        path=f"{self.cfg.get('logging', {}).get('output_dir', './outputs')}/best.pt",
                    )

            # Save last checkpoint
            save_checkpoint(
                {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "best_acc": best_acc,
                },
                path=f"{self.cfg.get('logging', {}).get('output_dir', './outputs')}/last.pt",
            )

            dt = time.time() - t0
            self.logger.info(f"Epoch {epoch} done in {dt:.1f}s, avg loss {epoch_loss / max(1, len(self.train_loader)):.4f}")

        return {"best_acc": best_acc}

    def validate(self) -> float:
        self.model.eval()
        acc_sum = 0.0
        n = 0
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["images"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)
                out = self.model({"images": images})
                acc = top1_accuracy(out["sa"]["logits"], labels)
                acc_sum += float(acc)
                n += 1
        acc1 = acc_sum / max(1, n)
        self.logger.info(f"Validation Acc@1 {acc1:.4f}")
        return acc1


