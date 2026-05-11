import csv
import os
from dataclasses import asdict

import torch

from src.config import GPTConfig, TrainerConfig
from src.dataset import CharDataset
from src.utils import ensure_dir


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: CharDataset,
        gpt_config: GPTConfig,
        trainer_config: TrainerConfig,
    ):
        self.model = model
        self.dataset = dataset
        self.gpt_config = gpt_config
        self.config = trainer_config
        self.device = trainer_config.device

        self.model.to(self.device)
        self.optimizer = self.configure_optimizer()

        self.best_val_loss = float("inf")

        ensure_dir(os.path.dirname(self.config.latest_ckpt_path))
        ensure_dir(os.path.dirname(self.config.best_ckpt_path))
        ensure_dir(os.path.dirname(self.config.meta_path))
        ensure_dir(os.path.dirname(self.config.log_path))

        self.init_log_file()

    def configure_optimizer(self):
        decay_params = []
        nodecay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if param.dim() >= 2:
                decay_params.append(param)
            else:
                nodecay_params.append(param)

        optim_groups = [
            {
                "params": decay_params,
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": nodecay_params,
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.config.learning_rate,
            betas=self.config.betas,
        )

        return optimizer

    def init_log_file(self):
        with open(self.config.log_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["iter", "train_loss", "val_loss"])

    def append_log(self, iter_num: int, train_loss: float, val_loss: float):
        with open(self.config.log_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([iter_num, train_loss, val_loss])

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()

        for split in ["train", "val"]:
            losses = torch.zeros(self.config.eval_iters)

            for k in range(self.config.eval_iters):
                x, y = self.dataset.get_batch(
                    split=split,
                    batch_size=self.config.batch_size,
                )
                _, loss, _, _ = self.model(x, y)
                losses[k] = loss.item()

            out[split] = losses.mean().item()

        self.model.train()
        return out

    def save_checkpoint(self, path: str, iter_num: int, val_loss: float):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "iter_num": iter_num,
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "gpt_config": asdict(self.gpt_config),
            "trainer_config": asdict(self.config),
            "meta_path": self.config.meta_path,
        }

        torch.save(checkpoint, path)

    def train(self):
        self.dataset.save_tokenizer(self.config.meta_path)

        print("=" * 60)
        print("Start Training")
        print("=" * 60)
        print(f"Device       : {self.device}")
        print(f"Max iters    : {self.config.max_iters}")
        print(f"Batch size   : {self.config.batch_size}")
        print(f"Block size   : {self.gpt_config.block_size}")
        print(f"Eval interval: {self.config.eval_interval}")
        print(f"Log path     : {self.config.log_path}")
        print(f"Meta path    : {self.config.meta_path}")
        print("=" * 60)

        self.model.train()

        for iter_num in range(1, self.config.max_iters + 1):
            x, y = self.dataset.get_batch(
                split="train",
                batch_size=self.config.batch_size,
            )

            _, loss, _, _ = self.model(x, y)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if self.config.grad_clip is not None and self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )

            self.optimizer.step()

            if iter_num % self.config.log_interval == 0:
                print(f"iter {iter_num:5d} | train loss {loss.item():.4f}")

            if iter_num % self.config.eval_interval == 0:
                losses = self.estimate_loss()
                train_loss = losses["train"]
                val_loss = losses["val"]

                print(
                    f"iter {iter_num:5d} | "
                    f"train loss {train_loss:.4f} | "
                    f"val loss {val_loss:.4f}"
                )

                self.append_log(iter_num, train_loss, val_loss)

                self.save_checkpoint(
                    path=self.config.latest_ckpt_path,
                    iter_num=iter_num,
                    val_loss=val_loss,
                )

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(
                        path=self.config.best_ckpt_path,
                        iter_num=iter_num,
                        val_loss=val_loss,
                    )
                    print(
                        f"Saved best checkpoint to {self.config.best_ckpt_path} "
                        f"with val loss {val_loss:.4f}"
                    )

        print("=" * 60)
        print("Training Finished")
        print("=" * 60)
        print(f"Best val loss: {self.best_val_loss:.4f}")