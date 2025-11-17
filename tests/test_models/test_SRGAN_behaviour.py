import types

import torch

from opensr_srgan.model import SRGAN
from opensr_srgan.model import training_step_PL


class LoggerMixin:
    def __init__(self):
        self.logged = {}

    def log(self, key, value, **kwargs):
        self.logged[key] = value

    def _log_generator_content_loss(self, loss):
        self.log("generator/content_loss", loss)

    def _log_adv_loss_weight(self, weight):
        self.log("training/adv_loss_weight", weight)


class DummyContentLoss:
    def return_loss(self, sr, hr):
        loss = torch.nn.functional.l1_loss(sr, hr)
        return loss, {"l1": loss.detach()}


class DummyConfig:
    class Schedulers:
        gradient_clip_val = 0.0


class TrainingHarness(LoggerMixin):
    def __init__(self, pretrain=False):
        super().__init__()
        self.pretrain_mode = pretrain
        self.content_loss_criterion = DummyContentLoss()
        self.adversarial_loss_criterion = torch.nn.BCEWithLogitsLoss()
        self.adv_target = 0.9
        self.generator = torch.nn.Conv2d(1, 1, kernel_size=1)
        self.discriminator = torch.nn.Conv2d(1, 1, kernel_size=1)
        self._ema_update_after_step = 0
        self.ema = None
        self.global_step = 0
        self.config = DummyConfig()
        self.trainer = types.SimpleNamespace(precision_plugin=None)

    def forward(self, lr):
        return self.generator(lr)

    def _pretrain_check(self):
        return self.pretrain_mode

    def _compute_adv_loss_weight(self):
        return torch.tensor(0.5)

    def _adv_loss_weight(self):
        return self._compute_adv_loss_weight()

    def manual_backward(self, loss):
        loss.backward()

    def optimizers(self):
        opt_d = torch.optim.SGD(self.discriminator.parameters(), lr=0.1)
        opt_g = torch.optim.SGD(self.generator.parameters(), lr=0.1)
        return opt_d, opt_g

    def toggle_optimizer(self, *args, **kwargs):
        return None

    def untoggle_optimizer(self, *args, **kwargs):
        return None


def _sample_batch():
    lr = torch.ones(1, 1, 2, 2, requires_grad=True)
    hr = torch.ones(1, 1, 2, 2)
    return lr, hr


def test_setup_lightning_selects_training_step_branches():
    model = SRGAN.SRGAN_model.__new__(SRGAN.SRGAN_model)
    model.pl_version = (2, 0, 0)
    model.automatic_optimization = True
    model.setup_lightning()
    assert model.automatic_optimization is False
    assert model._training_step_implementation.__func__ is training_step_PL.training_step_PL2

    model = SRGAN.SRGAN_model.__new__(SRGAN.SRGAN_model)
    model.pl_version = (1, 9, 0)
    model.automatic_optimization = True
    model.setup_lightning()
    assert model.automatic_optimization is True
    assert model._training_step_implementation.__func__ is training_step_PL.training_step_PL1


def test_training_step_pl1_handles_pretraining_branch():
    harness = TrainingHarness(pretrain=True)
    loss = training_step_PL.training_step_PL1(harness, _sample_batch(), batch_idx=0, optimizer_idx=1)
    assert torch.is_tensor(loss)
    assert harness.logged["training/pretrain_phase"] == 1.0
    assert "generator/content_loss" in harness.logged


def test_training_step_pl2_runs_manual_optimization():
    harness = TrainingHarness(pretrain=False)
    harness.pl_version = (2, 0, 0)
    harness.automatic_optimization = False

    loss = training_step_PL.training_step_PL2(harness, _sample_batch(), batch_idx=0)

    assert torch.is_tensor(loss)
    assert "discriminator/adversarial_loss" in harness.logged
    assert "generator/total_loss" in harness.logged