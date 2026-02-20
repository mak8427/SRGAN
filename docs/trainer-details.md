# Trainer details

This page describes the training control flow used by OpenSR-SRGAN on PyTorch Lightning 2+.

## Bootstrap sequence

1. **Validate Lightning version.** `SRGAN_model.setup_lightning()` enforces Lightning `>= 2.0`.
2. **Enable manual optimization.** The model sets `automatic_optimization = False`.
3. **Bind training-step helper.** `training_step_PL2` is attached as the active training-step implementation.
4. **Build trainer kwargs.** `build_lightning_kwargs()` normalises accelerator/devices and prepares `fit_kwargs` (including `ckpt_path` when resuming).

## Training-step anatomy

`training_step_PL2(self, batch, batch_idx)` always performs manual optimizer control.

```python
opt_d, opt_g = self.optimizers()
pretrain_phase = self._pretrain_check()
self.log("training/pretrain_phase", float(pretrain_phase), sync_dist=True)
```

### Pretraining branch

When `pretrain_phase` is active:

1. Discriminator metrics are logged as zeros (no discriminator optimizer step).
2. Generator content loss is computed and logged.
3. The generator optimizer performs `zero_grad -> manual_backward -> step`.
4. EMA updates after the generator step when enabled and active.

### Adversarial branch

When pretraining is finished:

1. **Discriminator update**
1. Compute `D(hr)` and `D(sr.detach())`.
1. Apply BCE or Wasserstein objective (+ optional R1 penalty).
1. Log discriminator losses/probabilities.
1. Run `manual_backward` and `opt_d.step()`.
2. **Generator update**
1. Compute content loss + metrics.
1. Compute adversarial generator objective from `D(sr)`.
1. Apply ramped adversarial weight (`training/adv_loss_weight`).
1. Log `generator/content_loss`, `generator/adversarial_loss`, `generator/total_loss`.
1. Run `manual_backward` and `opt_g.step()`.
3. EMA updates after the generator step when enabled and active.

## Resume behavior

`Model.continue_training` is passed through `build_lightning_kwargs()` and forwarded as:

```python
trainer.fit(model, datamodule=pl_datamodule, ckpt_path=resume_ckpt)
```

This restores optimizer/scheduler state, EMA state, and global step before continuing.

## Runtime checks summary

| Check | Source | Purpose |
| --- | --- | --- |
| Lightning `>= 2.0` | `SRGAN_model.setup_lightning()` | Reject unsupported runtime versions. |
| Manual optimization enabled | `setup_lightning()` | Ensure GAN optimizer alternation is explicit. |
| Pretraining active? | `_pretrain_check()` | Gate between content-only and adversarial training. |
| Adversarial weight | `_adv_loss_weight()` | Log and apply the current GAN loss multiplier. |
| EMA active? | `self.global_step >= self._ema_update_after_step` | Delay EMA updates until configured step. |

## Workflow map

See `opensr_srgan/model/training_workflow.txt` for the full text branch map aligned to the current implementation.
