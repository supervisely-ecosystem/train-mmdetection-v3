from mmengine.registry import PARAM_SCHEDULERS
from mmengine.optim.optimizer import OptimWrapper


class DummyOptimWrapper(OptimWrapper):
    @property
    def param_groups(self):
        return self._param_groups

    def __init__(self, start_lr) -> None:
        self._param_groups = [{"lr": start_lr}]

    def get(self):
        return self.param_groups[0]["lr"]


def step_by_epoch(schedulers):
    for scheduler in schedulers:
        if scheduler.by_epoch:
            scheduler.step()


def step_by_iter(schedulers):
    for scheduler in schedulers:
        if not scheduler.by_epoch:
            scheduler.step()


def test_schedulers(schedulers_cfg: list, start_lr, dataloader_len=100, total_epochs=20):
    optim_wrapper = DummyOptimWrapper(start_lr)

    schedulers = []
    for scheduler in schedulers_cfg:
        schedulers.append(
            PARAM_SCHEDULERS.build(
                scheduler, default_args=dict(optimizer=optim_wrapper, epoch_length=dataloader_len)
            )
        )

    lrs = []
    for ep in range(total_epochs):
        for i in range(dataloader_len):
            step_by_iter(schedulers)
            lrs.append(optim_wrapper.get())
        step_by_epoch(schedulers)

    return lrs
