from mmengine.hooks import Hook


class CheckpointEvalHook(Hook):
    """Run validation right after checkpoint-saving iterations."""

    priority = 'LOWEST'

    def __init__(self, interval=-1, by_epoch=False, save_begin=0, save_last=True):
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_begin = save_begin
        self.save_last = save_last
        self._last_eval_iter = -1

    def _should_eval_after_iter(self, runner):
        if self.by_epoch:
            return False
        if self.interval <= 0:
            return False
        max_iters = getattr(runner, 'max_iters', None)
        if max_iters is None and getattr(runner, 'train_loop', None) is not None:
            max_iters = getattr(runner.train_loop, '_max_iters', None)
        is_last_iter = max_iters is not None and runner.iter + 1 == max_iters
        return (
            self.every_n_train_iters(runner, self.interval, self.save_begin)
            or (self.save_last and is_last_iter)
        )

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if not self._should_eval_after_iter(runner):
            return
        current_iter = runner.iter + 1
        if current_iter == self._last_eval_iter:
            return
        self._last_eval_iter = current_iter

        if runner.val_loop is None:
            runner.logger.warning(
                'Skip checkpoint validation because val_loop is not configured.')
            return

        runner.logger.info(
            f'Running validation after checkpoint at {current_iter} iterations')
        was_training = runner.model.training
        runner.val_loop.run()
        if was_training:
            runner.model.train()
