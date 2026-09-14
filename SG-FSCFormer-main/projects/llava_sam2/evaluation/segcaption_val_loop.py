from mmengine.runner import ValLoop
from mmengine.runner.amp import autocast
from mmengine.registry import LOOPS


@LOOPS.register_module()
class SegCaptionValLoop(ValLoop):
    """Validation loop for SegCaption metrics.
    """

    def run_iter(self, idx, data_batch):
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)

        model = self.runner.model
        if hasattr(model, 'module') and hasattr(model.module, 'val_step'):
            model = model.module

        with autocast(enabled=self.fp16):
            outputs = model.val_step(data_batch)

        if isinstance(outputs, dict):
            outputs = [outputs]

        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
