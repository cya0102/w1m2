import collections
import logging
import os
import shutil
import time

import numpy as np
import torch

from models.loss import (
    cal_nll_loss,
    event_disentanglement_loss,
    ivc_loss,
    mixture_pull_push_loss,
    rec_loss,
)
from utils import TimeMeter, AverageMeter
from runners.stage_a5 import (
    CANDIDATE_NAMES,
    REASON_CODES,
    select_stage_a5_candidates,
)

import pickle
import copy
from pathlib import Path
import pdb


STAGE_A_EPSILON_SCAN = (0.00, 0.01, 0.02, 0.05)

def info(msg):
    print(msg)
    logging.info(msg)


class MainRunner:
    def __init__(self, args):
        self.args = args
        run_timestamp = self.args.get(
            'run_timestamp', time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
        run_name = "{}_{}".format(self.args.get('tag', 'base'), run_timestamp)
        self.model_saved_path = os.path.join(
            self.args['train']['model_saved_path'], run_name)
        event_config = self.args['model']['config'].get(
            'event_disentanglement', {})
        self.inference_event_weight = event_config.get(
            'inference_event_weight', 0.1)
        self.inference_vote_event_weight = event_config.get(
            'inference_vote_event_weight', 0.5)
        self.selection_strategy = self.args.get(
            'selection_strategy',
            'semantic_vote' if self.args.get('vote', False) else 'nll')
        if self.selection_strategy not in {
                'nll', 'geometric_vote', 'semantic_vote'}:
            raise ValueError(
                'unknown proposal selection strategy: {}'.format(
                    self.selection_strategy))
        self.selection_temperature = float(
            self.args.get('selection_temperature', 0.1))
        if self.selection_temperature <= 0:
            raise ValueError('selection temperature must be positive')
        info('Proposal selection strategy: {} (temperature={}).'.format(
            self.selection_strategy, self.selection_temperature))
        stage_a_config = self.args['model']['config'].get(
            'event_boundary_refinement', {})
        self.stage_a_enabled = bool(stage_a_config.get('enabled', False))
        self.stage_a5_config = dict(stage_a_config.get('stage_a5', {}))
        self.stage_a5_enabled = bool(self.stage_a5_config.get('enabled', False))
        if self.stage_a5_enabled and not self.stage_a_enabled:
            raise ValueError('Stage A.5 requires Stage A to be enabled')
        self.stage_a5_mask_seeds = tuple(int(seed) for seed in self.stage_a5_config.get(
            'eval_mask_seeds', [8, 18, 28]))
        if self.stage_a5_enabled and not self.stage_a5_mask_seeds:
            raise ValueError('Stage A.5 requires at least one eval mask seed')
        if len(set(self.stage_a5_mask_seeds)) != len(self.stage_a5_mask_seeds):
            raise ValueError('Stage A.5 eval mask seeds must be unique')
        self.stage_a_report_only = bool(stage_a_config.get('report_only', True))
        self.stage_a_max_nll_increase = float(
            stage_a_config.get('max_nll_increase', 0.02))
        if (self.stage_a_max_nll_increase < 0 or
                not np.isfinite(self.stage_a_max_nll_increase)):
            raise ValueError('Stage-A max_nll_increase must be finite and non-negative')
        epsilon_scan = stage_a_config.get(
            'epsilon_scan', STAGE_A_EPSILON_SCAN)
        self.stage_a_epsilon_scan = tuple(sorted(set(
            [float(epsilon) for epsilon in epsilon_scan] +
            [self.stage_a_max_nll_increase])))
        if not self.stage_a_epsilon_scan:
            self.stage_a_epsilon_scan = STAGE_A_EPSILON_SCAN
        if any(epsilon < 0 for epsilon in self.stage_a_epsilon_scan):
            raise ValueError('Stage-A epsilon values must be non-negative')
        self._build_dataset()

        self.args['model']['config']['vocab_size'] = self.train_set.vocab_size
        self.args['model']['config']['max_epoch'] = self.args['train']['max_num_epochs']

        self._build_model()
        if 'train' in args:
            self._build_optimizer()
            self.num_updates = 0

    def train(self):
        select_on_val = bool(self.args.get('select_on_val', True))
        if select_on_val and self.val_loader is None:
            raise ValueError(
                '--select-on-val requires dataset.val_data in the config')
        selection_loader = self.val_loader if select_on_val else self.test_loader
        selection_split = 'Validation' if select_on_val else 'Test'
        info('Checkpoint selection split: {}'.format(selection_split))

        best_by_objective = {
            name: {'score': float('-inf'), 'results': None, 'epoch': None}
            for name in ('r1', 'r5', 'composite')
        }
        best_event_results = None
        best_event_epoch = None
        for epoch in range(1, self.args['train']['max_num_epochs']+1):
            info('Start Epoch {}'.format(epoch))
            os.makedirs(self.model_saved_path, mode=0o755, exist_ok=True)
            save_path = os.path.join(self.model_saved_path, 'model-{}.pt'.format(epoch))

            self._train_one_epoch(epoch)
            self._save_model(save_path)
            # Keep the current V3 test-selection path exactly: its historical
            # no-diversity run evaluated negative proposals with epoch=0.
            evaluation_epoch = epoch if select_on_val else 0
            results = self.eval(
                epoch=evaluation_epoch, loader=selection_loader,
                split=selection_split)
            selection_scores = calculate_checkpoint_selection_scores(results)
            for objective, score in selection_scores.items():
                best = best_by_objective[objective]
                if score > best['score']:
                    best.update(
                        score=score, results=results, epoch=epoch)
                    objective_path = os.path.join(
                        self.model_saved_path,
                        'model-best-{}.pt'.format(objective))
                    shutil.copyfile(save_path, objective_path)
                    # Preserve the historical filename as an alias of the
                    # Rank-1-selected checkpoint.
                    if objective == 'r1':
                        shutil.copyfile(
                            save_path,
                            os.path.join(
                                self.model_saved_path, 'model-best.pt'))
                    info('Best {} {} score updated to {:.4f} at epoch {}.'.format(
                        selection_split, objective, score, epoch))
            event_active = (
                not self.model.use_event_disentanglement
                or self.model.event_disentangler.subspace_updates.item() > 0)
            if event_active and (
                    best_event_results is None
                    or results['R@1,mIoU'].avg
                    > best_event_results['R@1,mIoU'].avg):
                best_event_results = results
                best_event_epoch = epoch
                shutil.copyfile(
                    save_path,
                    os.path.join(self.model_saved_path, 'model-best-event.pt'))
                info('Best active-Event {} results have been updated at epoch {}.'.format(
                    selection_split, epoch))
            info('=' * 60)
        
        for objective, best in best_by_objective.items():
            msg = '|'.join([
                ' {} {:.4f} '.format(k, v.avg)
                for k, v in best['results'].items()])
            info('Best {} {} results (epoch {}, score {:.4f}):'.format(
                selection_split, objective, best['epoch'], best['score']))
            info('|'+msg+'|')
        if best_event_results is not None:
            msg = '|'.join([
                ' {} {:.4f} '.format(k, v.avg)
                for k, v in best_event_results.items()])
            info('Best active-Event {} results (epoch {}):'.format(
                selection_split, best_event_epoch))
            info('|'+msg+'|')

        # In the strict protocol, the test set is untouched during model
        # selection. Report all three validation-selected objectives once.
        if select_on_val:
            for objective, best in best_by_objective.items():
                best_checkpoint = os.path.join(
                    self.model_saved_path,
                    'model-best-{}.pt'.format(objective))
                self._load_model_parameters(best_checkpoint)
                info('Final test evaluation of validation-selected {} '
                     'checkpoint (epoch {}):'.format(
                         objective, best['epoch']))
                self.eval(
                    epoch=best['epoch'], loader=self.test_loader,
                    split='Test(best-{}-validation)'.format(objective))

    def _train_one_epoch(self, epoch, **kwargs):
        self.model.train()

        def print_log():
            msg = 'Epoch {}, Batch {}, lr = {:.5f}, '.format(epoch, bid, curr_lr)
            for k, v in loss_meter.items():
                msg += '{} = {:.4f}, '.format(k, v.avg)
                v.reset()
            msg += '{:.3f} seconds/batch'.format(1.0 / time_meter.avg)
            info(msg)

        display_n_batches, bid = 50, 0
        time_meter = TimeMeter()
        loss_meter = collections.defaultdict(lambda: AverageMeter())

        for bid, batch in enumerate(self.train_loader, 1):
            self.optimizer.zero_grad()
            net_input = move_to_cuda(batch['net_input'])
            output = self.model(epoch=epoch, **net_input)

            loss, loss_dict = rec_loss(**output, num_props=self.model.num_props, **self.args['loss'])
            rnk_loss, rnk_loss_dict = ivc_loss(**output, num_props=self.model.num_props, **self.args['loss'])
            loss_dict.update(rnk_loss_dict)
            loss = loss + rnk_loss
            event_loss, event_loss_dict = event_disentanglement_loss(
                **output, num_props=self.model.num_props, **self.args['loss'])
            loss_dict.update(event_loss_dict)
            loss = loss + event_loss
            mixture_loss, mixture_loss_dict = mixture_pull_push_loss(
                **output, num_props=self.model.num_props,
                **self.args['loss'])
            loss_dict.update(mixture_loss_dict)
            loss = loss + mixture_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            self.optimizer.step()

            self.num_updates += 1
            curr_lr = self.lr_scheduler.step_update(self.num_updates)
            time_meter.update()
            for k, v in loss_dict.items():
                loss_meter[k].update(v)

            if bid % display_n_batches == 0:
                print_log()

        if bid % display_n_batches != 0:
            print_log()

    def eval(self, save=None, epoch=0, loader=None, split='Test'):
        loader = self.test_loader if loader is None else loader
        self.model.eval()
        metrics_logger = collections.defaultdict(lambda: AverageMeter())
        diagnostics_logger = collections.defaultdict(
            lambda: AverageMeter())
        stage_a_metric_loggers = {
            epsilon: collections.defaultdict(lambda: AverageMeter())
            for epsilon in self.stage_a_epsilon_scan
        } if self.stage_a_enabled else {}
        stage_a_diagnostics_logger = collections.defaultdict(
            lambda: AverageMeter())
        stage_a5_metric_logger = collections.defaultdict(
            lambda: AverageMeter()) if self.stage_a5_enabled else None
        stage_a5_diagnostics_logger = collections.defaultdict(
            lambda: AverageMeter()) if self.stage_a5_enabled else None
        with torch.no_grad():
            for bid, batch in enumerate(loader, 1):
                durations = np.asarray([i[1] for i in batch['raw']])
                gt = np.asarray([i[2] for i in batch['raw']])

                mask_seeds = (self.stage_a5_mask_seeds
                              if self.stage_a5_enabled else (None,))
                outputs = []
                parent_nlls = []
                for mask_seed in mask_seeds:
                    net_input = move_to_cuda(batch['net_input'])
                    if self.stage_a_enabled:
                        # The explicit flag prevents Stage A from ever running
                        # in training forward passes, even when the config is
                        # shared.
                        net_input['run_stage_a'] = True
                    if self.stage_a5_enabled:
                        net_input['run_stage_a5'] = True
                        sample_ids = []
                        for sample_index, raw in enumerate(batch['raw']):
                            if len(raw) < 5:
                                raise ValueError(
                                    'Stage A.5 requires raw sample ids at raw[4]')
                            sample_ids.append(str(raw[4]))
                        from models.cpl import deterministic_eval_word_mask
                        mask = deterministic_eval_word_mask(
                            sample_ids,
                            batch['net_input']['words_len'],
                            batch['net_input']['words_feat'].size(1) - 1,
                            mask_seed,
                            weights=batch['net_input']['weights'])
                        net_input['eval_word_mask'] = mask.to(
                            next(self.model.parameters()).device)
                    current_output = self.model(epoch=epoch, **net_input)
                    current_words_mask = current_output['words_mask'].unsqueeze(1) \
                        .expand(len(durations), self.model.num_props, -1).contiguous().view(
                            len(durations) * self.model.num_props, -1)
                    current_words_id = current_output['words_id'].unsqueeze(1) \
                        .expand(len(durations), self.model.num_props, -1).contiguous().view(
                            len(durations) * self.model.num_props, -1)
                    current_nll, _ = cal_nll_loss(
                        current_output['words_logit'], current_words_id,
                        current_words_mask)
                    parent_nlls.append(
                        current_nll.view(len(durations), self.model.num_props))
                    # Candidate scores and geometry are already materialized by
                    # this point.  Do not keep any vocabulary-sized logits for
                    # the other mask seeds in memory.
                    for key in ('words_logit', 'neg_words_logit_1',
                                'neg_words_logit_2', 'ref_words_logit'):
                        if key in current_output:
                            current_output[key] = None
                    outputs.append(current_output)
                output = outputs[0]
                bsz = len(durations)
                num_props = self.model.num_props
                k = min(num_props, 5)

                words_mask = output['words_mask'].unsqueeze(1) \
                    .expand(bsz, num_props, -1).contiguous().view(
                        bsz * num_props, -1)
                words_id = output['words_id'].unsqueeze(1) \
                    .expand(bsz, num_props, -1).contiguous().view(
                        bsz * num_props, -1)

                parent_nll_stack = torch.stack(parent_nlls, dim=0)
                parent_nll = parent_nll_stack.mean(dim=0)
                parent_nll_std = parent_nll_stack.std(
                    dim=0, unbiased=False) if len(outputs) > 1 else torch.zeros_like(parent_nll)
                event_score = output.get('event_score')
                if event_score is not None:
                    event_scores = torch.stack([
                        current_output['event_score'].view(bsz, num_props)
                        for current_output in outputs], dim=0)
                    event_score = event_scores.mean(dim=0)
                    parent_score = parent_nll - (
                        self.inference_event_weight * event_score)
                else:
                    parent_score = parent_nll
                idx = parent_score.argsort(dim=-1)
                sorted_proposal_score = parent_score.gather(
                    index=idx, dim=-1).cpu().numpy()

                raw_width = output['width'].view(bsz, num_props)
                raw_center = output['center'].view(bsz, num_props)
                raw_props = torch.stack([
                    torch.clamp(raw_center - raw_width / 2, min=0),
                    torch.clamp(raw_center + raw_width / 2, max=1),
                ], dim=-1).cpu().numpy()
                width = raw_width.gather(index=idx, dim=-1)
                center = raw_center.gather(index=idx, dim=-1)
                selected_props = torch.stack([
                    torch.clamp(center - width / 2, min=0),
                    torch.clamp(center + width / 2, max=1)], dim=-1)
                selected_props = selected_props.cpu().numpy()
                gt = gt / durations[:, np.newaxis]

                context_width = output.get('mixture_context_width')
                if context_width is not None:
                    context_width = context_width.view(
                        bsz, num_props).cpu().numpy()
                diagnostics = calculate_candidate_diagnostics(
                    raw_props=raw_props,
                    nll_sorted_props=selected_props,
                    gt=gt,
                    context_width=context_width)
                for name, (value, count) in diagnostics.items():
                    diagnostics_logger[name].update(value, count)

                _update_metric_logger(
                    metrics_logger, selected_props, sorted_proposal_score,
                    gt, self.selection_strategy, self.selection_temperature,
                    self.args['dataset']['dataset'], k)

                if self.stage_a_enabled:
                    required = (
                        'stage_a_candidate_start',
                        'stage_a_candidate_end',
                        'stage_a_candidate_valid',
                        'stage_a_candidate_nll')
                    missing = [name for name in required if name not in output]
                    if missing:
                        raise RuntimeError(
                            'Stage A model output is missing: {}'.format(
                                ', '.join(missing)))
                    candidate_start = output[
                        'stage_a_candidate_start'].detach()
                    candidate_end = output[
                        'stage_a_candidate_end'].detach()
                    candidate_valid = output[
                        'stage_a_candidate_valid'].detach()
                    candidate_nll = output[
                        'stage_a_candidate_nll'].detach()
                    boundaries = {
                        key: net_input[key].detach().cpu().numpy()
                        for key in ('event_boundary_positions',
                                    'event_boundary_mask')}
                    event_score_np = (
                        event_score.detach() if event_score is not None
                        else None)
                    for epsilon in self.stage_a_epsilon_scan:
                        refined_props_t, refined_nll_t, selected_candidate = (
                            select_minimal_sufficient_candidates(
                                candidate_start, candidate_end, candidate_nll,
                                candidate_valid, epsilon))
                        refined_score = refined_nll_t
                        if event_score is not None:
                            refined_score = refined_score - (
                                self.inference_event_weight * event_score_np)
                        stage_idx = refined_score.argsort(dim=-1)
                        stage_score = refined_score.gather(
                            index=stage_idx, dim=-1).cpu().numpy()
                        stage_props = refined_props_t.gather(
                            1, stage_idx.unsqueeze(-1).expand(-1, -1, 2))
                        stage_props = stage_props.cpu().numpy()
                        stage_logger = stage_a_metric_loggers[epsilon]
                        _update_metric_logger(
                            stage_logger, stage_props, stage_score, gt,
                            self.selection_strategy,
                            self.selection_temperature,
                            self.args['dataset']['dataset'], k)
                        if epsilon == self.stage_a_max_nll_increase:
                            stage_diagnostics = calculate_stage_a_diagnostics(
                                raw_props=raw_props,
                                refined_props=refined_props_t.cpu().numpy(),
                                candidate_nll=candidate_nll.cpu().numpy(),
                                candidate_valid=candidate_valid.cpu().numpy(),
                                selected_candidate=selected_candidate.cpu().numpy(),
                                boundary_positions=boundaries[
                                    'event_boundary_positions'],
                                boundary_mask=boundaries[
                                    'event_boundary_mask'],
                                gt=gt,
                                stage_sorted_props=stage_props)
                            for name, (value, count) in stage_diagnostics.items():
                                stage_a_diagnostics_logger[name].update(
                                    value, count)

                if self.stage_a5_enabled:
                    required_a5 = (
                        'stage_a_candidate_start',
                        'stage_a_candidate_end',
                        'stage_a_candidate_valid',
                        'stage_a_candidate_nll',
                        'stage_a_candidate_shell_nll',
                        'stage_a_candidate_boundary_confidence',
                    )
                    for current_output in outputs:
                        missing_a5 = [
                            name for name in required_a5
                            if name not in current_output]
                        if missing_a5:
                            raise RuntimeError(
                                'Stage A.5 model output is missing: {}'.format(
                                    ', '.join(missing_a5)))
                    candidate_nll_stack = torch.stack([
                        current_output['stage_a_candidate_nll'].detach()
                        for current_output in outputs], dim=0)
                    shell_nll_stack = torch.stack([
                        current_output['stage_a_candidate_shell_nll'].detach()
                        for current_output in outputs], dim=0)
                    candidate_start = output[
                        'stage_a_candidate_start'].detach().cpu().numpy()
                    candidate_end = output[
                        'stage_a_candidate_end'].detach().cpu().numpy()
                    candidate_valid = output[
                        'stage_a_candidate_valid'].detach().cpu().numpy()
                    boundary_confidence = output[
                        'stage_a_candidate_boundary_confidence'].detach().cpu().numpy()
                    candidate_nll_np = candidate_nll_stack.cpu().numpy()
                    shell_nll_np = shell_nll_stack.cpu().numpy()
                    selector_config = self.stage_a5_config
                    refined_props, selector_scores, selected_candidate, reasons = (
                        select_stage_a5_candidates(
                            candidate_start, candidate_end, candidate_valid,
                            candidate_nll_np.mean(axis=0),
                            candidate_nll_np.std(axis=0),
                            shell_nll_np.mean(axis=0),
                            shell_nll_np.std(axis=0),
                            boundary_confidence, selector_config,
                            contrast_mean=(shell_nll_np - candidate_nll_np).mean(axis=0),
                            contrast_std=(shell_nll_np - candidate_nll_np).std(axis=0),
                        ))
                    selected_nll = np.take_along_axis(
                        candidate_nll_np.mean(axis=0),
                        selected_candidate[..., None], axis=-1)[..., 0]
                    stage5_score = selected_nll - self.inference_event_weight * (
                        event_score.detach().cpu().numpy()
                        if event_score is not None else 0.0)
                    stage5_idx = np.argsort(stage5_score, axis=-1)
                    stage5_sorted_props = np.take_along_axis(
                        refined_props,
                        stage5_idx[..., None].repeat(2, axis=-1), axis=1)
                    _update_metric_logger(
                        stage_a5_metric_logger,
                        stage5_sorted_props,
                        np.take_along_axis(stage5_score, stage5_idx, axis=1),
                        gt, self.selection_strategy,
                        self.selection_temperature,
                        self.args['dataset']['dataset'], k)
                    a5_diagnostics = calculate_stage_a5_diagnostics(
                        raw_props=raw_props,
                        refined_props=refined_props,
                        selected_candidate=selected_candidate,
                        reasons=reasons,
                        candidate_nll=candidate_nll_np.mean(axis=0),
                        shell_nll=shell_nll_np.mean(axis=0),
                        gt=gt)
                    for name, (value, count) in a5_diagnostics.items():
                        stage_a5_diagnostics_logger[name].update(value, count)

        msg = '|'.join([' {} {:.4f} '.format(k, v.avg)
                        for k, v in metrics_logger.items()])
        info('Baseline({}): |{}|'.format(split, msg))
        diagnostic_msg = '|'.join([
            ' {} {:.4f} '.format(k, v.avg)
            for k, v in diagnostics_logger.items()
            if v.count > 0])
        info('Diagnostics({}): |{}|'.format(split, diagnostic_msg))
        for epsilon, logger in stage_a_metric_loggers.items():
            stage_msg = '|'.join([
                ' {} {:.4f} '.format(k, v.avg)
                for k, v in logger.items()])
            info('StageA({}, epsilon={:.2f}): |{}|'.format(
                split, epsilon, stage_msg))
        if self.stage_a_enabled:
            stage_diagnostic_msg = '|'.join([
                ' {} {:.4f} '.format(k, v.avg)
                for k, v in stage_a_diagnostics_logger.items()
                if v.count > 0])
            info('StageA diagnostics({}, epsilon={:.2f}): |{}|'.format(
                split, self.stage_a_max_nll_increase, stage_diagnostic_msg))
        if self.stage_a5_enabled:
            stage5_msg = '|'.join([
                ' {} {:.4f} '.format(k, v.avg)
                for k, v in stage_a5_metric_logger.items()])
            info('StageA5({}, selector={}): |{}|'.format(
                split, self.stage_a5_config.get('selector',
                                                'counterfactual_gated'),
                stage5_msg))
            stage5_diagnostic_msg = '|'.join([
                ' {} {:.4f} '.format(k, v.avg)
                for k, v in stage_a5_diagnostics_logger.items()
                if v.count > 0])
            info('StageA5 diagnostics({}): |{}|'.format(
                split, stage5_diagnostic_msg))
        # The unprefixed result remains the baseline result, so enabling
        # report-only Stage A cannot affect checkpoint selection.
        return metrics_logger


    def _build_dataset(self):
        import datasets as da
        import pickle
        from torch.utils.data import DataLoader
        args = self.args['dataset']
        cls = getattr(da, args['dataset'], None)
        with open(args['vocab_path'], 'rb') as fp:
            vocab = pickle.load(fp)
        self.train_set = cls(data_path=args['train_data'], vocab=vocab, args=args, is_training=True, split='train')
        self.test_set = cls(data_path=args['test_data'], vocab=vocab, args=args, split='test')
        self.val_set = cls(data_path=args['val_data'], vocab=vocab, args=args, split='val') if args['val_data'] else None
        info('train: {} samples, test: {} samples'.format(len(self.train_set), len(self.test_set)))
        batch_size = self.args['train']['batch_size']
        train_num_workers = self.args['train'].get('num_workers', 2)
        test_num_workers = self.args['train'].get('test_num_workers', 0)
        val_num_workers = self.args['train'].get('val_num_workers', 1)

        def worker_init_fn(worker_id):
            def set_seed(seed):
                import random
                import numpy as np
                import torch

                random.seed(seed)
                np.random.seed(seed + 1)
                torch.manual_seed(seed + 3)
                torch.cuda.manual_seed(seed + 4)
                torch.cuda.manual_seed_all(seed + 4)

            set_seed(8 + worker_id)

        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True,
                                       collate_fn=self.train_set.collate_data, num_workers=train_num_workers,
                                       worker_init_fn=worker_init_fn)
        self.test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=False,
                                      collate_fn=self.test_set.collate_data,
                                      num_workers=test_num_workers)
        self.val_loader = DataLoader(self.val_set, batch_size=batch_size, shuffle=False,
                                     collate_fn=self.val_set.collate_data,
                                     num_workers=val_num_workers) if args['val_data'] else None

    def _build_model(self):
        model_config = self.args['model']
        import models

        self.model = getattr(models, model_config['name'], None)(model_config['config'])
        self.model = self.model.cuda()
        print(self.model)
        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('Total:', total_num, 'Trainable:', trainable_num)

    def _build_optimizer(self):
        from optimizers import AdamOptimizer
        from optimizers.lr_schedulers import InverseSquareRootSchedule
        
        parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        args = self.args['train']["optimizer"]
        self.optimizer = AdamOptimizer(args, parameters)
        self.lr_scheduler = InverseSquareRootSchedule(args, self.optimizer)

    def _save_model(self, path):
        state_dict = {
            'num_updates': self.num_updates,
            'config': self.args,
            'model_parameters': self.model.state_dict(),
        }
        torch.save(state_dict, path)
        info('save model to {}, num_updates {}.'.format(path, self.num_updates))

    def _load_model(self, path):
        state_dict = torch.load(path)
        self.num_updates = state_dict['num_updates']
        self.lr_scheduler.step_update(self.num_updates)
        parameters = state_dict['model_parameters']
        self.model.load_state_dict(parameters)
        info('load model from {}, num_updates {}.'.format(path, self.num_updates))

    def _load_pretrained_model(self, path):
        """Warm-start V4 from compatible V3 weights only.

        V3's single-Gaussian proposal head has no V4 counterpart, while V4's
        mixture generator must be learned from scratch.  The running LRRV
        subspace is also intentionally reset because its statistics were
        estimated from V3 single-Gaussian positives.  Unlike ``_load_model``,
        this method never restores ``num_updates`` or the LR schedule.
        """
        state_dict = torch.load(path, map_location='cpu')
        source = state_dict.get('model_parameters', state_dict)
        target = self.model.state_dict()
        compatible = {}
        skipped = []

        for name, value in source.items():
            if name.startswith('event_disentangler.'):
                skipped.append('{} (reset Event subspace)'.format(name))
                continue
            if name not in target:
                skipped.append('{} (not used by V4)'.format(name))
                continue
            if target[name].shape != value.shape:
                skipped.append('{} (shape {} -> {})'.format(
                    name, tuple(value.shape), tuple(target[name].shape)))
                continue
            compatible[name] = value

        if not compatible:
            raise ValueError(
                'no compatible parameters found in {}'.format(path))

        result = self.model.load_state_dict(compatible, strict=False)
        # A warm start must begin a fresh V4 optimization schedule.
        self.num_updates = 0
        self.lr_scheduler.step_update(self.num_updates)

        source_config = state_dict.get('config', {})
        info('initialize V4 from V3 checkpoint {} (source tag: {}).'.format(
            path, source_config.get('tag', 'unknown')))
        info('loaded {} compatible tensors; num_updates reset to 0.'.format(
            len(compatible)))
        info('skipped source tensors: {}'.format(skipped))
        info('randomly initialized/reset V4 tensors: {}'.format(
            result.missing_keys))

    def _load_model_parameters(self, path):
        """Load weights only, for final evaluation after model selection."""
        state_dict = torch.load(path, map_location='cpu')
        self.model.load_state_dict(state_dict['model_parameters'])
        info('load model parameters from {}.'.format(path))


def calculate_IoU_batch(i0, i1):
    union = (np.min(np.stack([i0[0], i1[0]], 0), 0), np.max(np.stack([i0[1], i1[1]], 0), 0))
    inter = (np.max(np.stack([i0[0], i1[0]], 0), 0), np.min(np.stack([i0[1], i1[1]], 0), 0))
    iou = 1.0 * (inter[1] - inter[0] + 1e-10) / (union[1] - union[0] + 1e-10)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou


def calculate_checkpoint_selection_scores(results):
    """Return the three validation objectives used by stage0."""
    rank1 = results['R@1,mIoU'].avg
    rank5 = results['R@5,mIoU'].avg
    return {
        'r1': rank1,
        'r5': rank5,
        'composite': 0.5 * (rank1 + rank5),
    }


def calculate_candidate_diagnostics(raw_props, nll_sorted_props, gt,
                                    context_width=None):
    """Return weighted candidate-level diagnostics for one evaluation batch.

    Values are returned as ``name: (batch_mean, sample_count)`` so callers can
    aggregate batches without bias from the final short batch.
    """
    batch_size, num_props, _ = raw_props.shape
    if nll_sorted_props.shape != raw_props.shape:
        raise ValueError('sorted and raw proposal shapes must match')
    if gt.shape != (batch_size, 2):
        raise ValueError('ground truth shape must be [batch, 2]')

    diagnostics = {}
    raw_width = raw_props[:, :, 1] - raw_props[:, :, 0]
    for proposal_index in range(num_props):
        proposal_iou = calculate_IoU_batch(
            (raw_props[:, proposal_index, 0],
             raw_props[:, proposal_index, 1]),
            (gt[:, 0], gt[:, 1]))
        prefix = 'proposal{}_'.format(proposal_index + 1)
        diagnostics[prefix + 'mean_width'] = (
            raw_width[:, proposal_index].mean(), batch_size)
        diagnostics[prefix + 'mIoU'] = (
            proposal_iou.mean(), batch_size)
        diagnostics[prefix + 'IoU@0.3'] = (
            (proposal_iou >= 0.3).mean(), batch_size)
        diagnostics[prefix + 'IoU@0.5'] = (
            (proposal_iou >= 0.5).mean(), batch_size)

    pairwise_iou = []
    for first in range(num_props):
        for second in range(first + 1, num_props):
            pairwise_iou.append(calculate_IoU_batch(
                (raw_props[:, first, 0], raw_props[:, first, 1]),
                (raw_props[:, second, 0], raw_props[:, second, 1])))
    if pairwise_iou:
        pairwise_iou = np.stack(pairwise_iou, axis=1)
        pair_count = pairwise_iou.size
        diagnostics['pairwise_iou_mean'] = (
            pairwise_iou.mean(), pair_count)
        for threshold in (0.5, 0.7, 0.9):
            diagnostics['pairwise_iou_gt_{:.1f}'.format(threshold)] = (
                (pairwise_iou > threshold).mean(), pair_count)

    max_top = min(num_props, 5)
    for top_k in range(1, max_top + 1):
        result = top_n_metric(
            nll_sorted_props[:, :top_k].transpose(1, 0, 2), gt)
        diagnostics['nll_top{}_mIoU'.format(top_k)] = (
            result['mIoU'], batch_size)
        diagnostics['nll_top{}_IoU@0.3'.format(top_k)] = (
            result['IoU@0.3'], batch_size)
        diagnostics['nll_top{}_IoU@0.5'.format(top_k)] = (
            result['IoU@0.5'], batch_size)

    gt_width = gt[:, 1] - gt[:, 0]
    width_groups = {
        'short': (0.0, 0.15),
        'medium_short': (0.15, 0.35),
        'medium_long': (0.35, 0.60),
        'long': (0.60, float('inf')),
    }
    for group, (lower, upper) in width_groups.items():
        group_mask = (gt_width >= lower) & (gt_width < upper)
        group_count = int(group_mask.sum())
        if group_count == 0:
            continue
        group_result = top_n_metric(
            nll_sorted_props[group_mask, :max_top].transpose(1, 0, 2),
            gt[group_mask])
        diagnostics['gt_{}_R5_mIoU'.format(group)] = (
            group_result['mIoU'], group_count)
        diagnostics['gt_{}_R5_IoU@0.3'.format(group)] = (
            group_result['IoU@0.3'], group_count)
        diagnostics['gt_{}_R5_IoU@0.5'.format(group)] = (
            group_result['IoU@0.5'], group_count)

    if context_width is not None:
        if context_width.shape != raw_width.shape:
            raise ValueError('context widths must match raw proposal widths')
        boundary_gap = context_width - raw_width
        diagnostics['negative_envelope_mean_width'] = (
            context_width.mean(), context_width.size)
        diagnostics['negative_localization_width_gap'] = (
            boundary_gap.mean(), boundary_gap.size)

    return diagnostics


def select_minimal_sufficient_candidates(
        candidate_start, candidate_end, candidate_nll, candidate_valid,
        max_nll_increase, width_tolerance=1e-6):
    """Select the shortest NLL-sufficient candidate for every parent.

    The comparison is lexicographic: width first, then NLL, then candidate
    index.  Candidate zero is always retained as the fallback, including when
    all trim candidates are invalid.
    """
    if not (torch.is_tensor(candidate_start) and
            torch.is_tensor(candidate_end) and
            torch.is_tensor(candidate_nll) and
            torch.is_tensor(candidate_valid)):
        raise TypeError('Stage-A selector expects torch tensors')
    if candidate_start.shape != candidate_end.shape or \
            candidate_start.ndim != 3 or candidate_start.size(-1) != 7:
        raise ValueError('candidate geometry must have shape [B, N, 7]')
    if candidate_nll.shape != candidate_start.shape or \
            candidate_valid.shape != candidate_start.shape:
        raise ValueError('candidate NLL/validity shapes must match geometry')
    if max_nll_increase < 0 or not np.isfinite(max_nll_increase):
        raise ValueError('max_nll_increase must be finite and non-negative')
    if width_tolerance < 0 or not np.isfinite(width_tolerance):
        raise ValueError('width_tolerance must be finite and non-negative')

    widths = candidate_end - candidate_start
    base_nll = candidate_nll[..., :1]
    eligible = (candidate_valid.bool() & torch.isfinite(candidate_nll) &
                (candidate_nll <= base_nll + max_nll_increase))
    # The original is the explicit safety net.  Its NLL remains the selected
    # NLL even if malformed upstream validity marks it false.
    eligible[..., 0] = True
    best_index = torch.zeros(
        candidate_start.shape[:2], dtype=torch.long,
        device=candidate_start.device)
    best_width = widths[..., 0]
    best_nll = candidate_nll[..., 0]
    for candidate_index in range(1, 7):
        current_width = widths[..., candidate_index]
        current_nll = candidate_nll[..., candidate_index]
        same_width = torch.abs(current_width - best_width) <= width_tolerance
        better = eligible[..., candidate_index] & (
            (current_width < best_width - width_tolerance) |
            (same_width & (current_nll < best_nll)))
        best_index = torch.where(
            better,
            torch.full_like(best_index, candidate_index), best_index)
        best_width = torch.where(better, current_width, best_width)
        best_nll = torch.where(better, current_nll, best_nll)

    gather_index = best_index.unsqueeze(-1)
    refined_start = candidate_start.gather(-1, gather_index).squeeze(-1)
    refined_end = candidate_end.gather(-1, gather_index).squeeze(-1)
    refined_nll = candidate_nll.gather(-1, gather_index).squeeze(-1)
    refined_props = torch.stack([refined_start, refined_end], dim=-1)
    return refined_props, refined_nll, best_index


def _update_metric_logger(logger, sorted_props, sorted_scores, gt,
                          selection_strategy, selection_temperature,
                          dataset_name, top_k):
    selected_index = select_proposal_by_strategy(
        sorted_props, sorted_scores, strategy=selection_strategy,
        temperature=selection_temperature,
        charades_anchor=(dataset_name == 'CharadesSTA'))
    result = top_1_metric(
        sorted_props[np.arange(len(gt)), selected_index], gt)
    for key, value in result.items():
        logger['R@1,' + key].update(value, len(gt))
    result = top_n_metric(sorted_props[:, :top_k].transpose(1, 0, 2), gt)
    for key, value in result.items():
        logger['R@%d,' % top_k + key].update(value, len(gt))


def calculate_stage_a_diagnostics(
        raw_props, refined_props, candidate_nll, candidate_valid,
        selected_candidate, boundary_positions, boundary_mask, gt,
        stage_sorted_props=None):
    """Calculate the required width, NLL, boundary and duration diagnostics."""
    raw_props = np.asarray(raw_props)
    refined_props = np.asarray(refined_props)
    candidate_nll = np.asarray(candidate_nll)
    candidate_valid = np.asarray(candidate_valid).astype(bool)
    selected_candidate = np.asarray(selected_candidate)
    boundary_positions = np.asarray(boundary_positions)
    boundary_mask = np.asarray(boundary_mask).astype(bool)
    gt = np.asarray(gt)
    if raw_props.ndim != 3 or raw_props.shape[-1] != 2 or \
            refined_props.shape != raw_props.shape:
        raise ValueError('raw/refined proposals must both have shape [B, N, 2]')
    batch_size, num_props, _ = raw_props.shape
    if candidate_nll.shape != (batch_size, num_props, 7) or \
            candidate_valid.shape != candidate_nll.shape or \
            selected_candidate.shape != (batch_size, num_props):
        raise ValueError('Stage-A candidate arrays have inconsistent shapes')
    if gt.shape != (batch_size, 2):
        raise ValueError('ground truth must have shape [B, 2]')
    if boundary_positions.shape != boundary_mask.shape or \
            boundary_positions.shape[0] != batch_size:
        raise ValueError('boundary arrays have inconsistent shapes')

    before_width = raw_props[..., 1] - raw_props[..., 0]
    after_width = refined_props[..., 1] - refined_props[..., 0]
    selected_nll = np.take_along_axis(
        candidate_nll, selected_candidate[..., None], axis=-1)[..., 0]
    original_nll = candidate_nll[..., 0]
    diagnostics = {
        'stage_a_changed_fraction': (
            np.mean(selected_candidate != 0), batch_size * num_props),
        'stage_a_mean_width_before': (
            np.mean(before_width), batch_size * num_props),
        'stage_a_mean_width_after': (
            np.mean(after_width), batch_size * num_props),
        'stage_a_mean_width_reduction': (
            np.mean(before_width - after_width), batch_size * num_props),
        'stage_a_mean_nll_delta': (
            np.nanmean(selected_nll - original_nll), batch_size * num_props),
        'stage_a_valid_candidates_per_proposal': (
            np.mean(candidate_valid.sum(axis=-1)), batch_size * num_props),
        'stage_a_original_fraction': (
            np.mean(selected_candidate == 0), batch_size * num_props),
        'stage_a_left_trim_fraction': (
            np.mean(np.isin(selected_candidate, (1, 2))),
            batch_size * num_props),
        'stage_a_right_trim_fraction': (
            np.mean(np.isin(selected_candidate, (3, 4))),
            batch_size * num_props),
        'stage_a_both_trim_fraction': (
            np.mean(np.isin(selected_candidate, (5, 6))),
            batch_size * num_props),
        'stage_a_near_whole_video_before': (
            np.mean(before_width >= 0.9), batch_size * num_props),
        'stage_a_near_whole_video_after': (
            np.mean(after_width >= 0.9), batch_size * num_props),
    }

    before_distances = []
    after_distances = []
    for batch_index in range(batch_size):
        positions = boundary_positions[batch_index][boundary_mask[batch_index]]
        if positions.size == 0:
            continue
        before_distances.extend(np.min(
            np.abs(raw_props[batch_index, :, :, None] - positions[None, None, :]),
            axis=-1).reshape(-1).tolist())
        after_distances.extend(np.min(
            np.abs(refined_props[batch_index, :, :, None] - positions[None, None, :]),
            axis=-1).reshape(-1).tolist())
    if before_distances:
        diagnostics['stage_a_endpoint_boundary_distance_before'] = (
            np.mean(before_distances), len(before_distances))
        diagnostics['stage_a_endpoint_boundary_distance_after'] = (
            np.mean(after_distances), len(after_distances))

    if stage_sorted_props is None:
        stage_sorted_props = refined_props
    stage_sorted_props = np.asarray(stage_sorted_props)
    if stage_sorted_props.shape != raw_props.shape:
        raise ValueError('stage_sorted_props must have shape [B, N, 2]')
    max_top = min(num_props, 5)
    gt_width = gt[:, 1] - gt[:, 0]
    width_groups = {
        'short': (0.0, 0.15),
        'medium_short': (0.15, 0.35),
        'medium_long': (0.35, 0.60),
        'long': (0.60, float('inf')),
    }
    for group, (lower, upper) in width_groups.items():
        group_mask = (gt_width >= lower) & (gt_width < upper)
        group_count = int(group_mask.sum())
        if group_count == 0:
            continue
        result = top_n_metric(
            stage_sorted_props[group_mask, :max_top].transpose(1, 0, 2),
            gt[group_mask])
        diagnostics['stage_a_gt_{}_R5_mIoU'.format(group)] = (
            result['mIoU'], group_count)
        diagnostics['stage_a_gt_{}_R5_IoU@0.3'.format(group)] = (
            result['IoU@0.3'], group_count)
        diagnostics['stage_a_gt_{}_R5_IoU@0.5'.format(group)] = (
            result['IoU@0.5'], group_count)
    return diagnostics


def calculate_stage_a5_diagnostics(
        raw_props, refined_props, selected_candidate, reasons,
        candidate_nll, shell_nll, gt):
    """Return paired online diagnostics for one Stage-A.5 batch.

    GT is accepted only by this reporting function.  The selector itself is
    kept in ``runners.stage_a5`` and has no GT-shaped argument.
    """
    raw_props = np.asarray(raw_props)
    refined_props = np.asarray(refined_props)
    selected_candidate = np.asarray(selected_candidate)
    reasons = np.asarray(reasons)
    candidate_nll = np.asarray(candidate_nll)
    shell_nll = np.asarray(shell_nll)
    gt = np.asarray(gt)
    if raw_props.shape != refined_props.shape or raw_props.ndim != 3 or \
            raw_props.shape[-1] != 2:
        raise ValueError('raw/refined props must have shape [B, N, 2]')
    batch_size, num_props, _ = raw_props.shape
    expected = (batch_size, num_props)
    if selected_candidate.shape != expected or reasons.shape != expected:
        raise ValueError('Stage-A.5 selection arrays have invalid shape')
    if candidate_nll.shape != (batch_size, num_props, 7) or \
            shell_nll.shape != candidate_nll.shape:
        raise ValueError('Stage-A.5 score arrays have invalid shape')
    if gt.shape != (batch_size, 2):
        raise ValueError('ground truth must have shape [B, 2]')

    original_iou = calculate_IoU_batch(
        (raw_props[..., 0], raw_props[..., 1]),
        (np.broadcast_to(gt[:, 0, None], (batch_size, num_props)),
         np.broadcast_to(gt[:, 1, None], (batch_size, num_props))))
    refined_iou = calculate_IoU_batch(
        (refined_props[..., 0], refined_props[..., 1]),
        (np.broadcast_to(gt[:, 0, None], (batch_size, num_props)),
         np.broadcast_to(gt[:, 1, None], (batch_size, num_props))))
    delta_iou = refined_iou - original_iou
    changed = selected_candidate != 0
    helpful = changed & (delta_iou > 0.01)
    harmful = changed & (delta_iou < -0.01)
    neutral = changed & ~(helpful | harmful)
    changed_count = max(int(changed.sum()), 1)
    diagnostics = {
        'stage_a5_changed_fraction': (changed.mean(), batch_size * num_props),
        'stage_a5_mean_width_before': (
            (raw_props[..., 1] - raw_props[..., 0]).mean(), batch_size * num_props),
        'stage_a5_mean_width_after': (
            (refined_props[..., 1] - refined_props[..., 0]).mean(),
            batch_size * num_props),
        'stage_a5_helpful_fraction': (
            helpful.sum() / changed_count, batch_size * num_props),
        'stage_a5_harmful_fraction': (
            harmful.sum() / changed_count, batch_size * num_props),
        'stage_a5_neutral_fraction': (
            neutral.sum() / changed_count, batch_size * num_props),
        'stage_a5_trim_precision': (
            helpful.sum() / max(int((helpful | harmful).sum()), 1),
            batch_size * num_props),
        'stage_a5_mean_delta_iou': (
            delta_iou[changed].mean() if changed.any() else 0.0,
            max(int(changed.sum()), 1)),
        'stage_a5_mean_contrast': (
            np.nanmean((shell_nll - candidate_nll)[..., 1:]),
            batch_size * num_props * 6),
    }
    reason_names = {value: key for key, value in REASON_CODES.items()}
    for reason_code in sorted(reason_names):
        diagnostics['stage_a5_reason_{}'.format(reason_names[reason_code])] = (
            np.mean(reasons == reason_code), batch_size * num_props)
    return diagnostics


def select_proposal_by_strategy(selected_props, proposal_scores,
                                strategy='nll', temperature=0.1,
                                charades_anchor=False):
    """Select Rank-1 from NLL-sorted proposals.

    ``geometric_vote`` reproduces CPL's original unweighted medoid vote.
    ``semantic_vote`` lets proposals vote in proportion to reconstruction
    confidence, preventing a cluster of weak background intervals from
    outvoting one semantically strong event interval.
    """
    batch_size, num_proposals, _ = selected_props.shape
    if proposal_scores.shape != (batch_size, num_proposals):
        raise ValueError('proposal scores must match selected proposals')
    if strategy == 'nll':
        return np.zeros(batch_size, dtype=np.int64)
    if strategy not in {'geometric_vote', 'semantic_vote'}:
        raise ValueError(
            'unknown proposal selection strategy: {}'.format(strategy))
    if temperature <= 0:
        raise ValueError('selection temperature must be positive')

    pairwise_iou = np.zeros(
        (batch_size, num_proposals, num_proposals), dtype=np.float64)
    for i in range(num_proposals):
        for j in range(num_proposals):
            pairwise_iou[:, i, j] = calculate_IoU_batch(
                (selected_props[:, i, 0], selected_props[:, i, 1]),
                (selected_props[:, j, 0], selected_props[:, j, 1]))

    if strategy == 'semantic_vote':
        logits = -(
            proposal_scores - proposal_scores.min(axis=1, keepdims=True))
        logits = logits / temperature
        logits = logits - logits.max(axis=1, keepdims=True)
        support = np.exp(logits)
        support = support / support.sum(axis=1, keepdims=True)
    elif charades_anchor:
        support = pairwise_iou[:, 0, :]
    else:
        support = np.ones(
            (batch_size, num_proposals), dtype=np.float64)

    votes = (pairwise_iou * support[:, np.newaxis, :]).sum(axis=-1)
    return np.argmax(votes, axis=1)


# [nb, 2], [nb, 2]
def top_n_metric(preds, label):
    result = {}
    bsz = preds[0].shape[0]
    top_iou = []
    for pred in preds:
        iou = calculate_IoU_batch((pred[:, 0], pred[:, 1]), (label[:, 0], label[:, 1]))
        top_iou.append(iou)
    iou = np.max(np.stack(top_iou, 1), 1)
    result['mIoU'] = np.mean(iou)
    for i in range(1, 10, 2):
        result['IoU@0.{}'.format(i)] = 1.0 * np.sum(iou >= i / 10) / bsz
    return result


def top_1_discount_metric(pred, label):
    result = {}
    bsz = pred.shape[0]
    iou = calculate_IoU_batch((pred[:, 0], pred[:, 1]), (label[:, 0], label[:, 1]))
    discount = (1-np.abs(pred[:, 0]-label[:,0])) * (1-np.abs(pred[:, 1]-label[:, 1]))
    result['mIoU'] = np.mean(iou)
    for i in range(1, 10, 2):
        result['IoU@0.{}'.format(i)] = 1.0 * np.sum((iou >= i / 10).astype(float) * discount) / bsz
    return result

def top_1_metric(pred, label):
    result = {}
    bsz = pred.shape[0]
    iou = calculate_IoU_batch((pred[:, 0], pred[:, 1]), (label[:, 0], label[:, 1]))
    result['mIoU'] = np.mean(iou)
    for i in range(1, 10, 2):
        result['IoU@0.{}'.format(i)] = 1.0 * np.sum(iou >= i / 10) / bsz
    return result


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {
                key: _apply(value)
                for key, value in x.items()
            }
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample):
    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)
