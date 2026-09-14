import collections
import csv
import json
import logging
import os
import random
import shutil
import time

import numpy as np
import torch

from models.loss import (
    cal_nll_loss,
    event_disentanglement_loss,
    ivc_loss,
    mixture_pull_push_loss,
    qcec_coherence_loss,
    rec_loss,
)
from utils import (
    AverageMeter,
    TimeMeter,
    isolated_rng,
    sha256_file,
    source_tree_digest,
)

import pickle
import copy
from pathlib import Path
import pdb

def info(msg):
    print(msg)
    logging.info(msg)


class MainRunner:
    def __init__(self, args):
        self.args = args
        self._resolve_project_relative_paths()
        self.eval_seed = int(self.args.get(
            'eval_seed', int(self.args.get('seed', 8)) + 1000033))
        self.initialization_source = 'random'
        self.initialization_kind = 'random'
        self.initialization_valid = False
        self._set_initialization_metadata(
            source='random', kind='random', valid=False, write=False)
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
        self._build_dataset()

        self.args['model']['config']['vocab_size'] = self.train_set.vocab_size
        self.args['model']['config']['max_epoch'] = self.args['train']['max_num_epochs']

        self._build_model()
        if 'train' in args:
            self._build_optimizer()
            self.num_updates = 0
        self.start_epoch = 0
        self.current_epoch = 0
        self._backbone_update_steps = 0
        self._epoch_backbone_update_steps = 0
        # Model construction and DataLoader shuffling must not make the
        # public initialization protocol depend on whether QCEC is enabled.
        self._reset_runtime_rng(int(self.args.get('seed', 8)))
        self._write_run_metadata()

    def _resolve_project_relative_paths(self):
        """Resolve bundled data/checkpoint paths independently of cwd."""
        project_root = Path(__file__).resolve().parents[1]
        dataset = self.args.get('dataset', {})
        for key in (
                'train_data', 'test_data', 'val_data', 'vocab_path',
                'qcec_cluster_index_path'):
            value = dataset.get(key)
            if not value or os.path.isabs(value):
                continue
            project_path = project_root / value
            if project_path.exists() or key == 'qcec_cluster_index_path':
                dataset[key] = str(project_path)
        train_config = self.args.get('train', {})
        model_path = train_config.get('model_saved_path')
        if model_path and not os.path.isabs(model_path):
            train_config['model_saved_path'] = str(project_root / model_path)

    def _set_initialization_metadata(self, source, kind, valid, write=True):
        """Record the initialization provenance used by this run."""
        if not hasattr(self, 'args'):
            self.args = {}
        self.initialization_source = str(source)
        self.initialization_kind = str(kind)
        self.initialization_valid = bool(valid)
        self.args['initialization'] = {
            'source': self.initialization_source,
            'kind': self.initialization_kind,
            'valid_for_qcec_freeze': self.initialization_valid,
        }
        if write and hasattr(self, 'model_saved_path'):
            self._write_run_metadata()

    def _reset_runtime_rng(self, seed):
        """Reset training/data RNGs after model construction or warm-start."""
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
        torch.manual_seed(seed + 2)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed + 4)
            torch.cuda.manual_seed_all(seed + 4)

    def _parameter_summary(self):
        parameters = list(self.model.named_parameters())
        qcec_ids = set(getattr(self.model, '_qcec_parameter_ids', set()))

        def count(items):
            return int(sum(parameter.numel() for _, parameter in items))

        trainable = [(name, parameter) for name, parameter in parameters
                     if parameter.requires_grad]
        qcec = [(name, parameter) for name, parameter in parameters
                if id(parameter) in qcec_ids]
        optimizer_parameters = []
        if hasattr(self, 'optimizer'):
            seen = set()
            for group in self.optimizer.optimizer.param_groups:
                for parameter in group['params']:
                    if id(parameter) not in seen:
                        optimizer_parameters.append(parameter)
                        seen.add(id(parameter))
        return {
            'total_numel': count(parameters),
            'initial_trainable_numel': count(trainable),
            'qcec_numel': count(qcec),
            'baseline_numel': int(sum(
                parameter.numel() for name, parameter in parameters
                if id(parameter) not in qcec_ids)),
            'optimizer_managed_numel': int(sum(
                parameter.numel() for parameter in optimizer_parameters)),
            'optimizer_managed_tensors': len(optimizer_parameters),
            'initial_trainable_tensors': len(trainable),
        }

    def _cluster_index_summary(self):
        path = getattr(self, 'args', {}).get(
            'dataset', {}).get('qcec_cluster_index_path')
        if not path or not os.path.exists(path):
            return None
        return {
            'path': os.path.abspath(path),
            'sha256': sha256_file(path),
            'size_bytes': os.path.getsize(path),
        }

    def _write_run_metadata(self):
        """Persist resolved config and immutable run provenance."""
        if not hasattr(self, 'model_saved_path'):
            return
        args = getattr(self, 'args', {})
        eval_seed = int(getattr(
            self, 'eval_seed', int(args.get('seed', 8)) + 1000033))
        os.makedirs(self.model_saved_path, mode=0o755, exist_ok=True)
        project_root = Path(__file__).resolve().parents[1]
        metadata = {
            'config_path': args.get('config_path'),
            'seed': args.get('seed'),
            'eval_seed': eval_seed,
            'source_sha256': source_tree_digest(project_root),
            'initialization': args.get('initialization', {}),
            'cluster_index': self._cluster_index_summary(),
            'parameters': self._parameter_summary(),
            'selection_strategy': getattr(self, 'selection_strategy', 'nll'),
            'selection_temperature': getattr(
                self, 'selection_temperature', 0.1),
            'select_on_val': bool(args.get('select_on_val', True)),
            'final_test_enabled': bool(args.get('final_test', False)),
        }
        self.run_metadata = metadata
        self.args['run_metadata'] = metadata
        with open(os.path.join(self.model_saved_path, 'run_metadata.json'), 'w',
                  encoding='utf8') as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        with open(os.path.join(self.model_saved_path, 'resolved_config.json'), 'w',
                  encoding='utf8') as handle:
            json.dump(self.args, handle, indent=2, sort_keys=True, default=str)
        info('Run metadata: {}'.format(json.dumps(
            metadata, sort_keys=True, default=str)))

    @staticmethod
    def _meter_values(meters):
        return {
            name: float(meter.avg)
            for name, meter in meters.items()
            if getattr(meter, 'count', 1) > 0
        }

    def _record_metrics(self, payload):
        """Append structured metrics in JSONL and long-form CSV formats."""
        if not hasattr(self, 'model_saved_path'):
            return
        os.makedirs(self.model_saved_path, mode=0o755, exist_ok=True)
        jsonl_path = os.path.join(self.model_saved_path, 'metrics.jsonl')
        with open(jsonl_path, 'a', encoding='utf8') as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str) + '\n')

        csv_path = os.path.join(self.model_saved_path, 'metrics.csv')
        exists = os.path.exists(csv_path)
        with open(csv_path, 'a', newline='', encoding='utf8') as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=['kind', 'epoch', 'split', 'section', 'metric',
                            'value'])
            if not exists:
                writer.writeheader()
            for section in ('metrics', 'raw_metrics', 'diagnostics',
                            'selection_diagnostics', 'stage'):
                values = payload.get(section, {})
                for metric, value in values.items():
                    writer.writerow({
                        'kind': payload.get('kind', ''),
                        'epoch': payload.get('epoch', ''),
                        'split': payload.get('split', ''),
                        'section': section,
                        'metric': metric,
                        'value': value,
                    })

    def _validate_training_protocol(self):
        """Reject accidental QCEC-only freezing from random scratch starts."""
        if not getattr(self.model, 'use_qcec', False):
            return
        freeze_epochs = int(getattr(
            self.model, 'qcec_freeze_backbone_epochs', 0))
        if freeze_epochs <= 0:
            return
        if self.initialization_valid:
            return
        if self.initialization_kind == 'resume' and self.start_epoch >= freeze_epochs:
            info(
                'Resume starts after QCEC freeze stage (epoch {} >= {}); '
                'continuing with the restored training history.'.format(
                    self.start_epoch, freeze_epochs))
            return
        raise ValueError(
            'QCEC freeze_backbone_epochs={} requires a validated pretrained '
            'initialization or a resume past the freeze stage; random scratch '
            'training must set model.config.qcec.freeze_backbone_epochs=0.'
            .format(freeze_epochs))

    def train(self):
        self._validate_training_protocol()
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
        start_epoch = int(getattr(self, 'start_epoch', 0))
        for epoch in range(
                start_epoch + 1,
                self.args['train']['max_num_epochs'] + 1):
            info('Start Epoch {}'.format(epoch))
            os.makedirs(self.model_saved_path, mode=0o755, exist_ok=True)
            save_path = os.path.join(self.model_saved_path, 'model-{}.pt'.format(epoch))

            train_metrics = self._train_one_epoch(epoch)
            if train_metrics is None:
                train_metrics = {}
            self.current_epoch = epoch
            self._save_model(save_path)
            self._record_metrics({
                'kind': 'train',
                'epoch': epoch,
                'split': 'Train',
                'stage': train_metrics.pop('_stage', {}),
                'metrics': train_metrics,
            })
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

        selection_summary = {
            'selection_split': selection_split,
            'primary_rule': 'Validation R@1,mIoU' if select_on_val
            else 'Test R@1,mIoU',
            'composite_formula': '0.5 * (R@1,mIoU + R@5,mIoU)',
            'objectives': {},
        }
        for objective, best in best_by_objective.items():
            selection_summary['objectives'][objective] = {
                'epoch': best['epoch'],
                'score': best['score'],
                'metrics': self._meter_values(best['results'])
                if best['results'] is not None else {},
            }
        if best_event_results is not None:
            selection_summary['event'] = {
                'epoch': best_event_epoch,
                'metrics': self._meter_values(best_event_results),
            }
        with open(os.path.join(
                self.model_saved_path, 'selection_summary.json'), 'w',
                encoding='utf8') as handle:
            json.dump(selection_summary, handle, indent=2, sort_keys=True)

        # In the strict protocol, the test set is untouched during model
        # selection.  Development runs leave this final test pass disabled;
        # it is enabled only after the method, seed and checkpoint rule are
        # locked.
        if select_on_val and bool(self.args.get('final_test', False)):
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

        if select_on_val and not bool(self.args.get('final_test', False)):
            info('Development mode: final Test evaluation is disabled.')

    def _train_one_epoch(self, epoch, **kwargs):
        self.model.train()
        self._epoch_backbone_update_steps = 0
        if getattr(self.model, 'use_qcec', False):
            freeze_epochs = getattr(
                self.model, 'qcec_freeze_backbone_epochs', 0)
            qcec_only = epoch <= freeze_epochs
            self.model.set_qcec_training_stage(qcec_only)
        else:
            qcec_only = False

        trainable_numel = int(sum(
            parameter.numel() for parameter in self.model.parameters()
            if parameter.requires_grad))
        optimizer_numel = int(sum(
            parameter.numel()
            for group in self.optimizer.optimizer.param_groups
            for parameter in group['params']))
        event_schedule = (
            float(self.model.event_loss_schedule(epoch))
            if getattr(self.model, 'use_event_disentanglement', False)
            else 0.0)
        current_lr = float(self.optimizer.get_lr())
        self._log_training_stage(
            epoch, qcec_only=qcec_only, trainable_numel=trainable_numel,
            optimizer_numel=optimizer_numel, current_lr=current_lr,
            event_schedule=event_schedule)

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
        epoch_loss_meter = collections.defaultdict(lambda: AverageMeter())

        for bid, batch in enumerate(self.train_loader, 1):
            self.optimizer.zero_grad()
            net_input = move_to_cuda(batch['net_input'])
            output = self.model(epoch=epoch, **net_input)

            reconstruction_loss, loss_dict = rec_loss(
                **output, num_props=self.model.num_props, **self.args['loss'])
            ranking_loss, rnk_loss_dict = ivc_loss(
                **output, num_props=self.model.num_props, **self.args['loss'])
            loss_dict.update(rnk_loss_dict)
            event_loss, event_loss_dict = event_disentanglement_loss(
                **output, num_props=self.model.num_props, **self.args['loss'])
            loss_dict.update(event_loss_dict)
            mixture_loss, mixture_loss_dict = mixture_pull_push_loss(
                **output, num_props=self.model.num_props,
                **self.args['loss'])
            loss_dict.update(mixture_loss_dict)
            coherence_loss, coherence_loss_dict = qcec_coherence_loss(
                **output, num_props=self.model.num_props,
                **self.args['loss'])
            loss_dict.update(coherence_loss_dict)
            loss = (reconstruction_loss + ranking_loss + event_loss
                    + mixture_loss + coherence_loss)
            loss_dict.update({
                'raw_rec_loss': float(reconstruction_loss.detach().item()),
                'weighted_rec_loss': float(reconstruction_loss.detach().item()),
                'raw_ivc_loss': float(ranking_loss.detach().item()),
                'weighted_ivc_loss': float(ranking_loss.detach().item()),
                'raw_event_loss': float(event_loss.detach().item()),
                'weighted_event_loss': float(event_loss.detach().item()),
                'raw_mixture_loss': float(mixture_loss.detach().item()),
                'weighted_mixture_loss': float(mixture_loss.detach().item()),
                'raw_qcec_loss': float(coherence_loss.detach().item()),
                'weighted_qcec_loss': float(coherence_loss.detach().item()),
                'total_loss': float(loss.detach().item()),
            })
            loss.backward()
            pre_clip_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 10)
            loss_dict['grad_norm_pre_clip'] = float(pre_clip_norm.detach().item())
            if self._has_nonzero_baseline_gradient():
                self._backbone_update_steps += 1
                self._epoch_backbone_update_steps += 1
            self.optimizer.step()

            self.num_updates += 1
            curr_lr = self.lr_scheduler.step_update(self.num_updates)
            time_meter.update()
            for k, v in loss_dict.items():
                loss_meter[k].update(v)
                epoch_loss_meter[k].update(v)

            if bid % display_n_batches == 0:
                print_log()

        if bid % display_n_batches != 0:
            print_log()

        epoch_metrics = self._meter_values(epoch_loss_meter)
        epoch_metrics.update({
            'backbone_update_steps_total': float(self._backbone_update_steps),
            'backbone_update_steps_epoch': float(
                self._epoch_backbone_update_steps),
        })
        epoch_metrics['_stage'] = {
            'qcec_only': bool(qcec_only),
            'trainable_numel': trainable_numel,
            'optimizer_managed_numel': optimizer_numel,
            'lr_start': current_lr,
            'lr_end': float(self.optimizer.get_lr()),
            'global_step_end': int(self.num_updates),
            'backbone_update_steps_total': int(self._backbone_update_steps),
            'backbone_update_steps_epoch': int(
                self._epoch_backbone_update_steps),
            'event_schedule': event_schedule,
        }
        return epoch_metrics

    def _log_training_stage(self, epoch, qcec_only, trainable_numel,
                            optimizer_numel, current_lr, event_schedule):
        info(
            'Training stage epoch {}: qcec_only={}, trainable_numel={}, '
            'optimizer_managed_numel={}, lr={:.8f}, global_step={}, '
            'backbone_update_steps_total={}, event_schedule={:.4f}'.format(
                epoch, bool(qcec_only), trainable_numel, optimizer_numel,
                current_lr, self.num_updates, self._backbone_update_steps,
                event_schedule))

    def _has_nonzero_baseline_gradient(self):
        qcec_ids = set(getattr(self.model, '_qcec_parameter_ids', set()))
        for parameter in self.model.parameters():
            if id(parameter) in qcec_ids or parameter.grad is None:
                continue
            if torch.isfinite(parameter.grad).all() and torch.count_nonzero(
                    parameter.grad).item() > 0:
                return True
        return False

    def eval(self, save=None, epoch=0, loader=None, split='Test'):
        # Keep evaluation deterministic and restore the training RNG streams
        # afterwards.  Per-sample mask seeds below additionally make the
        # result independent of DataLoader batching.
        with isolated_rng(self.eval_seed):
            return self._eval(
                save=save, epoch=epoch, loader=loader, split=split)

    def _eval(self, save=None, epoch=0, loader=None, split='Test'):
        loader = self.test_loader if loader is None else loader
        self.model.eval()
        with torch.no_grad():
            metrics_logger = collections.defaultdict(lambda: AverageMeter())
            raw_metrics_logger = collections.defaultdict(
                lambda: AverageMeter())
            diagnostics_logger = collections.defaultdict(
                lambda: AverageMeter())
            selection_diagnostics_logger = collections.defaultdict(
                lambda: AverageMeter())

            with torch.no_grad():
                for bid, batch in enumerate(loader, 1):
                    durations = np.asarray([i[1] for i in batch['raw']])
                    gt = np.asarray([i[2] for i in batch['raw']])

                    net_input = move_to_cuda(batch['net_input'])
                    output = self.model(epoch=epoch, **net_input)
                    bsz = len(durations)
                    num_props = self.model.num_props
                    k = min(num_props, 5)
                    
                    words_mask = output['words_mask'].unsqueeze(1) \
                        .expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)
                    words_id = output['words_id'].unsqueeze(1) \
                        .expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)

                    nll_loss, acc = cal_nll_loss(
                        output['words_logit'], words_id, words_mask)
                    nll_score = nll_loss.view(bsz, num_props)
                    proposal_score = nll_score.clone()
                    event_score = output.get('event_score')
                    if event_score is not None:
                        event_score = event_score.view(bsz, num_props)
                        if bool(self.args.get(
                                'selection_use_event_score', True)):
                            proposal_score = proposal_score - (
                                self.inference_event_weight * event_score)
                    idx = proposal_score.argsort(dim=-1)
                    sorted_proposal_score = proposal_score.gather(
                        index=idx, dim=-1).cpu().numpy()
                    nll_idx = nll_score.argsort(dim=-1)
                    nll_sorted_score = nll_score.gather(
                        index=nll_idx, dim=-1).cpu().numpy()

                    raw_width = output['width'].view(bsz, num_props)
                    raw_center = output['center'].view(bsz, num_props)
                    eval_width = output.get('qcec_eval_width')
                    eval_center = output.get('qcec_eval_center')
                    if eval_width is None or eval_center is None:
                        eval_width = raw_width
                        eval_center = raw_center
                    else:
                        eval_width = eval_width.view(bsz, num_props)
                        eval_center = eval_center.view(bsz, num_props)
                    width = eval_width.gather(index=idx, dim=-1)
                    center = eval_center.gather(index=idx, dim=-1)
                    selected_props = torch.stack([
                        torch.clamp(center - width / 2, min=0),
                        torch.clamp(center + width / 2, max=1),
                    ], dim=-1)
                    selected_props = selected_props.cpu().numpy()
                    gt = gt / durations[:, np.newaxis]

                    eval_props = torch.stack([
                        torch.clamp(eval_center - eval_width / 2, min=0),
                        torch.clamp(eval_center + eval_width / 2, max=1),
                    ], dim=-1)
                    eval_nll_sorted_props = eval_props.gather(
                        index=nll_idx.unsqueeze(-1).expand(-1, -1, 2),
                        dim=1).cpu().numpy()
                    raw_props = torch.stack([
                        torch.clamp(raw_center - raw_width / 2, min=0),
                        torch.clamp(raw_center + raw_width / 2, max=1),
                    ], dim=-1).cpu().numpy()
                    raw_nll_sorted_props = torch.stack([
                        torch.clamp(
                            raw_center.gather(index=nll_idx, dim=-1)
                            - raw_width.gather(index=nll_idx, dim=-1) / 2,
                            min=0),
                        torch.clamp(
                            raw_center.gather(index=nll_idx, dim=-1)
                            + raw_width.gather(index=nll_idx, dim=-1) / 2,
                            max=1),
                    ], dim=-1).cpu().numpy()
                    raw_sorted_props = torch.stack([
                        torch.clamp(
                            raw_center.gather(index=idx, dim=-1)
                            - raw_width.gather(index=idx, dim=-1) / 2,
                            min=0),
                        torch.clamp(
                            raw_center.gather(index=idx, dim=-1)
                            + raw_width.gather(index=idx, dim=-1) / 2,
                            max=1),
                    ], dim=-1).cpu().numpy()
                    qcec_positions = output.get(
                        'qcec_transition_positions')
                    qcec_barrier_lr = output.get('qcec_barrier_lr')
                    qcec_barrier_rl = output.get('qcec_barrier_rl')
                    qcec_transition_mask = output.get(
                        'qcec_transition_mask')
                    if (qcec_positions is not None
                            and qcec_barrier_lr is not None
                            and qcec_barrier_rl is not None
                            and qcec_transition_mask is not None):
                        raw_interval_tensor = torch.stack([
                            torch.clamp(raw_center - raw_width / 2, min=0),
                            torch.clamp(raw_center + raw_width / 2, max=1),
                        ], dim=-1)
                        eval_interval_tensor = torch.stack([
                            torch.clamp(
                                eval_center - eval_width / 2, min=0),
                            torch.clamp(
                                eval_center + eval_width / 2, max=1),
                        ], dim=-1)
                        crossing_diagnostics = {}
                        for prefix, intervals in (
                                ('raw_', raw_interval_tensor),
                                ('snapped_', eval_interval_tensor)):
                            crossing_diagnostics.update({
                                prefix + name: value
                                for name, value in calculate_qcec_crossing_diagnostics(
                                    intervals,
                                    qcec_positions,
                                    qcec_barrier_lr,
                                    qcec_barrier_rl,
                                    qcec_transition_mask,
                                    self.args['loss'].get(
                                        'qcec_cross_temperature', 0.02),
                                    ).items()
                            })
                        for name, (value, count) in crossing_diagnostics.items():
                            diagnostics_logger[name].update(value, count)
                    if output.get('qcec_snap_start_delta') is not None:
                        start_delta = output['qcec_snap_start_delta'].abs()
                        end_delta = output['qcec_snap_end_delta'].abs()
                        diagnostics_logger['qcec_mean_start_movement'].update(
                            float(start_delta.mean().item()), start_delta.numel())
                        diagnostics_logger['qcec_mean_end_movement'].update(
                            float(end_delta.mean().item()), end_delta.numel())
                    context_width = output.get('mixture_context_width')
                    if context_width is not None:
                        context_width = context_width.view(
                            bsz, num_props).cpu().numpy()
                    diagnostics = calculate_candidate_diagnostics(
                        raw_props=raw_props,
                        nll_sorted_props=eval_nll_sorted_props,
                        gt=gt,
                        context_width=context_width)
                    for name, (value, count) in diagnostics.items():
                        diagnostics_logger[name].update(value, count)

                    # Keep the unmodified intervals visible whenever
                    # inference snapping is enabled.  With snapping disabled
                    # these metrics intentionally equal the main metrics.
                    raw_diagnostics = calculate_candidate_diagnostics(
                        raw_props=raw_props,
                        nll_sorted_props=raw_nll_sorted_props,
                        gt=gt,
                        context_width=context_width)
                    for name, (value, count) in raw_diagnostics.items():
                        diagnostics_logger['raw_' + name].update(value, count)

                    # Rank-1 selector diagnostics share the exact same
                    # forward pass, proposal tensors and NLL values.  This
                    # isolates ranking from candidate generation and avoids
                    # comparing two selectors that saw different masks.
                    charades_anchor = (
                        self.args['dataset']['dataset'] == 'CharadesSTA')
                    for prefix, candidates, scores in (
                            ('', eval_nll_sorted_props, nll_sorted_score),
                            ('raw_', raw_nll_sorted_props, nll_sorted_score)):
                        nll_selected_index = select_proposal_by_strategy(
                            candidates, scores, strategy='nll',
                            temperature=self.selection_temperature,
                            charades_anchor=charades_anchor)
                        vote_selected_index = select_proposal_by_strategy(
                            candidates, scores, strategy='semantic_vote',
                            temperature=self.selection_temperature,
                            charades_anchor=charades_anchor)
                        nll_res = top_1_metric(
                            candidates[np.arange(bsz), nll_selected_index], gt)
                        vote_res = top_1_metric(
                            candidates[np.arange(bsz), vote_selected_index], gt)
                        for selector_name, selector_result in (
                                ('nll', nll_res),
                                ('semantic_vote', vote_res)):
                            for key, value in selector_result.items():
                                selection_diagnostics_logger[
                                    prefix + selector_name + '_R@1,' + key
                                ].update(value, bsz)

                    selected_index = select_proposal_by_strategy(
                        selected_props,
                        sorted_proposal_score,
                        strategy=self.selection_strategy,
                        temperature=self.selection_temperature,
                        charades_anchor=charades_anchor)
                    res = top_1_metric(
                        selected_props[np.arange(bsz), selected_index], gt)
                    
                    for key, v in res.items():
                        metrics_logger['R@1,'+key].update(v, bsz)
                    res = top_n_metric(selected_props[:, :k].transpose(1, 0, 2), gt)
                    for key, v in res.items():
                        metrics_logger['R@%d,'%(k)+key].update(v, bsz)

                    raw_selected_index = select_proposal_by_strategy(
                        raw_sorted_props,
                        sorted_proposal_score,
                        strategy=self.selection_strategy,
                        temperature=self.selection_temperature,
                        charades_anchor=charades_anchor)
                    raw_res = top_1_metric(
                        raw_sorted_props[
                            np.arange(bsz), raw_selected_index], gt)
                    for key, value in raw_res.items():
                        raw_metrics_logger['R@1,' + key].update(value, bsz)
                    raw_res = top_n_metric(
                        raw_sorted_props[:, :k].transpose(1, 0, 2), gt)
                    for key, value in raw_res.items():
                        raw_metrics_logger['R@%d,' % k + key].update(value, bsz)

            msg = '|'.join([' {} {:.4f} '.format(k, v.avg) for k, v in metrics_logger.items()])
            info('{}: |{}|'.format(split, msg))
            raw_msg = '|'.join([
                ' {} {:.4f} '.format(k, v.avg)
                for k, v in raw_metrics_logger.items()])
            info('Raw {}: |{}|'.format(split, raw_msg))
            diagnostic_msg = '|'.join([
                ' {} {:.4f} '.format(k, v.avg)
                for k, v in diagnostics_logger.items()
                if v.count > 0])
            info('Diagnostics({}): |{}|'.format(
                split, diagnostic_msg))
            selection_diagnostics = self._meter_values(
                selection_diagnostics_logger)
            info('Selector diagnostics({}): |{}|'.format(
                split, '|'.join([
                    ' {} {:.4f} '.format(k, value)
                    for k, value in selection_diagnostics.items()
                ])))
            payload = {
                'kind': 'eval',
                'epoch': int(epoch),
                'split': split,
                'eval_seed': self.eval_seed,
                'selection_strategy': self.selection_strategy,
                'selection_use_event_score': bool(self.args.get(
                    'selection_use_event_score', True)),
                'metrics': self._meter_values(metrics_logger),
                'raw_metrics': self._meter_values(raw_metrics_logger),
                'diagnostics': self._meter_values(diagnostics_logger),
                'selection_diagnostics': selection_diagnostics,
            }
            self.last_eval_payload = payload
            self._record_metrics(payload)
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
        worker_seed = int(self.args.get('seed', 8))

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

            set_seed(worker_seed + worker_id)

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

    def _save_model(self, path, epoch=None):
        state_dict = {
            'num_updates': self.num_updates,
            'epoch': int(epoch if epoch is not None else getattr(
                self, 'current_epoch', self.start_epoch)),
            'config': self.args,
            'model_parameters': self.model.state_dict(),
            'initialization': self.args.get('initialization', {}),
            'run_metadata': getattr(self, 'run_metadata', {}),
            # A resume should reproduce both the optimizer trajectory and
            # the random masking/data stream as closely as the DataLoader
            # implementation permits.
            'rng_state': torch.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'python_rng_state': random.getstate(),
        }
        if torch.cuda.is_available():
            state_dict['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
        if hasattr(self, 'optimizer'):
            state_dict['optimizer_state'] = self.optimizer.state_dict()
        torch.save(state_dict, path)
        info('save model to {}, epoch {}, num_updates {}.'.format(
            path, state_dict['epoch'], self.num_updates))

    def _save_initial_state(self, path):
        """Save the disabled-baseline public initialization for A/C runs."""
        if not hasattr(self, 'model'):
            raise RuntimeError('cannot save an initial state before model build')
        path = os.path.abspath(os.fspath(path))
        if os.path.exists(path):
            raise FileExistsError(
                'refusing to overwrite initial state {}; choose a new path'
                .format(path))
        os.makedirs(os.path.dirname(path) or os.curdir, exist_ok=True)
        state = {
            'model_parameters': {
                name: value.detach().cpu().clone()
                for name, value in self.model.state_dict().items()
            },
            'config': self.args,
            'initialization': {
                'source': 'random-baseline-initialization',
                'kind': 'public_initial_state',
                'valid_for_qcec_freeze': False,
            },
            'seed': self.args.get('seed'),
        }
        torch.save(state, path)
        self.args.setdefault('initialization', {}).update({
            'public_state_path': path,
            'public_state_sha256': sha256_file(path),
        })
        self._write_run_metadata()
        info('saved public initial state to {}.'.format(path))
        return path

    @staticmethod
    def _restore_rng_state(state_dict):
        if state_dict.get('rng_state') is not None:
            torch.set_rng_state(state_dict['rng_state'])
        if state_dict.get('numpy_rng_state') is not None:
            np.random.set_state(state_dict['numpy_rng_state'])
        if state_dict.get('python_rng_state') is not None:
            random.setstate(state_dict['python_rng_state'])
        if (state_dict.get('cuda_rng_state_all') is not None
                and torch.cuda.is_available()):
            torch.cuda.set_rng_state_all(state_dict['cuda_rng_state_all'])

    def _load_model(self, path):
        state_dict = torch.load(path, map_location='cpu')
        self.num_updates = int(state_dict.get('num_updates', 0))
        self.start_epoch = int(state_dict.get('epoch', 0))
        parameters = state_dict['model_parameters']
        self.model.load_state_dict(parameters)
        optimizer_state = state_dict.get('optimizer_state')
        if optimizer_state is not None and hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(optimizer_state)
        # The update count is the source of truth for the copied scheduler;
        # apply it after loading an optional optimizer state so a stale saved
        # param-group learning rate cannot win over the resumed schedule.
        self.lr_scheduler.step_update(self.num_updates)
        previous_initialization = state_dict.get(
            'initialization',
            state_dict.get('config', {}).get('initialization', {}))
        if not isinstance(previous_initialization, dict):
            previous_initialization = {}
        self._set_initialization_metadata(
            source='resume:{}'.format(path), kind='resume',
            valid=bool(previous_initialization.get(
                'valid_for_qcec_freeze', False)))
        self.args['initialization']['state_sha256'] = sha256_file(path)
        self._write_run_metadata()
        self._restore_rng_state(state_dict)
        if 'epoch' not in state_dict:
            info('checkpoint {} has no epoch; resume starts at epoch 1.'.format(
                path))
        info('load model from {}, epoch {}, num_updates {}.'.format(
            path, self.start_epoch, self.num_updates))

    def _load_public_initial_state(self, path):
        """Copy only common baseline tensors into a fresh model.

        A QCEC target is allowed to have its newly introduced QCEC/adapter
        tensors missing from the disabled baseline state.  Every public
        tensor must otherwise match by name and shape; silently dropping a
        backbone tensor would invalidate the B/A/C comparison.
        """
        if not os.path.exists(path):
            raise FileNotFoundError('public initial state does not exist: {}'.format(path))
        state_dict = torch.load(path, map_location='cpu')
        source = state_dict.get('model_parameters', state_dict)
        target = self.model.state_dict()
        allowed_prefixes = ('qcec.', 'qcec_adapter.', 'qcec_')
        source_qcec = [
            name for name in source
            if name.startswith(allowed_prefixes)]
        if source_qcec:
            raise ValueError(
                'public initial state must come from a QCEC-disabled model; '
                'found QCEC tensors: {}'.format(source_qcec))

        compatible = {}
        unexpected = []
        for name, value in source.items():
            if name not in target:
                unexpected.append(name)
                continue
            if target[name].shape != value.shape:
                raise ValueError(
                    'public tensor {} has shape {} but target expects {}'
                    .format(name, tuple(value.shape), tuple(target[name].shape)))
            compatible[name] = value
        result = self.model.load_state_dict(compatible, strict=False)
        illegal_missing = [
            name for name in result.missing_keys
            if not name.startswith(allowed_prefixes)]
        if unexpected or result.unexpected_keys or illegal_missing:
            raise ValueError(
                'public initial state is incompatible; unexpected={}, '
                'illegal_missing={}'.format(
                    unexpected + list(result.unexpected_keys), illegal_missing))
        self.num_updates = 0
        self.start_epoch = 0
        self.lr_scheduler.step_update(0)
        self._reset_runtime_rng(int(self.args.get('seed', 8)))
        self._set_initialization_metadata(
            source='public_initial_state:{}'.format(os.path.abspath(path)),
            kind='public_initial_state', valid=False)
        self.args['initialization']['state_sha256'] = sha256_file(path)
        self._write_run_metadata()
        info('initialize model from public baseline state {}.'.format(path))
        info('loaded {} public tensors; QCEC tensors left at initialization: {}.'.format(
            len(compatible), result.missing_keys))

    def _load_baseline_model(self, path):
        """Warm-start an enabled QCEC model from a matching baseline model.

        Unlike the V3 compatibility loader, this path expects the existing
        baseline architecture to match exactly.  Only newly introduced QCEC
        parameters may be absent; silently swallowing a missing backbone
        tensor would make a misleading experiment.
        """
        if not getattr(self.model, 'use_qcec', False):
            raise ValueError(
                '--init-from-baseline requires a QCEC-enabled model')
        state_dict = torch.load(path, map_location='cpu')
        source = state_dict.get('model_parameters', state_dict)
        target = self.model.state_dict()
        compatible = {}
        skipped = []
        for name, value in source.items():
            if name not in target:
                skipped.append('{} (not used by QCEC)'.format(name))
                continue
            if target[name].shape != value.shape:
                raise ValueError(
                    'baseline tensor {} has shape {} but QCEC expects {}'
                    .format(name, tuple(value.shape), tuple(target[name].shape)))
            compatible[name] = value

        result = self.model.load_state_dict(compatible, strict=False)
        allowed_prefixes = ('qcec.', 'qcec_adapter.', 'qcec_')
        illegal_missing = [
            key for key in result.missing_keys
            if not key.startswith(allowed_prefixes)
        ]
        if result.unexpected_keys or illegal_missing:
            raise ValueError(
                'baseline checkpoint is incompatible; unexpected={}, '
                'illegal_missing={}'.format(
                    result.unexpected_keys, illegal_missing))
        self.num_updates = 0
        self.start_epoch = 0
        self.lr_scheduler.step_update(0)
        self._reset_runtime_rng(int(self.args.get('seed', 8)))
        self._set_initialization_metadata(
            source='baseline_checkpoint:{}'.format(os.path.abspath(path)),
            kind='baseline_checkpoint', valid=True)
        self.args['initialization']['state_sha256'] = sha256_file(path)
        self._write_run_metadata()
        info('initialize QCEC from baseline checkpoint {}.'.format(path))
        info('loaded {} tensors; QCEC tensors left at initialization: {}.'.format(
            len(compatible), result.missing_keys))
        if skipped:
            info('ignored source tensors: {}'.format(skipped))

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
        self._reset_runtime_rng(int(getattr(self, 'args', {}).get('seed', 8)))
        self._set_initialization_metadata(
            source='v3_checkpoint:{}'.format(os.path.abspath(path)),
            kind='v3_checkpoint', valid=True)
        self.args['initialization']['state_sha256'] = sha256_file(path)
        self._write_run_metadata()

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


def calculate_qcec_crossing_diagnostics(
        intervals, transition_positions, barrier_lr, barrier_rl,
        transition_mask, temperature=0.02):
    """Measure proposal crossing of active directional QCEC barriers."""
    if intervals.dim() != 3 or intervals.size(-1) != 2:
        raise ValueError('intervals must have shape [B, N, 2]')
    batch_size, num_props, _ = intervals.shape
    if transition_positions.dim() != 2:
        raise ValueError('transition_positions must have shape [B, J]')
    if transition_positions.size(0) != batch_size:
        raise ValueError('QCEC diagnostics batch sizes do not match')
    if barrier_lr.shape != transition_positions.shape:
        raise ValueError('barrier_lr shape does not match transition positions')
    if barrier_rl.shape != transition_positions.shape:
        raise ValueError('barrier_rl shape does not match transition positions')
    if transition_mask.shape != transition_positions.shape:
        raise ValueError('transition mask shape does not match positions')
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    if transition_positions.size(-1) == 0:
        return {
            'active_barrier_fraction': (0.0, batch_size),
            'active_barrier_count': (0.0, batch_size),
            'proposal_crossing_fraction': (0.0, batch_size * num_props),
        }

    intervals = intervals.float()
    positions = transition_positions.to(
        device=intervals.device, dtype=torch.float32)
    mask = transition_mask.to(device=intervals.device, dtype=torch.bool)
    left_barrier = barrier_lr.to(
        device=intervals.device, dtype=torch.float32)
    right_barrier = barrier_rl.to(
        device=intervals.device, dtype=torch.float32)
    active = ((left_barrier + right_barrier) > 0) & mask
    proposal_start = intervals[..., 0]
    proposal_end = intervals[..., 1]
    gate = torch.sigmoid(torch.clamp(
        (positions.unsqueeze(1) - proposal_start.unsqueeze(-1)) / temperature,
        min=-30.0, max=30.0)) * torch.sigmoid(torch.clamp(
            (proposal_end.unsqueeze(-1) - positions.unsqueeze(1)) / temperature,
            min=-30.0, max=30.0))
    active_float = active.float()
    active_fraction = active_float.sum() / mask.float().sum().clamp_min(1.0)
    active_count = active_float.sum(dim=-1).mean()
    crossing_fraction = (
        (gate * active_float.unsqueeze(1)).sum(dim=-1) > 0.5
    ).float().mean()
    return {
        'active_barrier_fraction': (
            float(active_fraction.item()), batch_size),
        'active_barrier_count': (
            float(active_count.item()), batch_size),
        'proposal_crossing_fraction': (
            float(crossing_fraction.item()), batch_size * num_props),
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
    quantiles = (0.05, 0.25, 0.50, 0.75, 0.95)
    for quantile in quantiles:
        suffix = 'p{:02d}'.format(int(round(quantile * 100)))
        diagnostics['predicted_width_' + suffix] = (
            float(np.percentile(raw_width, quantile * 100)), raw_width.size)
    for proposal_index in range(num_props):
        proposal_iou = calculate_IoU_batch(
            (raw_props[:, proposal_index, 0],
             raw_props[:, proposal_index, 1]),
            (gt[:, 0], gt[:, 1]))
        prefix = 'proposal{}_'.format(proposal_index + 1)
        for quantile in quantiles:
            suffix = 'width_p{:02d}'.format(int(round(quantile * 100)))
            diagnostics[prefix + suffix] = (
                float(np.percentile(
                    raw_width[:, proposal_index], quantile * 100)), batch_size)
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
