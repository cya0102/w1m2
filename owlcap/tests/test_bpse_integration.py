import torch

from models.cpl import CPL
from models.loss import bpse_loss, ivc_loss, rec_loss


def _model():
    return CPL({
        'frames_input_size': 8,
        'words_input_size': 6,
        'hidden_size': 8,
        'vocab_size': 11,
        'use_negative': False,
        'num_props': 3,
        'sigma': 9,
        'gamma': 0,
        'dropout': 0,
        'max_epoch': 5,
        'DualTransformer': {
            'd_model': 8, 'num_heads': 2,
            'num_decoder_layers1': 1, 'num_decoder_layers2': 1,
            'dropout': 0,
        },
        'proposal_generator': {'type': 'single_gaussian'},
        'bpse': {
            'enabled': True,
            'max_query_units': 12,
            'num_event_tokens': 4,
            'quality_hidden_size': 8,
            'quality_dropout': 0,
            'eval_word_mask_mode': 'deterministic_topk',
        },
    })


def _batch():
    batch, frames, words, hidden = 2, 20, 5, 8
    unit_mask = torch.zeros(batch, 12, words)
    unit_mask[:, 0, :3] = 1
    unit_mask[:, 1, 3:] = 1
    valid = torch.zeros(batch, 12, dtype=torch.bool)
    valid[:, :2] = True
    unit_weights = torch.zeros(batch, 12)
    unit_weights[:, :2] = .5
    word_weights = torch.zeros(batch, words)
    word_weights[0] = .2
    word_weights[1, :3] = 1. / 3.
    return dict(
        frames_feat=torch.randn(batch, frames, hidden),
        frames_len=torch.tensor([frames, frames - 5]),
        words_id=torch.randint(0, 11, (batch, words)),
        words_feat=torch.randn(batch, words + 1, 6),
        words_len=torch.tensor([words, 3]),
        weights=word_weights,
        epoch=1,
        unit_token_mask=unit_mask,
        unit_valid_mask=valid,
        unit_weights=unit_weights,
    )


def test_cpl_bpse_train_eval_shapes_and_backward():
    model = _model()
    model.train()
    output = model(**_batch())
    assert output['words_logit'].shape == (6, 5, 11)
    assert output['bpse_quality_logits'].shape == (2, 3)
    assert output['bpse_group_quality_logits'].shape == (2, 3, 9)
    reconstruction, _ = rec_loss(**output, num_props=3)
    ranking, _ = ivc_loss(**output, num_props=3, margin_1=.1,
                          margin_2=.2, alpha_1=1., alpha_2=0., **{'lambda': .1})
    quality, _ = bpse_loss(**output)
    (reconstruction + ranking + quality).backward()
    assert model.bpse_scorer.quality_head[-1].weight.grad is not None
    assert torch.isfinite(model.bpse_scorer.quality_head[-1].weight.grad).all()

    model.eval()
    with torch.no_grad():
        evaluation = model(**_batch())
    assert evaluation['bpse_group_quality_logits'] is None
    assert torch.isfinite(evaluation['bpse_quality_logits']).all()
