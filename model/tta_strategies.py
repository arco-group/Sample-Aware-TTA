import torch

# Inference-time adaptation and configuration selection use only AE feature
# reconstruction errors, which do not require target references. SSIM, MAE,
# and PSNR compare against the paired target and are computed only once, after
# the configuration has been selected, for reference-based final evaluation.
import pandas as pd
from util.visualizer import calcola_mse, calculate_psnr, calculate_ssim
from collections import OrderedDict
import numpy as np
from itertools import combinations
import random
import warnings

warnings.filterwarnings("ignore")

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)

def compute_tnet_dim(opt):
    layers_to_dim = {'input': 1, 'first_conv': 64, 'second_conv': 128, 'third_conv': 256, 'resnet_block_1': 256,
                     'resnet_block_2': 256, 'resnet_block_3': 256, 'resnet_block_4': 256, 'final_output': 1}
    tnet_dim = []
    for layer in opt.return_layers:
        tnet_dim.append(layers_to_dim[layer])
    opt.tnet_dim = tnet_dim


def l2_reg_ortho(model, lambda_l2=1e-4):
    device = next(model.parameters()).device
    l2_loss = torch.tensor(0.0, device=device)
    for param in model.parameters():
        if param.requires_grad:
            l2_loss += torch.norm(param, p=2) ** 2
    return lambda_l2 * l2_loss


def TTA_rndm_50(adaptors, opt, task_model, save_dir, batch, rec_loss, stable=False, return_layers=None, plot=False, ae_model=None):
    n = len(return_layers[1:-1])
    indexs = [i for i in range(n)]
    tutte_combinazioni = set()
    num_random_comb = opt.__dict__.get('num_random_comb', 50)
    while len(tutte_combinazioni) < num_random_comb:
        r = random.randint(1, len(indexs))
        tutte_combinazioni.add(tuple(sorted(random.sample(indexs, r))))
    tutte_combinazioni = list(tutte_combinazioni)

    orthw, rec_loss, loss_config_output = _tta_prepare_strategy(adaptors, opt, rec_loss, ae_model)

    for comb in tutte_combinazioni:
        row, candidate_loss = _tta_evaluate_combination(adaptors, opt, task_model, batch, return_layers, comb, orthw)
        loss_config_output = _tta_append_loss_row(loss_config_output, row)

    return _tta_finalize_result(adaptors, opt, task_model, batch, return_layers, loss_config_output)


def _tta_comb_to_return_layers(comb_to_print, return_layers):
    n = len(return_layers[1:-1])
    chosen_comb = [0] + [x + 1 for x in sorted(comb_to_print)] + [n + 1]
    return [return_layers[i] for i in chosen_comb]


def _tta_comb_to_string(comb_to_print):
    return '_'.join([str(x + 1) for x in sorted(comb_to_print)])


def _tta_append_loss_row(loss_config_output, row):
    return pd.concat([loss_config_output, pd.DataFrame([row])], ignore_index=True)


def _tta_clone_adaptor_state(adaptors):
    """Snapshot adaptor weights in CPU RAM without changing optimizer state."""
    return {name: value.detach().cpu().clone() for name, value in adaptors.state_dict().items()}


def _tta_load_adaptor_state(adaptors, state):
    adaptors.load_state_dict(state)


def _tta_evaluate_combination(adaptors, opt, task_model, batch, return_layers, comb_to_print, orthw):
    comb_to_print = sorted(list(comb_to_print))
    opt.return_layers = _tta_comb_to_return_layers(comb_to_print, return_layers)
    compute_tnet_dim(opt)

    AE = adaptors._tta_ae_model
    active_ae_indices = [return_layers.index(name) for name in opt.return_layers]
    adaptors.reset(default=True)

    prev_loss = float('inf')
    loss_tot = []
    loss_output = []
    best_epoch_loss = float('inf')
    best_epoch_state = None

    for epoch in range(opt.tepochs):
        outputs = adaptors(batch, task_model, opt.model)

        loss = 0
        print('---------------------------------')

        for i, ae_index in enumerate(active_ae_indices):
            index = opt.return_layers[i]
            side_out = outputs[index]
            level_loss = 0

            if len(AE.AENetMatch[ae_index]) == 2:
                side_out_cat = torch.cat([side_out[0], side_out[1]], dim=1)
            else:
                side_out_cat = side_out

            ae_out = AE.AENet[ae_index](side_out_cat, side_out=False)

            scale = side_out_cat.pow(2).mean().sqrt().detach()
            level_loss = AE.AELoss(ae_out, side_out_cat) / (scale + 1e-6)

            print(f'loss {i} epoch {epoch}: {level_loss}')
            loss += level_loss

        loss_output.append(loss.detach().item())
        org_loss = orthw * l2_reg_ortho(adaptors.conv)
        loss += org_loss
        loss_tot.append(loss.data.item())

        adaptors.optimizer_ANet.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adaptors.parameters(), max_norm=10.0)
        adaptors.optimizer_ANet.step()

        if loss_output[-1] < best_epoch_loss:
            best_epoch_loss = loss_output[-1]
            best_epoch_state = _tta_clone_adaptor_state(adaptors)

        if prev_loss < loss:
            break
        else:
            prev_loss = loss

    min_index = loss_output.index(min(loss_output))
    used_comb = _tta_comb_to_string(comb_to_print)
    _tta_load_adaptor_state(adaptors, best_epoch_state)
    adaptors._tta_config_states[used_comb] = best_epoch_state
    del outputs, loss, org_loss, ae_out, side_out_cat

    row = {
        'config': used_comb,
        'loss_output': loss_output[min_index],
        'loss_tot': loss_tot[min_index] / len(comb_to_print),
    }

    return row, min(loss_output)


def _tta_finalize_result(adaptors, opt, task_model, batch, return_layers, loss_config_output):
    n = len(return_layers[1:-1])
    min_loss = loss_config_output['loss_output'].min()
    used_comb = loss_config_output.nsmallest(1, 'loss_output')['config'].iloc[0]

    _tta_load_adaptor_state(adaptors, adaptors._tta_config_states[used_comb])
    chosen_comb = used_comb.split('_')
    chosen_comb = [int(x) - 1 for x in chosen_comb]
    chosen_comb = sorted(chosen_comb)
    chosen_comb = [0] + [x + 1 for x in chosen_comb] + [n + 1]
    chosen_comb = [return_layers[i] for i in chosen_comb]
    opt.return_layers = chosen_comb
    compute_tnet_dim(opt)

    # Reference-based evaluation only: omega* has already been selected above.
    with torch.no_grad():
        outputs = adaptors(batch, task_model, opt.model)
    real_B = task_model.real_B
    fake_B = outputs[opt.return_layers[-1]]
    visuals_output = OrderedDict()
    visuals_output['real_B'] = real_B
    visuals_output['fake_B'] = fake_B

    ssim_score = calculate_ssim(visuals_output)
    mae_score = calcola_mse(visuals_output)
    psnr_score = calculate_psnr(visuals_output)

    selected_loss_output = float(loss_config_output[loss_config_output['config'] == used_comb]['loss_output'].values[0])
    del outputs, fake_B, visuals_output
    adaptors._tta_config_states.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return used_comb, ssim_score, mae_score, psnr_score, min_loss, selected_loss_output


def _tta_prepare_strategy(adaptors, opt, rec_loss, ae_model):
    if ae_model is None:
        raise ValueError("A preloaded ae_model is required for TTA")
    # Avoid registering the AE as an adaptor child: adaptor snapshots must
    # contain exactly the weights previously written by save_networks().
    object.__setattr__(adaptors, '_tta_ae_model', ae_model)
    adaptors._tta_config_states = {}
    for subnet in ae_model.AENet:
        subnet.train()  # Match the former per-candidate AENet construction.
    adaptors.set_requires_grad([adaptors.adpNet, adaptors.conv], True)
    adaptors.train()
    orthw = opt.__dict__.get('orthw', 1)
    rec_loss = round(rec_loss.item(), 4)
    loss_config_output = pd.DataFrame(columns=['config', 'loss_output', 'loss_tot'])
    return orthw, rec_loss, loss_config_output


def TTA_rndm_10(adaptors, opt, task_model, save_dir, batch, rec_loss, stable=False, return_layers=None, plot=False, ae_model=None):
    n = len(return_layers[1:-1])
    indexs = [i for i in range(n)]
    tutte_combinazioni = set()
    num_random_comb = opt.__dict__.get('num_random_comb', 10)
    while len(tutte_combinazioni) < num_random_comb:
        r = random.randint(1, len(indexs))
        tutte_combinazioni.add(tuple(sorted(random.sample(indexs, r))))
    tutte_combinazioni = list(tutte_combinazioni)

    orthw, rec_loss, loss_config_output = _tta_prepare_strategy(adaptors, opt, rec_loss, ae_model)

    for comb in tutte_combinazioni:
        row, candidate_loss = _tta_evaluate_combination(adaptors, opt, task_model, batch, return_layers, comb, orthw)
        loss_config_output = _tta_append_loss_row(loss_config_output, row)

    return _tta_finalize_result(adaptors, opt, task_model, batch, return_layers, loss_config_output)


def TTA_grid(adaptors, opt, task_model, save_dir, batch, rec_loss, stable=False, return_layers=None, plot=False, ae_model=None):
    n = len(return_layers[1:-1])
    indexs = [i for i in range(n)]
    tutte_combinazioni = []
    for r in range(1, len(indexs) + 1):
        tutte_combinazioni.extend(combinations(indexs, r))

    orthw, rec_loss, loss_config_output = _tta_prepare_strategy(adaptors, opt, rec_loss, ae_model)

    for comb in tutte_combinazioni:
        row, candidate_loss = _tta_evaluate_combination(adaptors, opt, task_model, batch, return_layers, comb, orthw)
        loss_config_output = _tta_append_loss_row(loss_config_output, row)

    return _tta_finalize_result(adaptors, opt, task_model, batch, return_layers, loss_config_output)


def TTA_forward(adaptors, opt, task_model, save_dir, batch, rec_loss, stable=False, return_layers=None, plot=False, ae_model=None):
    n = len(return_layers[1:-1])
    indexs = [i for i in range(n)]
    selected = []
    remaining = list(indexs)
    best_loss = float('inf')
    best_comb = None
    improved = True

    orthw, rec_loss, loss_config_output = _tta_prepare_strategy(adaptors, opt, rec_loss, ae_model)

    while improved and remaining:
        improved = False
        best_i = None
        for i in remaining:
            candidate = sorted(selected + [i])
            row, candidate_loss = _tta_evaluate_combination(adaptors, opt, task_model, batch, return_layers, candidate, orthw)
            loss_config_output = _tta_append_loss_row(loss_config_output, row)

            if candidate_loss < best_loss:
                best_loss = candidate_loss
                best_comb = candidate
                best_i = i

        if best_comb is not None and best_comb != selected:
            selected = best_comb
            remaining.remove(best_i)
            improved = True

    return _tta_finalize_result(adaptors, opt, task_model, batch, return_layers, loss_config_output)


def TTA_backward(adaptors, opt, task_model, save_dir, batch, rec_loss, stable=False, return_layers=None, plot=False, ae_model=None):
    n = len(return_layers[1:-1])
    indexs = [i for i in range(n)]
    selected = list(indexs)
    best_loss = float('inf')
    best_comb = None

    orthw, rec_loss, loss_config_output = _tta_prepare_strategy(adaptors, opt, rec_loss, ae_model)

    row, candidate_loss = _tta_evaluate_combination(adaptors, opt, task_model, batch, return_layers, selected, orthw)
    loss_config_output = _tta_append_loss_row(loss_config_output, row)
    if candidate_loss < best_loss:
        best_loss = candidate_loss
        best_comb = selected

    improved = True
    while improved and len(selected) > 1:
        improved = False
        best_i = None
        for i in selected:
            candidate = sorted([x for x in selected if x != i])
            row, candidate_loss = _tta_evaluate_combination(adaptors, opt, task_model, batch, return_layers, candidate, orthw)
            loss_config_output = _tta_append_loss_row(loss_config_output, row)

            if candidate_loss < best_loss:
                best_loss = candidate_loss
                best_comb = candidate
                best_i = i

        if best_comb is not None and best_i is not None:
            selected = best_comb
            improved = True

    return _tta_finalize_result(adaptors, opt, task_model, batch, return_layers, loss_config_output)


def TTA_bayesian(adaptors, opt, task_model, save_dir, batch, rec_loss, stable=False, return_layers=None, plot=False, ae_model=None):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n = len(return_layers[1:-1])
    indexs = [i for i in range(n)]
    orthw, rec_loss, loss_config_output = _tta_prepare_strategy(adaptors, opt, rec_loss, ae_model)

    def objective(trial):
        nonlocal loss_config_output

        n_layers = len(indexs)
        config_bits = [trial.suggest_int(f"b{i}", 0, 1) for i in range(n_layers)]
        if sum(config_bits) == 0:
            return float('inf')

        comb_to_print = sorted([i for i, val in enumerate(config_bits) if val == 1])
        row, candidate_loss = _tta_evaluate_combination(
            adaptors, opt, task_model, batch, return_layers, comb_to_print, orthw
        )
        loss_config_output = _tta_append_loss_row(loss_config_output, row)
        return candidate_loss

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=5)
    )
    study.optimize(objective, n_trials=opt.__dict__.get('n_trials', 20))

    return _tta_finalize_result(adaptors, opt, task_model, batch, return_layers, loss_config_output)
