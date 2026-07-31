from models.adaptor_3 import ANet
import torch
from models.Autoencoder_model import AENet
import pandas as pd
import os
from util.visualizer import calcola_mse, calculate_psnr, calculate_ssim
from collections import OrderedDict
import numpy as np
from itertools import combinations
import random
import warnings
import shutil

warnings.filterwarnings("ignore")

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

def compute_tnet_dim(opt):
    layers_to_dim = {'input': 1, 'first_conv': 64, 'second_conv': 128, 'third_conv': 256, 'resnet_block_1': 256,
                     'resnet_block_2': 256, 'resnet_block_3': 256, 'resnet_block_4': 256, 'final_output': 1}
    tnet_dim = []
    for layer in opt.return_layers:
        tnet_dim.append(layers_to_dim[layer])
    opt.tnet_dim = tnet_dim


def l2_reg_ortho(model, lambda_l2=1e-4):
    l2_loss = torch.tensor(0.0, device='cuda')
    for param in model.parameters():
        if param.requires_grad:
            l2_loss += torch.norm(param, p=2) ** 2
    return lambda_l2 * l2_loss


def TTA_rndm_50(adaptors, opt, task_model, save_dir, batch, rec_loss, stable=False, return_layers=None, plot=False, psnr_score_no_tta=0.0):
    n = len(return_layers[1:-1])
    indexs = [i for i in range(n)]
    tutte_combinazioni = set()
    num_random_comb = opt.__dict__.get('num_random_comb', 50)
    while len(tutte_combinazioni) < num_random_comb:
        r = random.randint(1, len(indexs))
        tutte_combinazioni.add(tuple(sorted(random.sample(indexs, r))))
    tutte_combinazioni = list(tutte_combinazioni)

    orthw, rec_loss, loss_config_output = _tta_prepare_strategy(adaptors, opt, rec_loss)

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


def _tta_remove_path(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.isfile(path):
        os.remove(path)


def _tta_cleanup_adaptor_weights(adaptors, opt, loss_config_output):
    epoch_names = {str(epoch) for epoch in range(opt.tepochs)}
    config_names = set()
    if not loss_config_output.empty and 'config' in loss_config_output:
        config_names = {str(config) for config in loss_config_output['config'].dropna().unique()}

    cleanup_targets = [
        (getattr(adaptors, 'save_dir', None), epoch_names),
        (getattr(adaptors, 'save_dir_config', None), config_names),
    ]

    for root, names in cleanup_targets:
        if not root:
            continue
        root_abs = os.path.abspath(root)
        for name in names:
            path = os.path.abspath(os.path.join(root_abs, str(name)))
            try:
                if os.path.commonpath([root_abs, path]) != root_abs:
                    continue
                _tta_remove_path(path)
            except FileNotFoundError:
                continue
            except OSError as exc:
                print(f"Warning: could not remove adaptor checkpoint path {path}: {exc}")


def _tta_evaluate_combination(adaptors, opt, task_model, batch, return_layers, comb_to_print, orthw):
    comb_to_print = sorted(list(comb_to_print))
    opt.return_layers = _tta_comb_to_return_layers(comb_to_print, return_layers)
    compute_tnet_dim(opt)

    if not opt.ae_checkpoint_dir:
        raise ValueError("opt.ae_checkpoint_dir must be set to load AE checkpoints")
    ae_epoch = str(opt.ae_epoch)
    AE = AENet(opt)
    for i in range(len(opt.return_layers)):
        name = opt.return_layers[i]
        elem = opt.tnet_dim[i]
        load_path_weights = os.path.join(opt.ae_checkpoint_dir, f'AE_{name}_{ae_epoch}.pt')

        state_dict = torch.load(load_path_weights, map_location=str(AE.device))

        AE.AENet[i].load_state_dict(state_dict)
        AE.set_requires_grad(AE.AENet[i], False)
    adaptors.reset(default=True)

    prev_loss = float('inf')
    loss_tot = []
    loss_output = []

    for epoch in range(opt.tepochs):
        outputs = adaptors(batch, task_model, opt.model)

        loss = 0
        print('---------------------------------')

        for i in range(len(AE.AENet)):
            index = opt.return_layers[i]
            side_out = outputs[index]
            level_loss = 0

            if len(AE.AENetMatch[i]) == 2:
                side_out_cat = torch.cat([side_out[0], side_out[1]], dim=1)
            else:
                side_out_cat = side_out

            ae_out = AE.AENet[i](side_out_cat, side_out=False)

            scale = side_out_cat.pow(2).mean().sqrt().detach()
            level_loss = AE.AELoss(ae_out, side_out_cat) / (scale + 1e-6)

            print(f'loss {i} epoch {epoch}: {level_loss}')
            loss += level_loss

        loss_output.append(level_loss.data.item())
        org_loss = orthw * l2_reg_ortho(adaptors.conv)
        loss += org_loss
        loss_tot.append(loss.data.item())

        adaptors.optimizer_ANet.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adaptors.parameters(), max_norm=10.0)
        adaptors.optimizer_ANet.step()
        adaptors.save_networks(str(epoch))

        if prev_loss < loss:
            break
        else:
            prev_loss = loss

    min_index = loss_output.index(min(loss_output))
    used_comb = _tta_comb_to_string(comb_to_print)
    adaptors.load_networks(str(min_index))
    adaptors.save_networks(used_comb, config=True)

    with torch.no_grad():
        outputs = adaptors(batch, task_model, opt.model)
    real_B = task_model.real_B
    fake_B = outputs[opt.return_layers[-1]]

    visuals_output = OrderedDict()
    visuals_output['real_B'] = real_B
    visuals_output['fake_B'] = fake_B

    psnr_score = calculate_psnr(visuals_output)
    row = {
        'config': used_comb,
        'loss_output': loss_output[min_index],
        'loss_tot': loss_tot[min_index] / len(comb_to_print),
        'PSNR': psnr_score,
    }

    return row, min(loss_output)


def _tta_finalize_result(adaptors, opt, task_model, batch, return_layers, loss_config_output):
    n = len(return_layers[1:-1])
    min_loss = loss_config_output[opt.criteria].min()
    used_comb = loss_config_output[loss_config_output[opt.criteria] == min_loss]['config'].values[0]

    adaptors.load_networks(used_comb, config=True)
    chosen_comb = used_comb.split('_')
    chosen_comb = [int(x) - 1 for x in chosen_comb]
    chosen_comb = sorted(chosen_comb)
    chosen_comb = [0] + [x + 1 for x in chosen_comb] + [n + 1]
    chosen_comb = [return_layers[i] for i in chosen_comb]
    opt.return_layers = chosen_comb
    compute_tnet_dim(opt)

    with torch.no_grad():
        outputs = adaptors(batch, task_model, opt.model)
    real_B = task_model.real_B
    fake_B = outputs[opt.return_layers[-1]]
    img_path = task_model.get_image_paths()

    visuals_output = OrderedDict()
    visuals_output['real_B'] = real_B
    visuals_output['fake_B'] = fake_B

    ssim_score = calculate_ssim(visuals_output)
    mae_score = calcola_mse(visuals_output)
    psnr_score = calculate_psnr(visuals_output)

    selected_loss_output = float(loss_config_output[loss_config_output['config'] == used_comb]['loss_output'].values[0])
    _tta_cleanup_adaptor_weights(adaptors, opt, loss_config_output)

    return used_comb, ssim_score, mae_score, psnr_score, min_loss, selected_loss_output


def _tta_prepare_strategy(adaptors, opt, rec_loss):
    adaptors.set_requires_grad([adaptors.adpNet, adaptors.conv], True)
    adaptors.train()
    orthw = opt.__dict__.get('orthw', 1)
    rec_loss = round(rec_loss.item(), 4)
    loss_config_output = pd.DataFrame(columns=['config', 'loss_output', 'loss_tot', 'PSNR'])
    return orthw, rec_loss, loss_config_output


def TTA_rndm_10(adaptors, opt, task_model, save_dir, batch, rec_loss, stable=False, return_layers=None, plot=False, psnr_score_no_tta=0.0):
    n = len(return_layers[1:-1])
    indexs = [i for i in range(n)]
    tutte_combinazioni = set()
    num_random_comb = opt.__dict__.get('num_random_comb', 10)
    while len(tutte_combinazioni) < num_random_comb:
        r = random.randint(1, len(indexs))
        tutte_combinazioni.add(tuple(sorted(random.sample(indexs, r))))
    tutte_combinazioni = list(tutte_combinazioni)

    orthw, rec_loss, loss_config_output = _tta_prepare_strategy(adaptors, opt, rec_loss)

    for comb in tutte_combinazioni:
        row, candidate_loss = _tta_evaluate_combination(adaptors, opt, task_model, batch, return_layers, comb, orthw)
        loss_config_output = _tta_append_loss_row(loss_config_output, row)

    return _tta_finalize_result(adaptors, opt, task_model, batch, return_layers, loss_config_output)


def TTA_grid(adaptors, opt, task_model, save_dir, batch, rec_loss, stable=False, return_layers=None, plot=False, psnr_score_no_tta=0.0):
    n = len(return_layers[1:-1])
    indexs = [i for i in range(n)]
    tutte_combinazioni = []
    for r in range(1, len(indexs) + 1):
        tutte_combinazioni.extend(combinations(indexs, r))

    orthw, rec_loss, loss_config_output = _tta_prepare_strategy(adaptors, opt, rec_loss)

    for comb in tutte_combinazioni:
        row, candidate_loss = _tta_evaluate_combination(adaptors, opt, task_model, batch, return_layers, comb, orthw)
        loss_config_output = _tta_append_loss_row(loss_config_output, row)

    return _tta_finalize_result(adaptors, opt, task_model, batch, return_layers, loss_config_output)


def TTA_forward(adaptors, opt, task_model, save_dir, batch, rec_loss, stable=False, return_layers=None, plot=False, psnr_score_no_tta=0.0):
    n = len(return_layers[1:-1])
    indexs = [i for i in range(n)]
    selected = []
    remaining = list(indexs)
    best_loss = float('inf')
    best_comb = None
    improved = True

    orthw, rec_loss, loss_config_output = _tta_prepare_strategy(adaptors, opt, rec_loss)

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


def TTA_backward(adaptors, opt, task_model, save_dir, batch, rec_loss, stable=False, return_layers=None, plot=False, psnr_score_no_tta=0.0):
    n = len(return_layers[1:-1])
    indexs = [i for i in range(n)]
    selected = list(indexs)
    best_loss = float('inf')
    best_comb = None

    orthw, rec_loss, loss_config_output = _tta_prepare_strategy(adaptors, opt, rec_loss)

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


def TTA_bayesian(adaptors, opt, task_model, save_dir, batch, rec_loss, stable=False, return_layers=None, plot=False, psnr_score_no_tta=0.0):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n = len(return_layers[1:-1])
    indexs = [i for i in range(n)]
    orthw, rec_loss, loss_config_output = _tta_prepare_strategy(adaptors, opt, rec_loss)

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
