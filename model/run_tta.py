import torch
from models.Autoencoder_model import AENet
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
import pandas as pd
import os
from util.visualizer import calcola_mse, calculate_psnr, calculate_ssim
from collections import OrderedDict
import numpy as np
from models.adaptor_3 import ANet
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
from tqdm import tqdm

from tta_strategies import (
    TTA_rndm_10,
    TTA_rndm_50,
    TTA_grid,
    TTA_forward,
    TTA_backward,
    TTA_bayesian,
    compute_tnet_dim,
)

TTA_STRATEGIES = {
    'rndm_10': TTA_rndm_10,
    'rndm_50': TTA_rndm_50,
    'grid': TTA_grid,
    'forward': TTA_forward,
    'backward': TTA_backward,
    'bayesian': TTA_bayesian,
}


def run_inference(task_model, AENet, adaptors, dataset, opt, save_dir, thr=0, tta_fn=TTA_rndm_50, strategy_name='rndm_50'):
    print(save_dir)
    print(f'TTA strategy: {strategy_name}')
    print(f'TTA threshold: {thr}')
    if opt.model == 'pix2pix':
        task_model.set_requires_grad([task_model.netG,task_model.netD], False)
    else:
        task_model.set_requires_grad([task_model.netG_A, task_model.netG_B, task_model.netD_A, task_model.netD_B], False)

    task_model.eval()

    return_layers = opt.return_layers

    for subnets in AENet.AENet:
        subnets.eval()

    mae, psnr, ssim_list = [], [], []
    df = pd.DataFrame(columns=['img_name', 'SSIM', 'MAE', 'PSNR'])
    df_tta = pd.DataFrame(columns=['img_name', 'SSIM', 'MAE', 'PSNR', 'config'])

    # Prepare output paths
    os.makedirs(save_dir, exist_ok=True)
    csv_path_no_tta = os.path.join(save_dir, 'metrics_no_tta.csv')
    csv_path_tta = os.path.join(save_dir, 'metrics_tta.csv')

    no_tta_columns = ['img_name', 'SSIM', 'MAE', 'PSNR']
    tta_columns = ['img_name', 'SSIM', 'MAE', 'PSNR', 'config']

    try:
        df_existing_no_tta = pd.read_csv(csv_path_no_tta) if os.path.exists(csv_path_no_tta) else pd.DataFrame(columns=no_tta_columns)
    except pd.errors.EmptyDataError:
        df_existing_no_tta = pd.DataFrame(columns=no_tta_columns)

    try:
        df_existing_tta = pd.read_csv(csv_path_tta) if os.path.exists(csv_path_tta) else pd.DataFrame(columns=tta_columns)
    except pd.errors.EmptyDataError:
        df_existing_tta = pd.DataFrame(columns=tta_columns)

    processed_img_names = set(df_existing_no_tta['img_name'].astype(str)) if 'img_name' in df_existing_no_tta else set()

    # Write headers only the first time
    write_header_no_tta = not os.path.exists(csv_path_no_tta)
    write_header_tta = not os.path.exists(csv_path_tta)

    for idx, data in tqdm(enumerate(dataset)):
        task_model.set_input(data)
        img_path = task_model.get_image_paths()
        img_name = img_path[0]

        if img_name in processed_img_names:
            opt.return_layers = return_layers
            continue

        # These lines replace BaseModel.test(), which we do not call here.
        with torch.no_grad():
            outputs = task_model.forward(return_layers=opt.return_layers)

        task_model.compute_visuals()
        visuals = task_model.get_current_visuals()

        mae_score = calcola_mse(visuals)
        mae.append(mae_score)
        psnr_score = calculate_psnr(visuals)
        psnr.append(psnr_score)
        ssim_score = calculate_ssim(visuals)
        ssim_list.append(ssim_score)
        
        index = opt.return_layers[-1]
        side_out = outputs[index]
            
        # Use separate features
        side_out = side_out
        ae_out = AENet.AENet[-1](side_out, side_out=False)
        rec_loss = AENet.AELoss(ae_out, side_out)

        row = {
            'img_name': img_path[0],
            'SSIM': np.round(ssim_score, 4),
            'MAE': np.round(mae_score, 4),
            'PSNR': np.round(psnr_score, 4),
        }
        pd.DataFrame([row]).to_csv(
            csv_path_no_tta,
            mode='a',
            header=write_header_no_tta,
            index=False
        )
        write_header_no_tta = False
        processed_img_names.add(img_name)
        if np.round(rec_loss.item(),4) > thr:
            used_comb, ssim_score, mae_score, psnr_score, min_loss, _ = tta_fn(
                adaptors,
                opt,
                task_model,
                save_dir,
                data,
                rec_loss,
                return_layers=opt.return_layers,
                psnr_score_no_tta=psnr_score,
            )

            
            row_tta = {
                'img_name': img_path[0],
                'SSIM': np.round(ssim_score, 4),
                'MAE': np.round(mae_score, 4),
                'PSNR': np.round(psnr_score, 4),
                'config': used_comb
            }
            pd.DataFrame([row_tta]).to_csv(
                csv_path_tta,
                mode='a',
                header=write_header_tta,
                index=False
            )
            write_header_tta = False

        opt.return_layers = return_layers

    # Compute summary statistics using saved CSVs
    summary = {}

    df_no_tta = pd.read_csv(csv_path_no_tta) if os.path.exists(csv_path_no_tta) else pd.DataFrame()
    df_tta = pd.read_csv(csv_path_tta) if os.path.exists(csv_path_tta) else pd.DataFrame()

    if not df_no_tta.empty:
        summary.update({
            'total_samples': len(df_no_tta),
            'no_tta_mae_mean': float(df_no_tta['MAE'].mean()),
            'no_tta_psnr_mean': float(df_no_tta['PSNR'].mean()),
            'no_tta_ssim_mean': float(df_no_tta['SSIM'].mean())
        })

    if not df_tta.empty:
        summary.update({
            'tta_samples': len(df_tta),
            'tta_mae_mean': float(df_tta['MAE'].mean()),
            'tta_psnr_mean': float(df_tta['PSNR'].mean()),
            'tta_ssim_mean': float(df_tta['SSIM'].mean())
        })

        triggered_subset = df_no_tta[df_no_tta['img_name'].isin(df_tta['img_name'])]
        if not triggered_subset.empty:
            summary.update({
                'no_tta_triggered_mae_mean': float(triggered_subset['MAE'].mean()),
                'no_tta_triggered_psnr_mean': float(triggered_subset['PSNR'].mean()),
                'no_tta_triggered_ssim_mean': float(triggered_subset['SSIM'].mean())
            })
    else:
        df_tta = pd.DataFrame(columns=tta_columns)

    summary_path = os.path.join(save_dir, 'summary.json')
    summary.update({
        'tta_strategy': strategy_name,
        'threshold': float(thr),
    })
    with open(summary_path, 'w') as f:
        import json
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    opt = TrainOptions().parse()
    return_layers = opt.return_layers
    compute_tnet_dim(opt)
    task_model = create_model(opt)

    dataset = create_dataset(opt)

    task_model.setup(opt)
    task_checkpoint_path = os.environ.get('TASK_CHECKPOINT_PATH', opt.task_checkpoint_path)
    if not task_checkpoint_path:
        raise ValueError("Task model checkpoint path is required. Set --task_checkpoint_path or TASK_CHECKPOINT_PATH.")

    state_dict_model = torch.load(task_checkpoint_path, map_location=str(task_model.device))

    if opt.model == 'pix2pix':
        if isinstance(task_model.netG, torch.nn.DataParallel):
            task_model.netG.module.load_state_dict(state_dict_model)
        else:
            task_model.netG.load_state_dict(state_dict_model)
    else:
        if isinstance(task_model.netG_A, torch.nn.DataParallel):
            task_model.netG_A.module.load_state_dict(state_dict_model)
        else:
            task_model.netG_A.load_state_dict(state_dict_model)

    AENet = AENet(opt)

    ae_checkpoint_dir = os.environ.get('AE_CHECKPOINT_DIR', opt.ae_checkpoint_dir)
    ae_epoch = os.environ.get('AE_EPOCH', opt.ae_epoch)
    if not ae_checkpoint_dir:
        raise ValueError("AE checkpoint directory is required. Set --ae_checkpoint_dir or AE_CHECKPOINT_DIR.")

    for i in range(len(opt.return_layers)):
        name = opt.return_layers[i]
        load_path_weights = os.path.join(ae_checkpoint_dir, f'AE_{name}_{ae_epoch}.pt')

        state_dict = torch.load(load_path_weights, map_location=str(
            AENet.device))

        AENet.AENet[i].load_state_dict(state_dict)
        AENet.set_requires_grad(AENet.AENet[i], False)
    
    adaptors = ANet(opt).cuda()

    strategy_name = os.environ.get('TTA_STRATEGY', opt.tta_strategy)
    if strategy_name not in TTA_STRATEGIES:
        raise ValueError(f'Unknown TTA_STRATEGY={strategy_name}. Available: {sorted(TTA_STRATEGIES)}')
    tta_fn = TTA_STRATEGIES[strategy_name]

    thr = float(os.environ.get('TTA_THRESHOLD', opt.tta_threshold))
    thr_tag = str(thr).replace('.', 'p').replace('-', 'm')

    output_dir = os.path.join(opt.results_dir, f'sample_aware_tta_{strategy_name}_thr_{thr_tag}')
    adaptors.save_dir = output_dir
    adaptors.save_dir_config = output_dir
    run_inference(task_model, AENet, adaptors, dataset, opt, output_dir, thr, tta_fn, strategy_name)
