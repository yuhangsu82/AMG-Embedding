import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils import data as Data
from tqdm import tqdm
from models import MT, MCNN, MLSTM
from torch.nn import functional as F
import matplotlib.pyplot as plt
import time
import pandas as pd
from torchvision import transforms
from tqdm import tqdm
import logging
from losses import InfoNCELoss, TripletLoss, ArcFaceLoss, TamLoss
from dataset import SpectrogramFingerprintData
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
from evaluate import eval


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # project's root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def init_model(model_type, feature_dim, is_residual, device):
    if model_type == 'mt':
        model = MT(128, feature_dim, 256, 10, 8, 0.1, is_residual).to(device)
    elif model_type == 'mlstm':
        model = MLSTM(128, feature_dim, 256, 8, 0.1, is_residual).to(device)
    elif model_type == 'mcnn':
        model = MCNN(feature_dim, is_residual).to(device)
    else:
        raise Exception('No such model! (Tip: You can fix the code and add your own model in this project.)')

    return model


def init_loss(loss_name, feature_dim, class_nums):
    if loss_name == 'Tam':
        loss_fc = TamLoss(in_features=feature_dim, out_features=class_nums, s=32.0).to(opt.device)
    elif loss_name == 'Arcface':
        loss_fc = ArcFaceLoss(in_features=feature_dim, out_features=class_nums, s=32.0).to(opt.device)
    elif loss_name == 'InfoNCE':
        loss_fc = InfoNCELoss().to(opt.device)
    elif loss_name == 'Triplet':
        loss_fc = TripletLoss(alpha=0.2).to(opt.device)
    else:
        raise Exception('No such loss function!')

    return loss_fc


def load_file(input_file_path, class_nums=0, sample_nums=0):
    files = list()
    for filename in os.listdir(input_file_path):
        if (int(filename.split('-')[0]) <= class_nums and int(filename.split('-')[1][:-4]) < sample_nums):
            files.append(filename)
    return files


def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt='[%(asctime)s|%(filename)s|%(levelname)s] %(message)s',
        datefmt='%a %b %d %H:%M:%S %Y',
    )

    date_string = time.strftime('%Y-%m-%d', time.localtime())
    work_dir = os.path.join(
        f'./logs/{date_string}', time.strftime('%Y-%m-%d-%H.%M', time.localtime())
    )
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    fHandler = logging.FileHandler(work_dir + '/log.txt', mode='w')
    fHandler.setLevel(logging.DEBUG)
    fHandler.setFormatter(formatter)
    logger.addHandler(fHandler)

    return logger


def train(model, train_data, optimizer, scheduler, epoch, device, logger, loss_fc=None):
    model.train()
    train_nb = len(train_data)
    loss_accumulation = loss_avg = 0
    pbar_train = tqdm(train_data, total=train_nb)

    for step, data in enumerate(pbar_train):
        if loss_fc.loss_name == 'Tam':
            sequence, label, margin, is_anchor, padding_mask = data
            sequence = sequence.to(device)
            label = label.to(device)
            margin = margin.to(device)
            is_anchor = is_anchor.to(device)
            padding_mask = padding_mask.to(device)

            local_embedding = model(sequence, padding_mask)
            _, loss = loss_fc(local_embedding, label, margin, is_anchor)

        elif loss_fc.loss_name == 'Arcface':
            sequence, label, margin, padding_mask = data
            sequence = sequence.to(device)
            label = label.to(device)
            margin = margin.to(device)
            padding_mask = padding_mask.to(device)

            local_embedding = model(sequence, padding_mask)
            _, loss = loss_fc(local_embedding, label, margin)

        elif loss_fc.loss_name == 'InfoNCE':
            org_seq, aug_seq, org_mask, aug_mask = data
            org_seq = org_seq.to(device)
            aug_seq = aug_seq.to(device)
            org_mask = org_mask.to(device)
            aug_mask = aug_mask.to(device)

            org_embedding = model(org_seq, org_mask)
            aug_embedding = model(aug_seq, aug_mask)
            loss = loss_fc(org_embedding, aug_embedding)

        elif loss_fc.loss_name == 'Triplet':
            a_seq, p_seq, n_seq, a_mask, p_mask, n_mask = data
            a_seq = a_seq.to(device)
            p_seq = p_seq.to(device)
            n_seq = n_seq.to(device)
            a_mask = a_mask.to(device)
            p_mask = p_mask.to(device)
            n_mask = n_mask.to(device)

            a_out, p_out, n_out = (
                model(a_seq, a_mask),
                model(p_seq, p_mask),
                model(n_seq, n_mask),
            )
            s_d = F.cosine_similarity(a_out, p_out)
            n_d = F.cosine_similarity(a_out, n_out)
            thing1 = (s_d - n_d < loss_fc.alpha).flatten().cpu()
            thing2 = (s_d - n_d >= loss_fc.alpha).flatten().cpu()
            mask = np.where(thing1.numpy() == 1)[0]
            # correct_num += torch.sum(thing2).item()
            if not len(mask):
                continue
            a_out, p_out, n_out = a_out[mask], p_out[mask], n_out[mask]
            loss = loss_fc(a_out, p_out, n_out)

        else:
            raise Exception('No such loss function!')

        if loss_accumulation is None and loss_avg is None:
            loss_accumulation = loss.item()
            loss_avg = loss.item()
        else:
            loss_accumulation += loss.item()
            loss_avg = loss_accumulation / (step + 1)

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        current_lr = scheduler.get_last_lr()[0]
        s = 'train ===> epoch:{} ---- step:{} ----lr:{}---- loss:{:.4f} ---- loss_avg:{:.4f}'.format(
            epoch, step, current_lr, loss, loss_avg
        )
        pbar_train.set_description(s)
        logger.info(s)

    return loss_avg


def init_class_center(loss):
    center = np.zeros((opt.class_nums, opt.feature_dim), dtype=np.float32)
    for i in range(1, opt.class_nums + 1):
        data = np.load(os.path.join(opt.dataset_dir, f"{i}-00.npy"))
        center[i-1] = np.mean(data, axis=0)
    t_c = torch.from_numpy(center).float().cuda()
    with torch.no_grad():
        loss.weight.data = t_c


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default=ROOT / 'datasets/mt/train/data-10w')
    parser.add_argument('--test_dir', type=str, default=ROOT / 'runs/retrieval/test')
    parser.add_argument('--test_dummy_dir', type=str, default=ROOT / 'database/fma_part_30s')
    parser.add_argument('--model', type=str, default='mt')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--is_residual', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--last_lr', type=float, default=0.00001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=320)
    parser.add_argument('--class_nums', type=int, default=20000)
    parser.add_argument('--sample_nums', type=int, default=10)  
    parser.add_argument('--max_len', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--start_id', type=int, default=0)
    parser.add_argument('--loss', type=str, default='Tam')
    parser.add_argument('--feature_dim', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    if torch.cuda.is_available() and 'cuda' in opt.device:
        device = torch.device(opt.device)
    else:
        device = torch.device('cpu')

    model = init_model(opt.model, opt.feature_dim, opt.is_residual, opt.device)
    logger = get_logger()
    ckpt_dir = f'./runs/checkpoint/{opt.model}_{opt.loss.lower()}/exp'
    if os.path.exists(ckpt_dir) is False:
        os.makedirs(ckpt_dir)
    
    ckpt_file_path = os.path.join(ckpt_dir, 'last.pth')
    train_files = load_file(opt.dataset_dir, opt.class_nums, opt.sample_nums)

    train_data = SpectrogramFingerprintData(
        files=train_files,
        root_dir=opt.dataset_dir,
        max_len=opt.max_len,
        seq_len=opt.seq_len,
        start_id=opt.start_id,
        feature_dim=opt.feature_dim,
        class_nums=opt.class_nums,
        mode='train-' + opt.loss.lower(),
    )

    train_data = Data.DataLoader(train_data, shuffle=True, batch_size=opt.batch_size)
    loss_fc = init_loss(opt.loss, opt.feature_dim, opt.class_nums).to(opt.device)
    optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': loss_fc.parameters()}], lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=opt.epochs * len(train_data), eta_min=opt.last_lr)

    if os.path.exists(ckpt_file_path):
        checkpoint = torch.load(ckpt_file_path)
        model.load_state_dict(checkpoint['model'])
        loss_fc.load_state_dict(checkpoint['loss'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        print("Loaded epoch {} successfully!".format(start_epoch))
    else:
        start_epoch = 0
        print("Train the model from scratch without saving previous model!")
        df_train = pd.DataFrame(columns=['time', 'epoch', 'train/loss'])
        df_train.to_csv(os.path.join(ckpt_dir, 'result.csv'), index=False)
        with open(os.path.join(ckpt_dir, 'config.yaml'), 'w') as file:
            yaml.dump(vars(opt), file, default_flow_style=False)
        start_epoch = 0

    for epoch in range(start_epoch + 1, opt.epochs + 1):
        if epoch == 1 and (opt.loss == 'Tam' or opt.loss == 'Arcface'):
            init_class_center(loss_fc)

        train_loss = train(
            model=model,
            train_data=train_data,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            device=device,
            logger=logger,
            loss_fc=loss_fc,
        )

        time_now = time.strftime('%Y-%m-%d-%H.%M', time.localtime())
        data_list = [time_now, epoch, train_loss]
        data = pd.DataFrame([data_list])
        data.to_csv(
            os.path.join(ckpt_dir, 'result.csv'),
            mode='a',
            header=False,
            index=False,
        )

        state = {
            'model': model.state_dict(),
            'loss': loss_fc.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
        }
        if epoch % 10 == 0:
            print('epoch: ', epoch)
            table_string = eval(
                emb_dir=opt.test_dir,
                emb_dummy_dir=opt.test_dummy_dir,
                model_type=opt.model,
                is_residual=opt.is_residual,
                checkpoint_path=ckpt_file_path,
                max_len=opt.max_len,
                device='cuda:0',
                batch_size=1500,
                feature_dim=128,
                k_prob=10,
            )
            with open(os.path.join(ckpt_dir, 'table.txt'), 'a') as file:
                file.write(f'epoch: {epoch}\n')
                file.write(table_string + '\n')

            torch.save(state, os.path.join(ckpt_dir, f'{opt.model}_{epoch}.pth').format(epoch))

        torch.save(state, ckpt_file_path)
        # torch.save(state, os.path.join(ckpt_dir, f'{opt.model}_{epoch}.pth').format(epoch))

