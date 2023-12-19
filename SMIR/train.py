import os
import glob
from pathlib import Path
import random
import ast
import numpy as np
from config import Config
import torch
from torch import nn
from torch.utils import data as Data
from tqdm import tqdm
from models.mt import MT
from torch.nn import functional as F
import matplotlib.pyplot as plt
import time
import pandas as pd
from torchvision import transforms
from tqdm import tqdm
import logging
from losses.InfoNCE import InfoNCELoss
import faiss


def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="[%(asctime)s|%(filename)s|%(levelname)s] %(message)s",
        datefmt="%a %b %d %H:%M:%S %Y",
    )

    date_string = time.strftime("%Y-%m-%d", time.localtime())
    work_dir = os.path.join(f"./log/{date_string}", time.strftime("%Y-%m-%d-%H.%M", time.localtime()))
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    fHandler = logging.FileHandler(work_dir + "/log.txt", mode="w")
    fHandler.setLevel(logging.DEBUG)
    fHandler.setFormatter(formatter)
    logger.addHandler(fHandler)

    return logger


def train(model, train_data, optimizer, scheduler, epoch, device, logger, loss_fc=None):
    model.trian()
    train_nb = len(train_data)
    loss_accumulation = loss_avg = 0
    pbar_train = tqdm(train_data, total=train_nb)

    for step, data in enumerate(pbar_train):
        data = data.to(device)

        if loss_fc.loss_name == "Tam":
            sequence, label, margin, is_anchor, padding_mask = data
            local_embedding = model(sequence, padding_mask)
            _, loss = loss_fc(local_embedding, label, margin, is_anchor)

        elif loss_fc.loss_name == "Arcface":
            sequence, label, margin, padding_mask = data
            local_embedding = model(sequence, padding_mask)
            _, loss = loss_fc(local_embedding, label, margin)

        elif loss_fc.loss_name == "InfoNCE":
            org_seq, aug_seq, org_mask, aug_mask = data
            org_embedding = model(org_seq, org_mask)
            aug_embedding = model(aug_seq, aug_mask)
            loss = loss_fc(org_embedding, aug_embedding)

        elif loss_fc.loss_name == "Triplet":
            a_seq, p_seq, n_seq, a_mask, p_mask, n_mask = data
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
            correct_num += torch.sum(thing2).item()
            if not len(mask):
                continue
            a_out, p_out, n_out = a_out[mask], p_out[mask], n_out[mask]
            loss = loss_fc(a_out, p_out, n_out)

        else:
            raise Exception("No such loss function!")

        if loss_accumulation is None and loss_avg is None:
            loss_accumulation = loss.item()
            loss_avg = loss.item()
        else:
            loss_accumulation += loss.item()
            loss_avg = loss_accumulation / (step + 1)

        loss.backward()
        optimizer.step()
        scheduler.step()
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        current_lr = scheduler.get_last_lr()[0]
        s = "train ===> epoch:{} ---- step:{} ----lr:{}---- loss:{:.4f} ---- loss_avg:{:.4f}".format(
            epoch, step, current_lr, loss, loss_avg
        )
        pbar_train.set_description(s)
        logger.info(s)

    return loss_avg

