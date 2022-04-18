# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from tensorboardX import SummaryWriter
import torch.nn.parallel
from utils.utils import distributed_all_gather
from utils.data_utils import yusongli_save_pred
import torch.utils.data.distributed
from monai.data import decollate_batch
from typing import Callable, Optional
from numpy.typing import ArrayLike
import argparse
import pathlib
from monai.data import write_nifti
from tqdm.auto import tqdm


def dice(x: ArrayLike, y: ArrayLike):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


class AverageMeter(object):
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n: Optional[int] = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args) -> float:
    model.train()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(tqdm(loader, leave=False, dynamic_ncols=True, desc='Train')):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data['image'], batch_data['label']
        data, target = data.cuda(args.gpu), target.cuda(args.gpu)
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            logits = model(data)
            loss = loss_func(logits, target)
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)

    for param in model.parameters():
        param.grad = None
    return run_loss.avg


def val_epoch(
    model, loader, epoch, acc_func, args, model_inferer=None, post_label=None, post_pred=None, post_invert=None
) -> float:
    model.eval()
    with torch.no_grad():
        for idx, batch_data in enumerate(tqdm(loader, leave=False, dynamic_ncols=True, desc='Val')):
            if isinstance(batch_data, list):
                data, target = batch_data
                yaml_meta_data = None
            else:
                data, target = batch_data['image'], batch_data['label']
                yaml_meta_data = batch_data['label_meta_dict']['yaml_meta_data']
            data, target = data.cuda(args.gpu), target.cuda(args.gpu)
            with autocast(enabled=args.amp):
                batch_data['pred'] = model_inferer(data) if model_inferer is not None else model(data)
                logits = batch_data['pred']
            if not logits.is_cuda:
                target = target.cpu()
            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_outputs_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            acc = acc_func(y_pred=val_outputs_convert, y=val_labels_convert)
            acc = acc.cuda(args.gpu)

            if args.distributed:
                acc_list = distributed_all_gather([acc], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            else:
                acc_list = acc.detach().cpu().numpy()
            avg_acc = np.mean([np.nanmean(l) for l in acc_list])

            batch_data = [post_invert(i) for i in decollate_batch(batch_data)]
            pred_array = batch_data[0]['pred'].detach().cpu().max(axis=0, keepdim=False)[1].numpy()
            pred_savepath = pathlib.Path(yusongli_save_pred(yaml_meta_data, epoch, args.logdir))
            pred_savepath.parent.mkdir(parents=True, exist_ok=True)
            write_nifti(pred_array, pred_savepath)

    return avg_acc


def save_checkpoint(model, epoch, args, filename='model.pt', best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.module.state_dict() if args.distributed else model.state_dict()
    save_dict = {'epoch': epoch, 'best_acc': best_acc, 'state_dict': state_dict}
    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print('Saving checkpoint', filename)


def run_training(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_func: torch.nn.Module,
    acc_func: Callable,
    args: argparse.ArgumentParser,
    model_inferer: Optional[Callable] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    start_epoch: Optional[int] = 0,
    post_label: Optional[object] = None,
    post_pred: Optional[object] = None,
    post_invert: Optional[object] = None,
) -> float:
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        print('Writing Tensorboard logs to ', args.logdir)
    scaler = GradScaler() if args.amp else None
    val_acc_max = 0.0
    epoch_max = 0
    for epoch in tqdm(range(start_epoch, args.max_epochs), leave=False, dynamic_ncols=True, desc='Epoch'):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler=scaler,
            epoch=epoch,
            loss_func=loss_func,
            args=args,
        )
        if scheduler is not None:
            scheduler.step()

        if args.rank == 0 and writer is not None:
            writer.add_scalar('train_loss', train_loss, epoch)
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            val_avg_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
                post_invert=post_invert,
            )
            if args.rank == 0:
                writer.add_scalar('val_acc', val_avg_acc, epoch)
                if val_avg_acc > val_acc_max:
                    epoch_max = epoch
                    print(f'new best ({val_acc_max:.6f} --> {val_avg_acc:.6f}) on epoch {epoch_max}')
                    val_acc_max = val_avg_acc
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        ckpt_name = f'epoch={epoch}-best_dice={val_acc_max:.4f}.pth'
                        save_checkpoint(
                            model,
                            epoch,
                            args,
                            filename=ckpt_name,
                            best_acc=val_acc_max,
                            optimizer=optimizer,
                            scheduler=scheduler,
                        )

    print(f'Model {args.model_name} training Finished ! Best Accuracy:  {val_acc_max} at epoch {epoch_max}')

    return val_acc_max
