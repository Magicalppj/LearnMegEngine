"""Train the model"""

import argparse
import logging
import os

import megengine
import numpy as np
import megengine.optimizer as optim
from megengine.autodiff import GradManager
import megengine.distributed as dist
from tqdm import tqdm

import dist_train_utils
import utils
import model.net as net
import model.data_loader as data_loader
from dist_train_utils import is_main_process
from evaluate import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/64x64_SIGNS',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'

args = parser.parse_args()  # 设置成全局变量


def train(model, optimizer, loss_fn, dataloader, metrics, params):
    """Train the model on `num_steps` batches

    Args:
        model: (megengine.nn.Module) the neural network
        optimizer: (megengine.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a megengine.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    # gm = GradManager().attach(model.parameters())
    # 改为分布式训练
    gm = GradManager().attach(model.parameters(), callbacks=[dist.make_allreduce_cb("sum")])

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    if is_main_process():
        dataloader = tqdm(dataloader)

    for i, (train_batch, labels_batch) in enumerate(dataloader):
        # convert to meg tensors
        train_batch, labels_batch = megengine.Tensor(
            train_batch), megengine.Tensor(labels_batch)
        with gm:
            # compute model output and loss
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)

            # loss grad backward
            gm.backward(loss)

        # update model parameters and clear grad
        optimizer.step().clear_grad()

        # 分布式训练，同步各线程loss
        loss = dist_train_utils.reduce_value(loss)

        if is_main_process():
            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from megengine Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.numpy()
                labels_batch = labels_batch.numpy()

                # compute all metrics on this batch
                summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.item())

            dataloader.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            dataloader.update()

    if is_main_process():
        # compute mean of all metrics in summary
        metrics_mean = {metric: np.mean([x[metric]
                                         for x in summ]) for metric in summ[0]}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                    for k, v in metrics_mean.items())
        logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, model_dir,
                       restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (megengine.nn.Module) the neural network
        train_dataloader: (DataLoader) a megengine.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a megengine.utils.data.DataLoader object that fetches validation data
        optimizer: (megengine.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None and is_main_process():
        restore_path = os.path.join(
            args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    # 分布式训练同步初始化网络参数
    # 先进程同步再更新参数
    dist.group_barrier()
    dist.bcast_list_(model.tensors())

    best_val_acc = 0.0

    for epoch in range(params.num_epochs):
        # Run one epoch
        if is_main_process():
            logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, metrics, params)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)

        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc

        if is_main_process():
            # Save weights
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'state_dict': model.state_dict(),
                                   'optim_dict': optimizer.state_dict()},
                                  is_best=is_best,
                                  checkpoint=model_dir)

            # If best_eval, best_save_path
            if is_best:
                logging.info("- Found new best accuracy")
                best_val_acc = val_acc

                # Save best val metrics in a json file in the model directory
                best_json_path = os.path.join(
                    model_dir, "metrics_val_best_weights.json")
                utils.save_dict_to_json(val_metrics, best_json_path)

            # Save latest val metrics in a json file in the model directory
            last_json_path = os.path.join(
                model_dir, "metrics_val_last_weights.json")
            utils.save_dict_to_json(val_metrics, last_json_path)


@dist.launcher
def main():
    """
    创建一个main函数用于分布式训练
    :return:
    """

    # Load the parameters from json file
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = megengine.is_cuda_available()

    # Set the random seed for reproducible experiments
    megengine.random.seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(
        ['train', 'val'], args.data_dir, params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']

    logging.info("- done.")

    # Define the model and optimizer
    model = net.Net(params)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # fetch loss function and metrics
    loss_fn = net.loss_fn
    metrics = net.metrics

    # cuda memory optimization
    megengine.dtr.enable()

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, metrics, params, args.model_dir,
                       args.restore_file)


if __name__ == '__main__':
    main()
