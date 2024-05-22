import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
from datasets import tqdm

from model_list import get_models, get_datasets
from utils.hparams import HParam
from utils.writer import MyWriter


def train(model, dataset, writer, hp_str, chkpt_path=None):
    torch.multiprocessing.set_sharing_strategy('file_system')

    prev_epoch = 1

    train_model = model.model
    if chkpt_path is not None:
        loaded = torch.load(chkpt_path)
        chkpt_model = loaded['model']
        train_model.load_state_dict(chkpt_model)
        logger.info('using checkpoint model for train: {}'.format(chkpt_path))

        prev_epoch = loaded['epoch'] + 1

    num_train_data = len(dataset.trainloader)
    num_validation_data = len(dataset.validationloader)
    optimizer = model.get_optimizer()
    for epoch in range(prev_epoch, args.epoch + 1):
        # train
        train_model.train()
        train_progress_bar = tqdm(range(num_train_data), ncols=100)
        train_losses = []
        for train_data in dataset.trainloader:
            output = model.train(train_data)
            train_loss = model.get_loss(train_data, output)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_losses.append(train_loss.item())

            train_progress_bar.update(1)
        train_progress_bar.close()

        # validate
        train_model.eval()
        validation_progress_bar = tqdm(range(num_validation_data), ncols=100)
        test_losses = []
        with torch.no_grad():
            for validate_data in dataset.validationloader:
                output = model.validate(validate_data)
                test_losses.extend(output)

                validation_progress_bar.update(1)
        validation_progress_bar.close()

        train_loss = np.mean(train_losses)

        # log
        writer.log_training(train_loss, epoch)
        test_loss = model.write_tensorboard(writer, test_losses, epoch)

        logger.info('[Epoch {}] train_loss: {}, validation_loss: {}\n'.format(epoch, train_loss, test_loss))

        # 1. save checkpoint file to resume training
        # 2. evaluate and save sample to tensorboard
        if epoch % hp.train.checkpoint_interval == 0:
            save_path = os.path.join(pt_dir, 'chkpt_%d.pt' % epoch)
            torch.save({
                'model': train_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'hp_str': hp_str,
            }, save_path)
            logger.info("Saved checkpoint to: %s\n" % save_path)


def inference(model, dataset, writer, hp_str, output_dir, chkpt_path=None):
    train_model = model.model

    if chkpt_path is not None:
        chkpt_model = torch.load(chkpt_path)['model']
        train_model.load_state_dict(chkpt_model)
        logger.info('using checkpoint model for inference: {}'.format(chkpt_path))
    train_model.eval()

    num_inference_data = len(dataset.inferenceloader)

    logger.info('performing inference...\n')
    inference_progress_bar = tqdm(range(num_inference_data))
    with torch.no_grad():
        for inference_data in dataset.inferenceloader:
            model.inference(inference_data, output_dir)
            inference_progress_bar.update(1)
    inference_progress_bar.close()


def test(model, dataset, writer, hp_str, output_dir, chkpt_path):
    if chkpt_path is None:
        raise Exception("Checkpoint is needed when doing test. Please specify proper checkpoint path")

    train_model = model.model
    chkpt_model = torch.load(chkpt_path)['model']
    train_model.load_state_dict(chkpt_model)
    train_model.eval()

    with torch.no_grad():
        with open(os.path.join(output_dir, 'avg.txt'), 'at') as file:
            file.write('{}\n'.format(model.get_model_name()))

        for test_class in model.tests:
            test_name = test_class.get_test_name()
            logger.info('testing {}...\n'.format(test_name))

            num_test_data = len(dataset.testloader)
            test_progress_bar = tqdm(range(num_test_data))
            results = []
            for test_data in dataset.testloader:
                result = test_class.test(test_data)
                results.append(result)
                test_progress_bar.update(1)
            test_progress_bar.close()

            avg = float(np.mean(results))

            with open(os.path.join(output_dir, '{}.txt'.format(test_name)), 'wt') as file:
                file.write(str(results) + '\n')
            with open(os.path.join(output_dir, 'avg.txt'), 'at') as file:
                file.write('{}: {}\n'.format(test_name, avg))

            logger.info('average {}: {}'.format(test_name, avg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_dir', type=str, default='.',
                        help="Root directory of run.")
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-p', '--checkpoint_path_train', type=str, default=None,
                        help="train pt file")
    parser.add_argument('-pp', '--checkpoint_path_inference', type=str, default=None,
                        help="inference pt file")
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="type of the model to use.")
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help="type of the dataset to use.")
    parser.add_argument('-e', '--epoch', type=int, required=True,
                        help="epoch")
    parser.add_argument('-o', '--output_data_dir', type=str, default=None,
                        help='path to store enhanced voice.')
    parser.add_argument('--train', type=int, default=1,
                        help='specify if run train.')
    parser.add_argument('--inference', type=int, default=0,
                        help='specify if run inference. 0=not run inference. 1=run inference, 2=run inference and test')
    args = parser.parse_args()

    train_mode = True if int(args.train) == 1 else False
    inference_mode = int(args.inference)

    hp_path = os.path.join(args.base_dir, args.config)
    hp = HParam(hp_path)
    with open(args.config, 'rt') as f:
        # store hparams as string
        hp_str = ''.join(f.readlines())

    pt_dir = os.path.join(args.base_dir, hp.log.chkpt_dir, args.model)
    os.makedirs(pt_dir, exist_ok=True)

    log_dir = os.path.join(args.base_dir, hp.log.log_dir, args.model)
    os.makedirs(log_dir, exist_ok=True)

    output_dir = args.output_data_dir
    chkpt_path_train = args.checkpoint_path_train
    chkpt_path_inference = args.checkpoint_path_inference

    model_name = args.model
    models = get_models(hp)
    if model_name not in models:
        raise Exception("Please check your model name %s \n"
                        "Choice within [%s]" % (model_name, ', '.join(models)))
    model = models[model_name]()

    dataset_name = args.dataset
    datasets = get_datasets(hp, model_name)
    if dataset_name not in datasets:
        raise Exception("Please check your dataset name %s \n"
                        "Choice within [%s]" % (dataset_name, ', '.join(datasets)))
    dataset = datasets[dataset_name]()

    writer = MyWriter(hp, log_dir)

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
        # handlers=[
        #     logging.FileHandler(os.path.join(log_dir, '%s-%d.log' % (model_name, time.time()))),
        # ]
    )
    logger = logging.getLogger("Trainer")

    logger.info('train: {}, inference: {}'.format(train_mode, inference_mode))
    logger.info('loaded model: {}, dataset: {}\n'.format(args.model, args.dataset))

    if train_mode:
        logger.info('start train...')
        train(model, dataset, writer, hp_str, chkpt_path_train)

    if inference_mode:
        logger.info('start inference...')
        os.makedirs(output_dir, exist_ok=True)
        inference(model, dataset, writer, hp_str, output_dir, chkpt_path_inference)

        logger.info('start test...')

        if inference_mode == 2:
            test(model, dataset, writer, hp_str, output_dir, chkpt_path_inference)
