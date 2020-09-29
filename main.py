### Comments with 1 # are from original authors
### Comments with 3 # are from me 

import os
import json
import time
import utils
import models
import logger
import inspect
import datetime
import argparse
import data_loader

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable

import numpy as np

parser = argparse.ArgumentParser()
# basic args
parser.add_argument('--task', type = str)
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--epochs', type = int, default = 100)

# evaluation args
parser.add_argument('--weight_file', type = str)
parser.add_argument('--result_file', type = str)

# cnn args
parser.add_argument('--kernel_size', type = int)

# rnn args
parser.add_argument('--pooling_method', type = str)

# multi-task args
parser.add_argument('--alpha', type = float)

# log file name
parser.add_argument('--log_file', type = str)

args = parser.parse_args()

config = json.load(open('./config.json', 'r'))


def train(model, elogger, train_set, eval_set):
    # record the experiment setting
    elogger.log(str(model))
    elogger.log(str(args._get_kwargs()))

    if torch.cuda.is_available():
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr = 1e-3)

    for epoch in range(args.epochs):
        print(f'Training on epoch {epoch}')
        model.train()

        for input_file in train_set:   ### Train set contains file names of training data
            print(f'Train on file {input_file}')

            # data loader, return two dictionaries, attr and traj
            data_iter = data_loader.get_loader(input_file, args.batch_size)

            running_loss = 0.0

            for idx, (attr, traj) in enumerate(data_iter):
                # transform the input to pytorch variable
                ### Originally attr and traj are dict of torch tensors on CPU 
                ### to_var function converts them into dict of CUDA tensors
                ### Each key of attr is size [batch, ]
                ### Each key of traj except lens is size [batch, max(lens)], lens is [batch, ]
                ### Each element is batch dimension is a single trip
                attr, traj = utils.to_var(attr), utils.to_var(traj)

                _, loss = model.eval_on_batch(attr, traj, config)

                # update the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.data
                print(f'Progress {round((idx + 1) * 100.0 / len(data_iter), 2)}%, average loss {running_loss / (idx + 1.0)}')
            elogger.log(f'Training Epoch {epoch}, File {input_file}, Loss {running_loss / (idx + 1.0)}')
        # evaluate the model after each epoch
        evaluate(model, elogger, eval_set, save_result = False)

        # save the weight file after each epoch
        ### EDIT: Save after every 10 epochs
        ###       Save weight of final epoch as "weights"
        save_flag = False
        if (epoch + 1) == args.epoch:
            weight_name = 'weights'
            save_flag = True
        elif (epoch + 1) % 10 == 0:
            weight_name = '{}_{}'.format(args.log_file, str(datetime.datetime.now()))
            save_flag = True
        
        if save_flag:
            elogger.log(f'Epoch {epoch}, save weight file {weight_name}')
            torch.save(model.state_dict(), './saved_weights/' + weight_name)


def write_result(fs, pred_dict, attr):
    pred = pred_dict['pred'].data.cpu().numpy()
    label = pred_dict['label'].data.cpu().numpy()

    for i in range(pred_dict['pred'].size()[0]):
        fs.write('%.6f %.6f\n' % (label[i][0], pred[i][0]))

        dateID = attr['dateID'].data[i]
        timeID = attr['timeID'].data[i]
        driverID = attr['driverID'].data[i]


def evaluate(model, elogger, files, save_result = False):
    model.eval()
    if save_result:
        fs = open('%s' % args.result_file, 'w')

    for input_file in files:
        running_loss = 0.0
        data_iter = data_loader.get_loader(input_file, args.batch_size)

        for idx, (attr, traj) in enumerate(data_iter):
            attr, traj = utils.to_var(attr), utils.to_var(traj)

            pred_dict, loss = model.eval_on_batch(attr, traj, config)

            if save_result: 
                write_result(fs, pred_dict, attr)

            running_loss += loss.data

        print(f'Evaluate on file {input_file}, loss {running_loss / (idx + 1.0)}')
        elogger.log('Evaluate File {}, Loss {}'.format(input_file, running_loss / (idx + 1.0)))

    if save_result: 
        fs.close()


def get_kwargs(model_class):
    model_args = inspect.getfullargspec(model_class.__init__).args
    shell_args = args._get_kwargs()

    kwargs = dict(shell_args)

    for arg, val in shell_args:
        if not arg in model_args:
            kwargs.pop(arg)

    return kwargs

def run():
    # get the model arguments
    kwargs = get_kwargs(models.DeepTTE.Net)

    # model instance
    model = models.DeepTTE.Net(**kwargs)

    # experiment logger
    elogger = logger.Logger(args.log_file)

    if args.task == 'train':
        train(model, elogger, train_set = config['train_set'], eval_set = config['eval_set'])

    elif args.task == 'test':
        # load the saved weight file
        model.load_state_dict(torch.load(args.weight_file))
        if torch.cuda.is_available():
            model.cuda()
        evaluate(model, elogger, config['test_set'], save_result = True)


if __name__ == '__main__':
    run()