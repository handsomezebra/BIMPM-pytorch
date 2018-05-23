import argparse
import copy
import os
import torch

from torch import nn, optim
import datetime

from model.BIMPM import BIMPM
from model.utils import SNLI, Quora
from test import test


def train(args, data):

    model = BIMPM(args, data)
    if args.gpu >= 0:
        model.cuda(args.gpu)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # saving the initial model
    dev_loss, dev_acc, dev_size = test(model, args, data, mode='dev')
    model.train()
    print(f'Init model, dev samples: {dev_size}, dev loss: {dev_loss:.3f}, dev acc: {dev_acc:.3f}')
    best_dev_loss, best_dev_acc, best_model = dev_loss, dev_acc, copy.deepcopy(model)
    
    iterator = data.train_iter
    for epoch in range(args.epoch):
        print('Start epoch:', epoch)
        iterator.init_epoch()
        num_of_batch = len(iterator)
        
        total_train_loss = 0.0
        train_size = 0
            
        for i, batch in enumerate(iterator):

            kwargs = data.get_features(batch)

            pred = model(**kwargs)

            optimizer.zero_grad()
            batch_loss = criterion(pred, batch.label)
            current_train_loss = batch_loss.item()
            total_train_loss += current_train_loss * len(pred)
            train_size += len(pred)
            batch_loss.backward()
            optimizer.step()

            if (i + 1) % args.print_freq == 0 or (i + 1) == num_of_batch:
                dev_loss, dev_acc, dev_size = test(model, args, data, mode='dev')
                model.train()   # switch back to train mode
                
                # saving the current model
                average_train_loss = total_train_loss / train_size
                print(f'Done batches: {i+1}, train samples: {train_size}, average train loss: {average_train_loss:.3f}, dev samples: {dev_size}, dev loss: {dev_loss:.3f}, dev acc: {dev_acc:.3f}')
                model_file_name = f'{args.model_dir}/BIMPM_{args.data_type}_loss_{dev_loss:.3f}_acc_{dev_acc:.3f}.pt'
                torch.save(best_model.state_dict(), model_file_name)
                
                # saving the best model
                if dev_acc > best_dev_acc:
                    best_dev_loss, best_dev_acc, best_model = dev_loss, dev_acc, copy.deepcopy(model)
            
        print(f'Done epoch: {epoch}, best dev loss: {best_dev_loss:.3f}, best dev acc: {best_dev_acc:.3f}')
        
    return best_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--char-dim', default=20, type=int)
    parser.add_argument('--char-hidden-size', default=50, type=int)
    parser.add_argument('--data-type', default='SNLI', help='available: SNLI or Quora')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--hidden-size', default=100, type=int)
    parser.add_argument('--learning-rate', default=0.001, type=float)
    parser.add_argument('--max-sent-len', default=-1, type=int,
                        help='max length of input sentences model can accept, if -1, it accepts any length')
    parser.add_argument('--num-perspective', default=20, type=int)
    parser.add_argument('--print-freq', default=500, type=int)
    parser.add_argument('--use-char-emb', default=True, action='store_true')
    parser.add_argument('--word-dim', default=300, type=int)
    args = parser.parse_args()

    if args.data_type == 'SNLI':
        print('loading SNLI data...')
        data = SNLI(args)
    elif args.data_type == 'Quora':
        print('loading Quora data...')
        data = Quora(args)
    else:
        raise NotImplementedError('only SNLI or Quora data is possible')

    setattr(args, 'char_vocab_size', len(data.char_vocab))
    setattr(args, 'word_vocab_size', len(data.TEXT.vocab))
    setattr(args, 'class_size', len(data.LABEL.vocab))
    setattr(args, 'max_word_len', data.max_word_len)
    setattr(args, 'model_dir', "saved_models_" + datetime.datetime.now().strftime("%Y%m%dT%H%M%S"))
    
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
        
    print('training start!')
    best_model = train(args, data)

    model_file_name = f'{args.model_dir}/BIMPM_{args.data_type}_best.pt'
    torch.save(best_model.state_dict(), model_file_name)
    print('training finished!')


