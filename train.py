import argparse
import copy
import os
import torch

from torch import nn, optim
import datetime

from model.BIMPM import BIMPM
from model.utils import SNLI, Quora
from test import test


def train(model, model_dir, args, data):

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=args.learning_rate)
    criterion = nn.MSELoss()

    # saving the initial model
    dev_loss, dev_acc, dev_size = test(model, args, data, mode='dev')
    model.train()
    print(f'Init model, dev samples: {dev_size}, dev loss: {dev_loss:.3f}, dev acc: {dev_acc:.3f}')
    best_dev_loss, best_dev_acc = dev_loss, dev_acc
    
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
            batch_loss = criterion(pred, batch.label.float())
            current_train_loss = batch_loss.item()
            total_train_loss += current_train_loss * len(pred)
            train_size += len(pred)
            batch_loss.backward()
            optimizer.step()

            if (i + 1) % args.print_freq == 0 or (i + 1) == num_of_batch:
                dev_loss, dev_acc, dev_size = test(model, args, data, mode='dev')
                model.train()   # switch back to train mode
                
                average_train_loss = total_train_loss / train_size
                print(f'Done batches: {i+1}, train samples: {train_size}, average train loss: {average_train_loss:.3f}, dev samples: {dev_size}, dev loss: {dev_loss:.3f}, dev acc: {dev_acc:.3f}')
                
                # saving the best model
                if dev_acc > best_dev_acc:
                    best_dev_loss, best_dev_acc = dev_loss, dev_acc
                    model_file_name = f'{model_dir}/BIMPM_{args.data_type}_best.pt'
                    torch.save(model, model_file_name)
            
        print(f'Done epoch: {epoch}, best dev loss: {best_dev_loss:.3f}, best dev acc: {best_dev_acc:.3f}')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=40, width=120))
    parser.add_argument('--data-type', default='Quora', 
                        help='data type, available: SNLI or Quora')
    parser.add_argument('--batch-size', default=48, type=int,
                        help='batch size')
    parser.add_argument('--epoch', default=20, type=int,
                        help='number of epoch')
    parser.add_argument('--gpu', default=0, type=int,
                        help='gpu id, -1 to use cpu')
    parser.add_argument('--print-freq', default=500, type=int,
                        help='number of batches to evaluate and checkpoint')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='dropout rate')
    parser.add_argument('--learning-rate', default=0.0005, type=float,
                        help='learning rate')
    parser.add_argument('--max-sent-len', default=-1, type=int,
                        help='max number of words per sentence, if -1, it accepts any length')
    parser.add_argument('--max-word-len', default=-1, type=int,
                        help='max number of chars per word, if -1, it accepts any length')
    parser.add_argument('--num-perspective', default=20, type=int,
                        help='number of perspective')
    parser.add_argument('--word-dim', default=300, type=int,
                        help='word embedding dimension')
    parser.add_argument('--char-dim', default=20, type=int,
                        help='character embedding dimension')
    parser.add_argument('--char-lstm-dim', default=100, type=int,
                        help='character LSTM\'s hidden size')
    parser.add_argument('--context-lstm-dim', default=100, type=int,
                        help='context LSTM\'s hidden size')
    parser.add_argument('--context-layer-num', default=2, type=int,
                        help='context LSTM\'s layer number')
    parser.add_argument('--aggregation-lstm-dim', default=100, type=int,
                        help='aggregation LSTM\'s hidden size')
    parser.add_argument('--aggregation-layer-num', default=2, type=int,
                        help='aggregation LSTM\'s layer number')
    parser.add_argument('--wo-char', default=False, action='store_true',
                        help='whether to learn without character features')
    parser.add_argument('--wo-full-match', default=False, action='store_true',
                        help='whether to learn without full match')
    parser.add_argument('--wo-maxpool-match', default=False, action='store_true',
                        help='whether to learn without max pool match')
    parser.add_argument('--wo-attentive-match', default=False, action='store_true',
                        help='whether to learn without attentive match')
    parser.add_argument('--wo-max-attentive-match', default=False, action='store_true',
                        help='whether to learn without max attentive match')
    args = parser.parse_args()

    if args.data_type == 'SNLI':
        print('Loading SNLI data...')
        data = SNLI(args)
    elif args.data_type == 'Quora':
        print('Loading Quora data...')
        data = Quora(args)
    else:
        raise NotImplementedError('only SNLI or Quora data is possible')
        
    if int(args.wo_full_match) + int(args.wo_maxpool_match) + int(args.wo_attentive_match) + int(args.wo_max_attentive_match) == 4:
        raise ValueError('At least one of the matching should be enabled')

    model_dir = "saved_models_" + datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    pretrained_word_embedding = data.TEXT.vocab.vectors
    word_vocab_size = len(data.TEXT.vocab)
    pretrained_char_embedding = data.char_vocab.vectors
    char_vocab_size = len(data.char_vocab)
    class_size = len(data.LABEL.vocab)

    print(f"Creating model with class_size: {class_size}, word_vocab_size: {word_vocab_size}, char_vocab_size: {char_vocab_size}")
    print(f"pretrained_word_embedding: {pretrained_word_embedding}")
    print(f"pretrained_char_embedding: {pretrained_char_embedding}")

    model = BIMPM(class_size,
                  word_vocab_size,
                  char_vocab_size, 
                  pretrained_word_embedding=pretrained_word_embedding,
                  pretrained_char_embedding=pretrained_char_embedding,
                  word_dim=args.word_dim, 
                  char_dim=args.char_dim, 
                  num_perspective=args.num_perspective, 
                  use_char_emb=(not args.wo_char), 
                  context_lstm_dim=args.context_lstm_dim, 
                  context_layer_num=args.context_layer_num, 
                  aggregation_lstm_dim=args.aggregation_lstm_dim, 
                  aggregation_layer_num=args.aggregation_layer_num, 
                  char_lstm_dim=args.char_lstm_dim, 
                  dropout=args.dropout,
                  wo_full_match=args.wo_full_match,
                  wo_maxpool_match=args.wo_maxpool_match,
                  wo_attentive_match=args.wo_attentive_match,
                  wo_max_attentive_match=args.wo_max_attentive_match,
                  )
                  
    if args.gpu >= 0:
        model.cuda(args.gpu)
    print(model)
    
    print('Training start!')
    train(model, model_dir, args, data)

    print('Training finished!')

    print('Start prediction using best model')

    best_model = torch.load(f'{model_dir}/BIMPM_{args.data_type}_best.pt')

    if args.gpu > -1:
        best_model.cuda(args.gpu)

    print(best_model)

    _, acc, size = test(best_model, args, data)

    print(f'Test samples {size}, accuracy: {acc:.3f}')
