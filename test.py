import time
import argparse

import torch
from torch import nn

from model.utils import SNLI, Quora


def test(model, args, data, mode='test'):
    if mode == 'dev':
        iterator = iter(data.dev_iter)
    else:
        iterator = iter(data.test_iter)

    criterion = nn.MSELoss(size_average=False)
    model.eval()
    acc, loss, size = 0, 0, 0

    for batch in iterator:
        kwargs = data.get_features(batch)

        pred = model(**kwargs)

        batch_loss = criterion(pred, batch.label.float())
        loss += batch_loss.item()

        pred = (pred > 0.5)
        acc += (pred == batch.label.byte()).sum().float()
        size += len(pred)

    acc /= size
    acc = acc.cpu().item()
    loss /= size
    return loss, acc, size

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=40, width=120))
    parser.add_argument('--model-path', required=True, help='path of trained model')
    parser.add_argument('--batch-size', default=16, type=int, help='batch size')
    parser.add_argument('--data-type', default='Quora', help='data type, available: SNLI or Quora')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id, -1 to use cpu')
    parser.add_argument('--max-sent-len', default=-1, type=int,
                        help='max number of words per sentence, if -1, it accepts any length')
    parser.add_argument('--max-word-len', default=-1, type=int,
                        help='max number of chars per word, if -1, it accepts any length')

    args = parser.parse_args()

    print('Loading model...')

    if args.gpu > -1:
        model = torch.load(args.model_path, map_location=lambda storage, loc: storage.cuda(args.gpu))
    else:
        model = torch.load(args.model_path, map_location=lambda storage, loc: storage)

    print(model)

    if args.data_type == 'SNLI':
        print('Loading SNLI data...')
        data = SNLI(args)
    elif args.data_type == 'Quora':
        print('Loading Quora data...')
        data = Quora(args)

    print('Doing prediction...')
    start_time = time.time()
    _, acc, size = test(model, args, data)

    time_used = time.time() - start_time

    print(f'Test samples: {size}, accuracy: {acc:.3f}, ms per example: {time_used / size}')
