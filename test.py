import argparse

import torch
from torch import nn

from model.utils import SNLI, Quora


def test(model, args, data, mode='test'):
    if mode == 'dev':
        iterator = iter(data.dev_iter)
    else:
        iterator = iter(data.test_iter)

    criterion = nn.CrossEntropyLoss(size_average=False)
    model.eval()
    acc, loss, size = 0, 0, 0

    for batch in iterator:
        kwargs = data.get_features(batch)

        pred = model(**kwargs)

        batch_loss = criterion(pred, batch.label)
        loss += batch_loss.item()

        _, pred = pred.max(dim=1)
        acc += (pred == batch.label).sum().float()
        size += len(pred)

    acc /= size
    acc = acc.cpu().item()
    loss /= size
    return loss, acc, size

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--data-type', default='SNLI', help='available: SNLI or Quora')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--max-sent-len', default=-1, type=int,
                        help='max number of words per sentence, if -1, it accepts any length')
    parser.add_argument('--max-word-len', default=-1, type=int,
                        help='max number of chars per word, if -1, it accepts any length')

    args = parser.parse_args()

    if args.data_type == 'SNLI':
        print('Loading SNLI data...')
        data = SNLI(args)
    elif args.data_type == 'Quora':
        print('Loading Quora data...')
        data = Quora(args)

    print('Loading model...')
   
    model = torch.load(args.model_path)

    if args.gpu > -1:
        model.cuda(args.gpu)

    print(model)

    print('Doing prediction...')
    _, acc, size = test(model, args, data)

    print(f'Test samples {size}, accuracy: {acc:.3f}')
