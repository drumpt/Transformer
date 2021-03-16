import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim as optim
import tqdm

from dataset.dataloader import load_data, get_loader
from dataset.field import Vocab
from utils import seq2sen
import model

class LabelSmoothingsLoss(nn.Module):
    def __init__(self, num_classes, smoothing, dim = -1):
        super().__init__()
        self.confidence = 1 - smoothing
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.dim = dim

    def forward(self, pred, target):
        with torch.no_gard():
            pred = pred.log_softmax(dim = self.dim)
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim = self.dim))

def main(args):
    src, tgt = load_data(args.path)

    src_vocab = Vocab(init_token='<sos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
    src_vocab.load(os.path.join(args.path, 'vocab.en'))
    tgt_vocab = Vocab(init_token='<sos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
    tgt_vocab.load(os.path.join(args.path, 'vocab.de'))

    # TODO: use these information.
    sos_idx = 0
    eos_idx = 1
    pad_idx = 2
    max_length = 50

    # TODO: use these values to construct embedding layers
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    model_dim = 512
    warmup_steps = 4000

    # Define model
    transformer = model.Transformer(model_dim, src_vocab_size, tgt_vocab_size, max_length)

    # Define optimizer
    step_num = 1
    learning_rate = get_learning_rate(model_dim, step_num, warmup_steps)
    optimizer = optim.Adam(transformer.parameters(), lr = learning_rate, betas = (0.9, 0.98), eps = 1e-8)

    # Define loss function
    smoothing = 0.1
    loss_function = LabelSmoothingsLoss(tgt_vocab_size, smoothing)

    if not args.test:
        train_loader = get_loader(src['train'], tgt['train'], src_vocab, tgt_vocab, batch_size=args.batch_size, shuffle=True)
        valid_loader = get_loader(src['valid'], tgt['valid'], src_vocab, tgt_vocab, batch_size=args.batch_size)

        # TODO: train
        for epoch in range(args.epochs):
            for src_batch, tgt_batch in tqdm(train_loader):
                for g in optimizer.param_groups: # update learning rate first
                    g['lr'] = get_learning_rate(model_dim, step_num, warmup_steps)
                optimizer.zero_grad()

                prd_batch = transformer(torch.tensor(src_batch), torch.tensor(tgt_batch))
                loss = loss_function(prd_batch, tgt_batch)
                loss.backward()
                optimizer.step()
                step_num += 1

            # TODO: validation
            for src_batch, tgt_batch in tqdm(valid_loader):
                prd_batch = transformer(torch.tensor(src_batch), torch.tensor(tgt_batch))
                loss = loss_function(prd_batch, tgt_batch)


    else:
        # test
        test_loader = get_loader(src['test'], tgt['test'], src_vocab, tgt_vocab, batch_size=args.batch_size)

        pred = []
        for src_batch, tgt_batch in tqdm(test_loader):
            # TODO: predict pred_batch from src_batch with your model.
            pred_batch = tgt_batch

            # every sentences in pred_batch should start with <sos> token (index: 0) and end with <eos> token (index: 1).
            # every <pad> token (index: 2) should be located after <eos> token (index: 1).
            # example of pred_batch:
            # [[0, 5, 6, 7, 1],
            #  [0, 4, 9, 1, 2],
            #  [0, 6, 1, 2, 2]]
            pred += seq2sen(pred_batch, tgt_vocab)

        with open('results/pred.txt', 'w') as f:
            for line in pred:
                f.write('{}\n'.format(line))

        os.system('bash scripts/bleu.sh results/pred.txt multi30k/test.de.atok')

def get_learning_rate(model_dim, step_num, warmup_steps)
    return (model_dim ** (-0.5)) * min(step_num ** (-0.5), step_num * (warmup_steps ** (-1.5)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument(
        '--path',
        type=str,
        default='multi30k')

    parser.add_argument(
        '--epochs',
        type=int,
        default=100)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128)

    parser.add_argument(
        '--test',
        action='store_true')
    args = parser.parse_args()

    main(args)
