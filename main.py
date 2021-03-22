import os
import argparse
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    # device = "cuda"
    device = "cpu"
else:
    device = "cpu"
print(f"Use {device} for torch")

from dataset.dataloader import load_data, get_loader
from dataset.field import Vocab
from utils import seq2sen
import model

class LabelSmoothingsLoss(nn.Module):
    def __init__(self, num_classes, smoothing, dim, is_train):
        super().__init__()
        self.confidence = 1 - smoothing
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.dim = dim
        self.is_train = is_train

    def forward(self, pred, target, pad_start_idx_list = None):
        # if self.is_train and pad_start_idx_list:
        #     pred_list, target_list = [], []
        #     for batch_idx in range(pred.shape[0]):
        #         pred_list.append(pred[batch_idx, :pad_start_idx_list[batch_idx], :])
        #         target_list.append(target[batch_idx, :pad_start_idx_list[batch_idx]])

        #     loss = 0
        #     for batch_idx in range(pred.shape[0]):
        #         pred_dist = torch.log(pred_list[batch_idx].clone().detach().requires_grad_(True))
        #         true_dist = torch.zeros_like(pred_dist)
        #         true_dist.fill_(self.smoothing / (self.num_classes - 1))
        #         true_dist.scatter_(self.dim, target_list[batch_idx].unsqueeze(self.dim), self.confidence)
        #         loss += torch.sum(-true_dist * pred_dist)
        #     loss /= pred.shape[0]
        #     return loss

        # else:
            # pred = torch.log(pred.clone().detach().requires_grad_(True))
            # true_dist = torch.zeros_like(pred)
            # true_dist.fill_(self.smoothing / (self.num_classes - 1))
            # true_dist.scatter_(self.dim, target.data.unsqueeze(self.dim), self.confidence)
            # return torch.mean(torch.sum(-true_dist * pred, dim = self.dim))
        pred = torch.log(pred.clone().detach().requires_grad_(True))
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        true_dist.scatter_(self.dim, target.data.unsqueeze(self.dim), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim = self.dim))

def get_pad_start_idx_list(tgt_batch):
    pad_start_idx_list = []
    for batch_idx in range(tgt_batch.shape[0]):
        try:
            pad_start_idx = list(tgt_batch[batch_idx]).index(2) # <pad>
        except ValueError:
            pad_start_idx = tgt_batch[batch_idx].shape[0]
        pad_start_idx_list.append(pad_start_idx)
    return pad_start_idx_list

def get_learning_rate(model_dim, step_num, warmup_steps):
    # return (model_dim ** (-0.5)) * min(step_num ** (-0.5), step_num * (warmup_steps ** (-1.5)))
    return 1e-3

def main(args):
    src, tgt = load_data(args.path)

    src_vocab = Vocab(init_token='<sos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
    src_vocab.load(os.path.join(args.path, 'vocab.en'))
    tgt_vocab = Vocab(init_token='<sos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
    tgt_vocab.load(os.path.join(args.path, 'vocab.de'))

    # TODO(completed): use these information.
    sos_idx = 0
    eos_idx = 1
    pad_idx = 2
    max_length = 50

    # TODO: use these values to construct embedding layers
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    # Define training parameters
    model_dim = 512
    warmup_steps = 4000

    # Define model
    transformer = model.Transformer(model_dim, src_vocab_size, tgt_vocab_size, max_length)
    if args.test and args.model_path:
        transformer.load_state_dict(torch.load(args.model_path))
        transformer.eval()

    # Define optimizer
    step_num = 1
    learning_rate = get_learning_rate(model_dim, step_num, warmup_steps)
    optimizer = optim.Adam(transformer.parameters(), lr = learning_rate, betas = (0.9, 0.98), eps = 1e-8)

    # Define loss function
    smoothing = 0.1
    train_loss_function = LabelSmoothingsLoss(tgt_vocab_size, smoothing, dim = -1, is_train = True)
    validation_loss_function = LabelSmoothingsLoss(tgt_vocab_size, smoothing, dim = -1, is_train = False)
    # loss_function = nn.CrossEntropyLoss()


    if not args.test:
        train_loader = get_loader(src['train'], tgt['train'], src_vocab, tgt_vocab, batch_size=args.batch_size, shuffle=True)
        valid_loader = get_loader(src['valid'], tgt['valid'], src_vocab, tgt_vocab, batch_size=args.batch_size)

        history = {
            "loss" : [],
            "val_loss" : [],
            "accuracy" : [],
            "val_accuracy" : []
        }

        for epoch in range(args.epochs):
            print(f"Epoch {epoch + 1}/{args.epochs}")
            total_train_size, total_validation_size = 0, 0
            epoch_train_loss, epoch_validation_loss = 0, 0
            epoch_train_correct, epoch_validation_correct = 0, 0

            # TODO(completed): train
            for src_batch, tgt_batch in tqdm(train_loader):
                for g in optimizer.param_groups: # update learning rate first
                    g['lr'] = get_learning_rate(model_dim, step_num, warmup_steps)
                optimizer.zero_grad()

                prd_batch = transformer(torch.tensor(src_batch).to(device), torch.tensor(tgt_batch).to(device))
                pad_start_idx_list = get_pad_start_idx_list(torch.tensor(tgt_batch))
                loss = train_loss_function(torch.tensor(prd_batch).cpu(), torch.tensor(tgt_batch).cpu(), pad_start_idx_list)
                loss.backward()
                optimizer.step()
                step_num += 1

                total_train_size += len(src_batch)
                epoch_train_loss += loss
                epoch_train_correct += int(torch.sum(torch.argmax(torch.tensor(prd_batch).data, -1) == torch.tensor(tgt_batch)))

            epoch_train_loss /= total_train_size
            epoch_train_accuracy = epoch_train_correct / total_train_size

            history["loss"].append(epoch_train_loss)    
            history["accuracy"].append(epoch_train_accuracy)

            # TODO: validation
            for src_batch, tgt_batch in tqdm(valid_loader):
                prd_batch = transformer(torch.tensor(src_batch).to(device), torch.tensor(tgt_batch).to(device))
                loss = validation_loss_function(torch.tensor(prd_batch).cpu(), torch.tensor(tgt_batch).cpu())

                total_validation_size += len(src_batch)
                epoch_validation_loss += loss
                epoch_validation_correct += int(torch.sum(torch.argmax(torch.tensor(prd_batch).data, -1) == torch.tensor(tgt_batch)))

            epoch_validation_loss /= total_validation_size
            epoch_validation_accuracy = epoch_validation_correct / total_validation_size

            history["val_loss"].append(epoch_validation_loss)
            history["val_accuracy"].append(epoch_validation_accuracy)

            print(f"loss : {epoch_train_loss:.6f}, val_loss : {epoch_validation_loss:.6f}") 
            print(f"accuracy : {epoch_train_accuracy:.6f}, val_accuracy : {epoch_validation_accuracy:.6f}") 

            transformer.save_model(args.output_path, epoch + 1, epoch_train_loss, epoch_validation_loss)

            gc.collect()
            torch.cuda.empty_cache()

        transformer.plot(args.output_path, history)

    else:
        # test
        test_loader = get_loader(src['test'], tgt['test'], src_vocab, tgt_vocab, batch_size=args.batch_size)

        pred = []
        for src_batch, tgt_batch in tqdm(test_loader):
            # TODO: predict pred_batch from src_batch with your model.
            pred_batch = transformer(torch.tensor(src_batch).to(device), torch.tensor(tgt_batch).to(device))
            pred_batch[:][:][0] += float("-inf") # exclude <sos>
            pred_batch[:][:][2] += float("-inf") # exclude <pad>
            pred_batch = torch.argmax(pred_batch, dim = -1)

            result_batch = []
            for batch_idx in range(pred_batch.shape[0]):
                result = []
                for idx in range(pred_batch.shape[1] + 1):
                    if idx == 0: # <sos>
                        result.append(0)
                    elif idx == pred_batch.shape[1] or idx == max_length - 1 or pred_batch[batch_idx, idx - 1] == 1: # <eos>
                        result.append(1)
                        break
                    else: # normal case
                        result.append(int(pred_batch[batch_idx, idx - 1]))
                result_batch.append(result)

            max_length = max([len(result) for result in result_batch])
            for batch_idx in range(len(result_batch)):
                result_batch[batch_idx] = result_batch[batch_idx] + [2] * (max_length - len(result_batch[batch_idx]))
            pred += seq2sen(result_batch, tgt_vocab)

        with open('results/pred.txt', 'w') as f:
            for line in pred:
                f.write('{}\n'.format(line))
            f.close()

        os.system('bash scripts/bleu.sh results/pred.txt multi30k/test.de.atok')

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
        default=16)
    parser.add_argument(
        '--test',
        action='store_true')
    parser.add_argument(
        "--output_path",
        type=str,
        default="resources")
    parser.add_argument(
        "--model_path",
        type=str,
        default="")

    args = parser.parse_args()
    main(args)