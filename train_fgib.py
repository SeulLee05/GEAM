import os
import argparse
import datetime
from time import time
from tqdm import tqdm

import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.fgib import FGIB
from utils_fgib.utils import get_loader
from utils_sac.utils import set_seed


class Trainer():
    def __init__(self, args, train_loader, test_loader):
        self.expname = f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{args.target}"
        print(f'\033[92m{self.expname}\033[0m')
        os.makedirs(f'ckpt/{self.expname}')
        with open(f'ckpt/{self.expname}/log.log', 'a') as f:
            f.write(f'{args}\n')

        self.args = args
        set_seed(args.seed)

        if args.gpu_id >= 0:
            self.device = torch.device(f'cuda:{args.gpu_id}')
        else:
            self.device = torch.device('cpu')
        
        self.model = FGIB(device=self.device).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=args.patience, mode='min', verbose=True)
        self.save_epoch = args.save_epoch
        self.train_loader, self.test_loader = train_loader, test_loader

    def train(self):
        loss_fn = torch.nn.MSELoss()
        
        start_time = time()
        for epoch in range(1, self.args.epochs + 1):
            train_loss, train_mse, test_mse, preserve = 0, 0, 0, 0

            self.model.train()
            for graph, value in tqdm(self.train_loader, desc=f'Epoch {epoch}'):
                graph, value = graph.to(self.device), value.to(self.device)
                self.optimizer.zero_grad()

                pred, KL_loss, preserve_rate = self.model(graph)
                MSE_loss = loss_fn(pred, value.reshape(-1, 1).float())
                loss = MSE_loss + self.args.beta * KL_loss
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_mse += MSE_loss.item()
                preserve += preserve_rate        

            if self.test_loader is not None:
                self.model.eval()
                for graph, value in self.test_loader:
                    graph, value = graph.to(self.device), value.to(self.device)
                    
                    pred, _, _ = self.model(graph)
                    MSE_loss = loss_fn(pred, value.reshape(-1, 1).float())
                    test_mse += MSE_loss.item()

            self.scheduler.step(loss)

            train_loss /= len(self.train_loader)
            train_mse /= len(self.train_loader)
            preserve /= len(self.train_loader)
            if self.test_loader is not None:
                test_mse /= len(self.test_loader)

            with open(f'ckpt/{self.expname}/log.log', 'a') as f:
                log = f'[Epoch {epoch:03d} | {time() - start_time:.1f} sec] ' + \
                      f'Train Loss: {train_loss:.4e} | Train MSE: {train_mse:.4e} | '
                if self.test_loader:
                    log += f'Test MSE: {test_mse:.4e} | Preserve: {preserve:.4f}\n'
                else:
                    log += f'Preserve: {preserve:.4f}\n'
                f.write(log)
            
            if epoch % self.save_epoch == 0:
                torch.save({'state_dict': self.model.state_dict(), 'args': self.args},
                           f'ckpt/{self.expname}/epoch_{epoch}.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu_id", type=int, default=-1)
    parser.add_argument("-t", "--target", type=str, default='parp1',
                        choices=['parp1', 'fa7', '5ht1b', 'braf', 'jak2',
                                 'amlodipine_mpo', 'fexofenadine_mpo',
                                 'osimertinib_mpo', 'perindopril_mpo',
                                 'ranolazine_mpo', 'sitagliptin_mpo', 'zaleplon_mpo'])
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--save_epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--beta", type=float, default=1e-5)
    args = parser.parse_args()

    train_loader, test_loader = get_loader(args.target, args.batch_size)
    trainer = Trainer(args, train_loader, test_loader)
    trainer.train()
