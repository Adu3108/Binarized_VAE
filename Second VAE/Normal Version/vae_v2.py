from typing import Tuple

import argparse
import json
import math
import os

import numpy
import torch
import torchvision
import matplotlib.pyplot as plt

class Vae(torch.nn.Module):

    def __init__(self,
                 input_d: Tuple[int],
                 n_frames: int,
                 n_latent: int,
                 batch: int):
        super(Vae, self).__init__()
        self.n_frames = n_frames
        self.n_latent = n_latent
        self.input_d = input_d
        self.batch = batch

        x, y = input_d
        for i in range(4):
            x = math.floor((x - 1) / 3 + 1)
            y = math.floor((y - 1) / 3 + 1)
        self.hidden_x = x
        self.hidden_y = y

        self.enc_conv1 = torch.nn.Conv2d(self.n_frames, 32, (5, 5), stride=(3, 3), padding=(2, 2))
        self.enc_bn1 = torch.nn.BatchNorm2d(32)
        self.enc_af1 = torch.nn.Hardtanh(inplace=True)
        
        self.enc_conv2 = torch.nn.Conv2d(32, 64, (5, 5), stride=(3, 3), padding=(2, 2))
        self.enc_bn2 = torch.nn.BatchNorm2d(64)
        self.enc_af2 = torch.nn.Hardtanh(inplace=True)

        self.enc_conv3 = torch.nn.Conv2d(64, 128, (5, 5), stride=(3, 3), padding=(2, 2))
        self.enc_bn3 = torch.nn.BatchNorm2d(128)
        self.enc_af3 = torch.nn.Hardtanh(inplace=True)

        self.enc_conv4 = torch.nn.Conv2d(128, 256, (5, 5), stride=(3, 3), padding=(2, 2))
        self.enc_bn4 = torch.nn.BatchNorm2d(256)
        self.enc_af4 = torch.nn.Hardtanh(inplace=True)

        self.linear_mu = torch.nn.Linear(256 * self.hidden_x * self.hidden_y, self.n_latent)
        self.linear_var = torch.nn.Linear(256 * self.hidden_x * self.hidden_y, self.n_latent)

        self.dec_linear = torch.nn.Linear(self.n_latent, 256 * self.hidden_x * self.hidden_y)

        self.dec_conv4 = torch.nn.ConvTranspose2d(256, 128, (5, 5), stride=(3, 3), padding=(2, 2), output_padding=(1, 2))
        self.dec_bn4 = torch.nn.BatchNorm2d(128)
        self.dec_af4 = torch.nn.Hardtanh(inplace=True)

        self.dec_conv3 = torch.nn.ConvTranspose2d(128, 64, (5, 5), stride=(3, 3), padding=(2, 2), output_padding=(1, 2))
        self.dec_bn3 = torch.nn.BatchNorm2d(64)
        self.dec_af3 = torch.nn.Hardtanh(inplace=True)

        self.dec_conv2 = torch.nn.ConvTranspose2d(64, 32, (5, 5), stride=(3, 3), padding=(2, 2), output_padding=(0, 2))
        self.dec_bn2 = torch.nn.BatchNorm2d(32)
        self.dec_af2 = torch.nn.Hardtanh(inplace=True)

        self.dec_conv1 = torch.nn.ConvTranspose2d(32, self.n_frames, (5, 5), stride=(3, 3), padding=(2, 2), output_padding=(2,0))
        self.dec_bn1 = torch.nn.BatchNorm2d(self.n_frames)
        self.dec_af1 = torch.nn.Sigmoid()

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        #print(x.shape)
        x = self.enc_conv1(x)
        x = self.enc_bn1(x)
        x = self.enc_af1(x)
        #print(x.shape)
        x = self.enc_conv2(x)
        x = self.enc_bn2(x)
        x = self.enc_af2(x)
        #print(x.shape)
        x = self.enc_conv3(x)
        x = self.enc_bn3(x)
        x = self.enc_af3(x)
        #print(x.shape)
        x = self.enc_conv4(x)
        x = self.enc_bn4(x)
        x = self.enc_af4(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        var = self.linear_var(x)
        return mu, var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.dec_linear(z)
        z = torch.reshape(z, [self.batch, 256, self.hidden_x, self.hidden_y])
        z = self.dec_conv4(z)
        z = self.dec_bn4(z)
        z = self.dec_af4(z)
        #print(z.shape)
        z = self.dec_conv3(z)
        z = self.dec_bn3(z)
        z = self.dec_af3(z)
        #print(z.shape)
        z = self.dec_conv2(z)
        z = self.dec_bn2(z)
        z = self.dec_af2(z)
        #print(z.shape)
        z = self.dec_conv1(z)
        z = self.dec_bn1(z)
        #print(z.shape)
        return self.dec_af1(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        mu, logvar = self.encode(x)
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        z = mu + std * eps
        out = self.decode(z)
        return out, mu, logvar

    def train_self(self,
                   train_path: str,
                   val_path: str,
                   weights: str,
                   epochs: int,
                   use_flows: bool = False) -> None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')

        model = self.to(device)
        
        def npy_loader(path: str) -> torch.Tensor:
            sample = torch.from_numpy(numpy.load(path))
            sample = torch.swapaxes(sample, 1, 2)
            sample = torch.swapaxes(sample, 0, 1)
            sample = sample.nan_to_num(0)
            sample = ((sample + 64) / 128).clamp(0, 1)
            return sample.type(torch.FloatTensor) 

        if use_flows:
            train_set = torchvision.datasets.DatasetFolder(
                root=train_path,
                loader=npy_loader,
                extensions=['.npy'])
            val_set = torchvision.datasets.DatasetFolder(
                root=val_path,
                loader=npy_loader,
                extensions=['.npy'])
        else:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(self.input_d),
                torchvision.transforms.Grayscale()])
            train_set = torchvision.datasets.ImageFolder(
                root=train_path,
                transform=transforms)
            val_set = torchvision.datasets.ImageFolder(
                root=val_path,
                transform=transforms)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=self.batch,
            shuffle=True,
            drop_last=True)
        val_loader = torch.utils.data.DataLoader(
            dataset=val_set,
            batch_size=self.batch,
            shuffle=True,
            drop_last=True)
        
        optimizer = torch.optim.Adam(model.parameters())

        for epoch in range(epochs):
            print('----------------------------------------------------')
            print(f'Epoch: {epoch}')

            model.train()
            epoch_tl = 0
            train_count = 0
            for data in train_loader:
                x, _ = data
               
                #print(f'MAX: {x.max()}')
                #print(f'MIN: {x.min()}')

                x = x.to(device)
                x_hat, mu, logvar = model(x)

                kl_loss = torch.mul(
                    input=torch.sum(mu.pow(2) + logvar.exp() - logvar - 1),
                    other=0.5)
                mse_loss = torch.nn.functional.mse_loss(x_hat, x, reduction='sum')
                loss = mse_loss + kl_loss
                #ce_loss = torch.nn.functional.binary_cross_entropy(
                #    input=x_hat,
                #    target=x,
                #    reduction='sum')
                #loss = ce_loss + kl_loss
                epoch_tl += loss
                train_count += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'Training Loss: {epoch_tl/train_count}')
            
            with torch.no_grad():
                epoch_vl = 0
                val_count = 0
                for data in val_loader:
                    x, _ = data
                    x = x.to(device)
                    x_hat, mu, logvar = model(x)

                    kl_loss = torch.mul(
                        input=torch.sum(mu.pow(2) + logvar.exp() - logvar - 1),
                        other=0.5)
                    mse_loss = torch.nn.functional.mse_loss(x_hat, x, reduction='sum')
                    epoch_vl += mse_loss + kl_loss
                    #ce_loss = torch.nn.functional.binary_cross_entropy(
                    #    input=x_hat,
                    #    target=x,
                    #    reduction='sum')
                    #epoch_vl += ce_loss + kl_loss
                    val_count += 1
                print(f'Validation Loss: {epoch_vl/val_count}')
    
            print('----------------------------------------------------')
        print('Training finished, saving weights...')
        torch.save(model, weights)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train or convert a VAE model.')
    parser.add_argument(
        'action',
        choices=['train', 'calibrate'],
        metavar='ACTION')
    parser.add_argument(
        '--weights',
        help='Path to weights file')
    parser.add_argument(
        '--n_latent',
        help='Number of latent variables in the model.')
    parser.add_argument(
        '--dimensions',
        help='Dimensions of input image accepted by the network (height x width).')
    parser.add_argument(
        '--train_set',
        help='Path to the training set.')
    parser.add_argument(
        '--validation_set',
        help='Path to the cross validation set (for monitoring training only, does not affect weight calculation).')
    parser.add_argument(
        '--cal_set',
        help='Path to calibration set (calibrate action only).')
    parser.add_argument(
        '--batch',
        help='Batch size to use for training.')
    parser.add_argument(
        '--epochs',
        help='Epochs to train.')
    parser.add_argument(
        '--flows',
        type=int,
        default=0,
        help='Training set is optical flows (.npy cubic tensors).')
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(0)

    if args.action == 'train':
        model = Vae(
            input_d=tuple([int(i) for i in args.dimensions.split('x')]),
            n_frames=1 if args.flows <= 0 else args.flows,
            n_latent=int(args.n_latent),
            batch=int(args.batch))
        model.train_self(
            train_path=args.train_set,
            val_path=args.validation_set,
            weights=args.weights,
            epochs=int(args.epochs),
            use_flows=False if args.flows <= 0 else True)
    if args.action == 'calibrate':
        model = torch.load(args.weights)
        model.eval()

        def npy_loader(path: str) -> torch.Tensor:
            sample = torch.from_numpy(numpy.load(path))
            sample = torch.swapaxes(sample, 1, 2)
            sample = torch.swapaxes(sample, 0, 1)
            sample = sample.nan_to_num(0)
            sample = ((sample + 64) / 128).clamp(0, 1)
            return sample.type(torch.FloatTensor)

        if args.flows <= 0:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(model.input_d),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.ToTensor()])
            cal_set = torchvision.datasets.ImageFolder(
                root=args.cal_set,
                transform=transforms)
        else:
            cal_set = torchvision.datasets.DatasetFolder(
                root=args.cal_set,
                loader=npy_loader,
                extensions=['.npy'])
        cal_loader = torch.utils.data.DataLoader(
            dataset=cal_set,
            batch_size=model.batch,
            shuffle=True,
            drop_last=True)
        
        kl_losses = []
        #ce_losses = []
        mse_losses = []
        final = []
        index = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            for data in cal_loader:
                x, _ = data
                final_input = x
                x = x.to(device)
                x_hat, mu, logvar = model(x)
                
                kl_loss = torch.mul(
                    input=torch.sum(mu.pow(2) + logvar.exp() - logvar - 1, 1),
                    other=0.5)
                #ce_loss = -torch.sum(
                #        x_hat * x.log2().nan_to_num(-100) + (1 - x_hat) * (1 - x).log2().nan_to_num(-100),
                #        (1, 2, 3))
                mse_loss = torch.sum((x - x_hat).pow(2), (1, 2, 3))
                l = [index, final_input, x_hat, kl_loss.detach().cpu().numpy(), mse_loss.detach().cpu().numpy()]
                final.append(l)
                kl_losses.extend(list(kl_loss.detach().cpu().numpy()))
                #ce_losses.extend(list(ce_loss.detach().cpu().numpy()))
                mse_losses.extend(list(mse_loss.detach().cpu().numpy()))
                index += 1
        kl_losses_2 = kl_losses[:]
        mse_losses_2 = mse_losses[:]
        kl_losses.sort()
        mse_losses.sort()
        #ce_losses.sort()
        kl_losses = [i.item() for i in kl_losses]
        mse_losses = [i.item() for i in mse_losses]
        #ce_losses = [i.item() for i in ce_losses]
        with open(f'cal_{".".join(args.weights.split(".")[:-1])}.json', 'w') as cal_f:
            cal_f.write(json.dumps({'kl_loss': kl_losses, 'mse_loss': mse_losses}))
        for i in range(len(final)):
          input_matrix = final[i][1][0][0] 
          output_matrix = final[i][2][0][0] 
          matrix_shape = input_matrix.shape
          input_img = input_matrix.cpu().numpy()
          output_img = output_matrix.cpu().numpy()
          numpy.savetxt('input.csv', input_img, delimiter=',')
          numpy.savetxt('output.csv', output_img, delimiter=',')
          plt.imsave(f'/content/Results/Input/{i}.png', input_img, cmap='gray')
          plt.imsave(f'/content/Results/Reconstructed/{i}.png', output_img, cmap='gray')
        final_kl_dict = {}
        final_mse_dict = {}
        for i in final:
            final_kl_dict[f'{i[0]}'] = str(i[3])
            final_mse_dict[f'{i[0]}'] = str(i[4])
        with open("kl_loss.json", "w") as KL_Json:
            json.dump(final_kl_dict, KL_Json)
        with open("mse_loss.json","w") as MSE_Json:
            json.dump(final_mse_dict, MSE_Json)