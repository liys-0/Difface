import pickle
import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from train import run
from encoder import FACE_encoder,  Transformer, CLIP
from decoder import Decoder
from writer import Writer
import utils
import mesh_sampling
from psbody.mesh import Mesh
from sklearn.decomposition import PCA
from sklearn import svm

from sklearn.naive_bayes import GaussianNB

torch.backends.cudnn.enabled = True

torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser(description='CLIP')

# network hyperparameters
parser.add_argument('--out_channels',
                    nargs='+',
                    default=[16, 16, 16, 32],
                    type=int)
parser.add_argument('--latent_channels', type=int, default=128)
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--seq_length', type=int, default=[9, 9, 9, 9], nargs='+')
parser.add_argument('--dilation', type=int, default=[1, 1, 1, 1], nargs='+')
parser.add_argument('--device_idx', type=int, default = 1)

# optimizer hyperparmeters
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay', type=float, default=0.99)
parser.add_argument('--decay_step', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=0.30) 
# training hyperparameters
parser.add_argument('--epochs', type=int, default=200)
args = parser.parse_args()
args.work_dir = osp.dirname(osp.realpath(__file__))
args.out_dir = osp.join(args.work_dir, 'out')
args.checkpoints_dir = osp.join(args.out_dir, 'checkpoints')
print(args)
device = torch.device('cuda', args.device_idx)
writer = Writer(args)

# print the JS visualization code to the notebook
shap.initjs()

template_fp = osp.join('template.obj')

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

#cc generate/load transform matrices
transform_fp = osp.join('/share/home/jiaomingqi/test8/data/face', 'transform.pkl')
if not osp.exists(transform_fp):
    print('Generating transform matrices...')
    mesh = Mesh(filename=template_fp)
    ds_factors = [32, 32, 32, 32]
    _, A, D, U, F, V = mesh_sampling.generate_transform_matrices(
        mesh, ds_factors)
    tmp = {
        'vertices': V,
        'face': F,
        'adj': A,
        'down_transform': D,
        'up_transform': U
    }

    with open(transform_fp, 'wb') as fp:
        pickle.dump(tmp, fp)
    print('Done!')
    print('Transform matrices are saved in \'{}\'s'.format(transform_fp))
else:
    with open(transform_fp, 'rb') as f:
        tmp = pickle.load(f, encoding='latin1')

spiral_indices_list = [
    utils.preprocess_spiral(tmp['face'][idx], args.seq_length[idx],
                            tmp['vertices'][idx],
                            args.dilation[idx]).to(device)
    for idx in range(len(tmp['face']) - 1)
]
down_transform_list = [
    utils.to_sparse(down_transform).to(device)
    for down_transform in tmp['down_transform']
]
up_transform_list = [
    utils.to_sparse(up_transform).to(device)
    for up_transform in tmp['up_transform']
]

image_encoder = FACE_encoder(args.in_channels, args.out_channels, args.latent_channels,
                            spiral_indices_list, down_transform_list,
                            up_transform_list).to(device)

text_encoder = Transformer(num_snps=train_dataset.num_snps_after_filter).to(device)

model = CLIP(
    image_encoder = image_encoder,
    text_encoder = text_encoder,
).to(device)

decoder = Decoder(args.in_channels, args.out_channels, args.latent_channels,
                            spiral_indices_list, down_transform_list,
                            up_transform_list).to(device)

optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': 0.073e-4, 'weight_decay':args.weight_decay}, 
	                          {'params': decoder.parameters(), 'lr': 0.32e-4, 'weight_decay':0.001}])

scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            args.decay_step,
                                            gamma=args.lr_decay)

run(model, decoder, train_loader, test_loader, args.epochs, optimizer, scheduler, writer, device)
