# edit the code path accordingly
code_root = '/home/daeyun/git/factored3d/factored3d'
import sys
import os
import numpy as np
import os.path as osp
import scipy.misc
import matplotlib.pyplot as pt
import scipy.io as sio
import torch
from os import path

sys.path.append(osp.join(code_root, '..'))
from absl import flags
from factored3d.demo import demo_utils


def ensure_dir_exists(dirname, log_mkdir=True):
    dirname = path.realpath(path.expanduser(dirname))
    if not path.isdir(dirname):
        # `exist_ok` in case of race condition.
        os.makedirs(dirname, exist_ok=True)
        if log_mkdir:
            print('mkdir -p {}'.format(dirname))
    return dirname


def main():
    flags.FLAGS(['demo'])
    opts = flags.FLAGS

    # do not change the options below
    opts.batch_size = 1
    opts.num_train_epoch = 1
    opts.name = 'dwr_shape_ft'
    opts.classify_rot = True
    opts.pred_voxels = True
    opts.use_context = True

    if opts.classify_rot:
        opts.nz_rot = 24
    else:
        opts.nz_rot = 4

    # Load the trained models
    tester = demo_utils.DemoTester(opts)
    tester.init_testing()

    renderer = demo_utils.DemoRenderer(opts)
    # Load input data

    with open('/data3/nyu/eval.txt', 'r') as f:
        lines = f.readlines()
    names = [item.strip() for item in lines if item.strip()]

    for i, name in enumerate(names):
        print(i, name)

        img = scipy.misc.imread('/data3//nyu/images/{}.jpg'.format(name))
        img = img[3:-3]  # avoid aspect ratio stretching.
        img_fine = scipy.misc.imresize(img, (opts.img_height_fine, opts.img_width_fine))
        img_fine = np.transpose(img_fine, (2, 0, 1))
        img_coarse = scipy.misc.imresize(img, (opts.img_height, opts.img_width))
        img_coarse = np.transpose(img_coarse, (2, 0, 1))
        proposals = sio.loadmat('/data3/nyu/edgebox_proposals/{}_proposals.mat'.format(name))['proposals'][:, 0:4]
        inputs = {}
        inputs['img'] = torch.from_numpy(img_coarse / 255.0).unsqueeze(0)
        inputs['img_fine'] = torch.from_numpy(img_fine / 255.0).unsqueeze(0)
        inputs['bboxes_test_proposals'] = [torch.from_numpy(proposals)]
        tester.set_input(inputs)
        objects, layout = tester.predict_factored3d()

        mesh_dir = '/data3/out/scene3d/factored3d_pred/nyu/{}/'.format(name)
        ensure_dir_exists(mesh_dir)

        # uses a modified version of their code.
        renderer.save_codes_mesh(mesh_dir, objects, prefix='codes')
        renderer.save_layout_mesh(mesh_dir, layout, prefix='layout')


if __name__ == '__main__':
    main()
