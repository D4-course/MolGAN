import os
import logging

from rdkit import RDLogger

from args import get_VAE_config
from util_dir.utils_io import get_date_postfix

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Remove flooding logs.
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

from solver_vae import Solver
from torch.backends import cudnn

class Conf:
    batch_size=128
    d_conv_dim=[[128, 64], 128, [128, 64]]
    d_lr=0.0001
    dropout=0.0
    g_conv_dim=[128, 256, 512]
    g_lr=0.0001
    lambda_cls=1
    lambda_rec=10
    lambda_wgan=1.0
    log_step=10
    lr_update_step=1000
    mode=str('test')
    model_save_step=1
    mol_data_dir='data/qm9_5k.sparsedataset'
    n_critic=5
    num_epochs=150
    num_workers=1
    post_method='softmax'
    resume_epoch=150
    sample_step=1000
    saving_dir='exp_results/VAE/2022-10-16_19-18-54'
    test_epochs=100
    z_dim=8


def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Timestamp
    if config.mode == 'train':
        config.saving_dir = os.path.join(config.saving_dir, get_date_postfix())
        config.log_dir_path = os.path.join(config.saving_dir, 'log_dir')
        config.model_dir_path = os.path.join(config.saving_dir, 'model_dir')
        config.img_dir_path = os.path.join(config.saving_dir, 'img_dir')
    else:
        a_test_time = get_date_postfix()
        config.saving_dir = os.path.join(config.saving_dir)
        config.log_dir_path = os.path.join(config.saving_dir, 'post_test', a_test_time, 'log_dir')
        config.model_dir_path = os.path.join(config.saving_dir, 'model_dir')
        config.img_dir_path = os.path.join(config.saving_dir, 'post_test', "final", 'img_dir')

    # Create directories if not exist.
    if not os.path.exists(config.log_dir_path):
        os.makedirs(config.log_dir_path)
    if not os.path.exists(config.model_dir_path):
        os.makedirs(config.model_dir_path)
    if not os.path.exists(config.img_dir_path):
        os.makedirs(config.img_dir_path)

    # Logger
    if config.mode == 'train':
        log_p_name = os.path.join(config.log_dir_path, get_date_postfix() + '_logger.log')
        logging.basicConfig(filename=log_p_name, level=logging.INFO)
        logging.info(config)

    # Solver for training and testing StarGAN.
    if config.mode == 'train':
        solver = Solver(config, logging)
    elif config.mode == 'test':
        solver = Solver(config)
    else:
        raise NotImplementedError

    solver.train_and_validate()


if __name__ == '__main__':
    # config = get_VAE_config()
    config = Conf()
    print(f"-------------{type(config.mode)}---------------")

    print(config)
    main(config)
