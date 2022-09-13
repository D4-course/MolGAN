from typing import Union, List, Any, Dict

import numpy as np
import os
import time
import datetime

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image

from utils import *
from models import Generator, Discriminator
from data.sparse_molecular_dataset import SparseMolecularDataset


class Solver(object):
    """Solver for training and testing StarGAN."""
    logger: object
    z_dim: object
    m_dim: int
    b_dim: int
    g_conv_dim: object
    d_conv_dim: object
    g_repeat_num: object
    d_repeat_num: object
    lambda_cls: object
    lambda_rec: object
    lambda_gp: object
    post_method: object
    metric: str
    batch_size: object
    num_iters: object
    num_iters_decay: object
    g_lr: object
    d_lr: object
    dropout: object
    n_critic: object
    beta1: object
    beta2: object
    resume_iters: object
    test_iters: object
    log_dir: object
    sample_dir: object
    model_save_dir: object
    result_dir: object
    log_step: object
    sample_step: object
    model_save_step: object
    lr_update_step: object
    use_tensorboard: object
    g_optimizer: object
    d_optimizer: object
    G: Generator
    D: Discriminator
    V: Discriminator

    def __init__(self, config: object) -> object:
        """Initialize configurations.
        @param config:
        """

        # Data loader.
        self.data = SparseMolecularDataset()
        self.data.load(config.mol_data_dir)

        # Model configurations.
        self.z_dim = config.z_dim
        self.m_dim = self.data.atom_num_types
        self.b_dim = self.data.bond_num_types
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.post_method = config.post_method

        self.metric = "validity,sas"

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.dropout = config.dropout
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self) -> object:
        """Create a generator and a discriminator."""
        self.G = Generator(
            self.g_conv_dim,
            self.z_dim,
            self.data.vertexes,
            self.data.bond_num_types,
            self.data.atom_num_types,
            self.dropout,
        )
        self.D = Discriminator(self.d_conv_dim, self.m_dim, self.b_dim, self.dropout)
        self.V = Discriminator(self.d_conv_dim, self.m_dim, self.b_dim, self.dropout)

        self.g_optimizer = torch.optim.Adam(
            list(self.G.parameters()) + list(self.V.parameters()),
            self.g_lr,
            [self.beta1, self.beta2],
        )
        self.d_optimizer = torch.optim.Adam(
            self.D.parameters(), self.d_lr, [self.beta1, self.beta2]
        )
        self.print_network(self.G, "G")
        self.print_network(self.D, "D")

        self.G.to(self.device)
        self.D.to(self.device)
        self.V.to(self.device)

    def print_network(self, model: object, name: object) -> object:
        """Print out the network information.
        @param model:
        @param name:
        """
        num_params: int = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters: object) -> object:
        """Restore the trained generator and discriminator.
        @param resume_iters:
        """
        print("Loading the trained models from step {}...".format(resume_iters))
        G_path: Union[bytes, str] = os.path.join(self.model_save_dir, "{}-G.ckpt".format(resume_iters))
        D_path: Union[bytes, str] = os.path.join(self.model_save_dir, "{}-D.ckpt".format(resume_iters))
        V_path: Union[bytes, str] = os.path.join(self.model_save_dir, "{}-V.ckpt".format(resume_iters))
        self.G.load_state_dict(
            torch.load(G_path, map_location=lambda storage, loc: storage)
        )
        self.D.load_state_dict(
            torch.load(D_path, map_location=lambda storage, loc: storage)
        )
        self.V.load_state_dict(
            torch.load(V_path, map_location=lambda storage, loc: storage)
        )

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger

        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr: object, d_lr: object) -> object:
        """Decay learning rates of the generator and discriminator.
        @param g_lr:
        @param d_lr:
        """
        param_group: object
        for param_group in self.g_optimizer.param_groups:
            param_group["lr"] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group["lr"] = d_lr

    def reset_grad(self) -> object:
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x: object) -> object:
        """Convert the range from [-1, 1] to [0, 1].
        @param x:
        @return:
        """
        out: float = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y: object, x: object) -> object:
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2.
        @param y:
        @param x:
        @return:
        """
        weight: object = torch.ones(y.size()).to(self.device)
        dydx: object = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=weight,
            retain_graph=True,
            create_graph=True,
            only_inputs=True,
        )[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm: object = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def label2onehot(self, labels: object, dim: object) -> object:
        """Convert label indices to one-hot vectors.
        @param labels:
        @param dim:
        @return:
        """
        out: object = torch.zeros(list(labels.size()) + [dim]).to(self.device)
        out.scatter_(len(out.size()) - 1, labels.unsqueeze(-1), 1.0)
        return out

    def classification_loss(self, logit: object, target: object, dataset: object = "CelebA") -> object:
        """Compute binary or softmax cross entropy loss.
        @param logit:
        @param target:
        @param dataset:
        @return:
        """
        if dataset == "CelebA":
            return F.binary_cross_entropy_with_logits(
                logit, target, size_average=False
            ) / logit.size(0)
        elif dataset == "RaFD":
            return F.cross_entropy(logit, target)

    def sample_z(self, batch_size: object) -> object:
        """
        @param batch_size:
        @return:
        """
        return np.random.normal(0, 1, size=(batch_size, self.z_dim))

    def postprocess(self, inputs: object, method: object, temperature: object = 1.0) -> object:
        """
        @param inputs:
        @param method:
        @param temperature:
        @return:
        """
        def listify(x: object) -> object:
            """
            @param x: 
            @return: 
            """
            return x if type(x) == list or type(x) == tuple else [x]

        def delistify(x: object) -> object:
            """
            @param x: 
            @return: 
            """
            return x if len(x) > 1 else x[0]

        if method == "soft_gumbel":
            softmax: List[Any] = [
                F.gumbel_softmax(
                    e_logits.contiguous().view(-1, e_logits.size(-1)) / temperature,
                    hard=False,
                ).view(e_logits.size())
                for e_logits in listify(inputs)
            ]
        elif method == "hard_gumbel":
            softmax = [
                F.gumbel_softmax(
                    e_logits.contiguous().view(-1, e_logits.size(-1)) / temperature,
                    hard=True,
                ).view(e_logits.size())
                for e_logits in listify(inputs)
            ]
        else:
            softmax = [
                F.softmax(e_logits / temperature, -1) for e_logits in listify(inputs)
            ]

        return [delistify(e) for e in (softmax)]

    def reward(self, mols: object) -> object:
        """
        @param mols:
        @return:
        """
        rr: float = 1.0
        m: str
        for m in ("logp,sas,qed,unique" if self.metric == "all" else self.metric).split(
            ","
        ):

            if m == "np":
                rr *= MolecularMetrics.natural_product_scores(mols, norm=True)
            elif m == "logp":
                rr *= MolecularMetrics.water_octanol_partition_coefficient_scores(
                    mols, norm=True
                )
            elif m == "sas":
                rr *= MolecularMetrics.synthetic_accessibility_score_scores(
                    mols, norm=True
                )
            elif m == "qed":
                rr *= MolecularMetrics.quantitative_estimation_druglikeness_scores(
                    mols, norm=True
                )
            elif m == "novelty":
                rr *= MolecularMetrics.novel_scores(mols, data)
            elif m == "dc":
                rr *= MolecularMetrics.drugcandidate_scores(mols, data)
            elif m == "unique":
                rr *= MolecularMetrics.unique_scores(mols)
            elif m == "diversity":
                rr *= MolecularMetrics.diversity_scores(mols, data)
            elif m == "validity":
                rr *= MolecularMetrics.valid_scores(mols)
            else:
                raise RuntimeError("{} is not defined as a metric".format(m))

        return rr.reshape(-1, 1)

    def train(self) -> object:
        """ Trains the model """
        # Learning rate cache for decaying.
        g_lr: object = self.g_lr
        d_lr: object = self.d_lr

        # Start training from scratch or resume training.
        start_iters: int = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print("Start training...")
        start_time: float = time.time()
        i: int
        for i in range(start_iters, self.num_iters):
            if (i + 1) % self.log_step == 0:
                mols, _, _, a, x, _, _, _, _ = self.data.next_validation_batch()
                z: object = self.sample_z(a.shape[0])
                print("[Valid]", "")
            else:
                mols, _, _, a, x, _, _, _, _ = self.data.next_train_batch(
                    self.batch_size
                )
                z = self.sample_z(self.batch_size)

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Adjacency.
            a = torch.from_numpy(a).to(self.device).long()
            x = torch.from_numpy(x).to(self.device).long()  # Nodes.
            a_tensor: object = self.label2onehot(a, self.b_dim)
            x_tensor: object = self.label2onehot(x, self.m_dim)
            z = torch.from_numpy(z).to(self.device).float()

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            logits_real: object
            features_real: object
            logits_real, features_real = self.D(a_tensor, None, x_tensor)
            d_loss_real: object = -torch.mean(logits_real)

            # Compute loss with fake images.
            edges_logits: object
            nodes_logits: object
            edges_logits, nodes_logits = self.G(z)
            # Postprocess with Gumbel softmax
            edges_hat: object
            nodes_hat: object
            (edges_hat, nodes_hat) = self.postprocess(
                (edges_logits, nodes_logits), self.post_method
            )
            logits_fake: object
            features_fake: object
            logits_fake, features_fake = self.D(edges_hat, None, nodes_hat)
            d_loss_fake: object = torch.mean(logits_fake)

            # Compute loss for gradient penalty.
            eps: object = torch.rand(logits_real.size(0), 1, 1, 1).to(self.device)
            x_int0: object = (eps * a_tensor + (1.0 - eps) * edges_hat).requires_grad_(True)
            x_int1: object = (
                eps.squeeze(-1) * x_tensor + (1.0 - eps.squeeze(-1)) * nodes_hat
            ).requires_grad_(True)
            grad0: object
            grad1: object
            grad0, grad1 = self.D(x_int0, None, x_int1)
            d_loss_gp: object = self.gradient_penalty(grad0, x_int0) + self.gradient_penalty(
                grad1, x_int1
            )

            # Backward and optimize.
            d_loss: object = d_loss_fake + d_loss_real + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss: Dict[str, Any] = {}
            loss["D/loss_real"] = d_loss_real.item()
            loss["D/loss_fake"] = d_loss_fake.item()
            loss["D/loss_gp"] = d_loss_gp.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i + 1) % self.n_critic == 0:
                # Z-to-target
                edges_logits, nodes_logits = self.G(z)
                # Postprocess with Gumbel softmax
                (edges_hat, nodes_hat) = self.postprocess(
                    (edges_logits, nodes_logits), self.post_method
                )
                logits_fake, features_fake = self.D(edges_hat, None, nodes_hat)
                g_loss_fake = -torch.mean(logits_fake)

                # Real Reward
                rewardR = torch.from_numpy(self.reward(mols)).to(self.device)
                # Fake Reward
                (edges_hard, nodes_hard) = self.postprocess(
                    (edges_logits, nodes_logits), "hard_gumbel"
                )
                edges_hard, nodes_hard = (
                    torch.max(edges_hard, -1)[1],
                    torch.max(nodes_hard, -1)[1],
                )
                mols = [
                    self.data.matrices2mol(
                        n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True
                    )
                    for e_, n_ in zip(edges_hard, nodes_hard)
                ]
                rewardF = torch.from_numpy(self.reward(mols)).to(self.device)

                # Value loss
                value_logit_real, _ = self.V(a_tensor, None, x_tensor, torch.sigmoid)
                value_logit_fake, _ = self.V(edges_hat, None, nodes_hat, torch.sigmoid)
                g_loss_value = torch.mean(
                    (value_logit_real - rewardR) ** 2
                    + (value_logit_fake - rewardF) ** 2
                )
                # rl_loss= -value_logit_fake
                # f_loss = (torch.mean(features_real, 0) - torch.mean(features_fake, 0)) ** 2

                # Backward and optimize.
                g_loss = g_loss_fake + g_loss_value
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss["G/loss_fake"] = g_loss_fake.item()
                loss["G/loss_value"] = g_loss_value.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i + 1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(
                    et, i + 1, self.num_iters
                )

                # Log update
                # 'mols' is output of Fake Reward
                m0: dict
                m1: Dict[str, Union[int, float]]
                m0, m1 = all_scores(mols, self.data, norm=True)
                m0 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m0.items()}
                m0.update(m1)
                loss.update(m0)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i + 1)

            # Save model checkpoints.
            if (i + 1) % self.model_save_step == 0:
                G_path: Union[bytes, str] = os.path.join(self.model_save_dir, "{}-G.ckpt".format(i + 1))
                D_path: Union[bytes, str] = os.path.join(self.model_save_dir, "{}-D.ckpt".format(i + 1))
                V_path: Union[bytes, str] = os.path.join(self.model_save_dir, "{}-V.ckpt".format(i + 1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                torch.save(self.V.state_dict(), V_path)
                print("Saved model checkpoints into {}...".format(self.model_save_dir))

            # Decay learning rates.
            if (i + 1) % self.lr_update_step == 0 and (i + 1) > (
                self.num_iters - self.num_iters_decay
            ):
                g_lr -= self.g_lr / float(self.num_iters_decay)
                d_lr -= self.d_lr / float(self.num_iters_decay)
                self.update_lr(g_lr, d_lr)
                print("Decayed learning rates, g_lr: {}, d_lr: {}.".format(g_lr, d_lr))

    def test(self):
        """ For testing model """
        # Load the trained generator.
        self.restore_model(self.test_iters)

        with torch.no_grad():
            mols, _, _, a, x, _, _, _, _ = self.data.next_test_batch()
            z = self.sample_z(a.shape[0])

            # Z-to-target
            edges_logits, nodes_logits = self.G(z)
            # Postprocess with Gumbel softmax
            (edges_hat, nodes_hat) = self.postprocess(
                (edges_logits, nodes_logits), self.post_method
            )
            logits_fake, features_fake = self.D(edges_hat, None, nodes_hat)
            g_loss_fake = -torch.mean(logits_fake)

            # Fake Reward
            (edges_hard, nodes_hard) = self.postprocess(
                (edges_logits, nodes_logits), "hard_gumbel"
            )
            edges_hard, nodes_hard = (
                torch.max(edges_hard, -1)[1],
                torch.max(nodes_hard, -1)[1],
            )
            mols = [
                self.data.matrices2mol(
                    n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True
                )
                for e_, n_ in zip(edges_hard, nodes_hard)
            ]

            # Log update
            # 'mols' is output of Fake Reward
            m0, m1 = all_scores(mols, self.data, norm=True)
            m0 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m0.items()}
            m0.update(m1)
            for tag, value in m0.items():
                log += ", {}: {:.4f}".format(tag, value)
