import torch
from pytorch_lightning import LightningModule, Trainer, loggers
import gpytorch
from boml.kernels import AnovaKernel, MultilinearKernel
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from boml import train, data
from argparse import ArgumentParser


# TODO move to a different module
class GPModel(ApproximateGP):
    def __init__(self, inducing_points, kernel=None):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()

        kernel = kernel or gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPModel_lightning(LightningModule):

    def __init__(self, model, mll):
        super().__init__()
        self.model = model
        self.mll = mll

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = -self.mll(self(x), y)
        self.log("train-loss", loss)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        pred = self(x)
        loss = -self.mll(pred, y)
        mse = torch.nn.MSELoss()
        self.log("test-loss", loss)
        self.log("test-MSE", mse(pred.loc, y))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.mll.likelihood.parameters()},
        ], lr=0.01)


def main(parser):
    args = parser.parse_args()
    train_set, dev_set, test_set = data.load_dataloaders(batch_size=args.batch_size, debug=args.debug)

    bx, by = next(iter(train_set))

    # TODO AD-hoc, taking the first batch as inducing points
    inducing_points, by = next(iter(train_set))
    # This is kinda inefficient, because I initialize them all
    kernel_choices = {'rbf': gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
                      'linear': gpytorch.kernels.LinearKernel(),
                      'anova': AnovaKernel(m=5, s=bx.shape[1]),
                      'multilinear': MultilinearKernel(dim=bx.shape[1])
                      }
    gp = GPModel(inducing_points=inducing_points, kernel=kernel_choices[args.kernel.lower()])
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    # Our loss object. We're using the VariationalELBO
    mll = gpytorch.mlls.VariationalELBO(likelihood, gp, num_data=by.size(0))

    model = GPModel_lightning(gp, mll)
    # Initialize a trainer
    trainer = Trainer(
        gpus=1,
        max_epochs=args.num_epochs,
        progress_bar_refresh_rate=20,
        logger=loggers.TensorBoardLogger(save_dir='lightning_logs',
                                         name=f'{args.kernel}-bs:{args.batch_size}-epochs:{args.num_epochs}')

    )

    trainer.fit(model, train_dataloaders=train_set, val_dataloaders=dev_set)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--debug', default=False)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument("--n-ind-points", type=int, default=12, help='TODO use me')
    parser.add_argument("--kernel", type=str, default="rbf", choices=['rbf', 'anova', 'multilinear', 'linear'])
    # TODO the rest
    main(parser)
