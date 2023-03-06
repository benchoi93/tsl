import wandb
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from tsl import logger
from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import MetrLA, PemsBay
from tsl.datasets.pems_benchmarks import PeMS03, PeMS04, PeMS07, PeMS08
from tsl.experiment import Experiment
from tsl.engines import Predictor
from tsl.metrics import torch as torch_metrics, numpy as numpy_metrics
from tsl.nn import models
from tsl.utils.casting import torch_to_numpy
from tsl.engines.better_loss import batch_opt


def get_model_class(model_str):
    if model_str == 'dcrnn':
        model = models.DCRNNModel
    elif model_str == 'stcn':
        model = models.STCNModel
    elif model_str == 'gwnet':
        model = models.GraphWaveNetModel
    elif model_str == 'ar':
        model = models.ARModel
    elif model_str == 'var':
        model = models.VARModel
    elif model_str == 'rnn':
        model = models.RNNModel
    elif model_str == 'fcrnn':
        model = models.FCRNNModel
    elif model_str == 'tcn':
        model = models.TCNModel
    elif model_str == 'transformer':
        model = models.TransformerModel
    elif model_str == 'gatedgn':
        model = models.GatedGraphNetworkModel
    elif model_str == 'evolvegcn':
        model = models.EvolveGCNModel
    elif model_str == 'grugcn':
        model = models.GRUGCNModel
    elif model_str == 'agcrn':
        model = models.AGCRNModel
    else:
        raise NotImplementedError(f'Model "{model_str}" not available.')
    return model


def get_dataset(dataset_name):
    if dataset_name == 'la':
        dataset = MetrLA(impute_zeros=True)
    elif dataset_name == 'bay':
        dataset = PemsBay()
    elif dataset_name == 'pems3':
        dataset = PeMS03()
    elif dataset_name == 'pems4':
        dataset = PeMS04()
    elif dataset_name == 'pems7':
        dataset = PeMS07()
    elif dataset_name == 'pems8':
        dataset = PeMS08()
    else:
        raise ValueError(f"Dataset {dataset_name} not available.")
    return dataset


def run_traffic(cfg: DictConfig):
    wandb.init(project="tsl_traffic", config=dict(
        model=cfg.model.name,
        **cfg.model.hparams,
        **cfg.batch_opt,
    )
    )

    cfg.batch_opt.train_L_batch = True if wandb.config.train_L_batch == 1 else False
    cfg.batch_opt.train_L_space = True if wandb.config.train_L_space == 1 else False
    cfg.batch_opt.train_L_time = True if wandb.config.train_L_time == 1 else False
    cfg.batch_opt.batch_delay = wandb.config.batch_delay
    print(
        f"train_L_batch: {cfg.batch_opt.train_L_batch}, train_L_space: {cfg.batch_opt.train_L_space}, train_L_time: {cfg.batch_opt.train_L_time}, delay: {cfg.batch_opt.batch_delay}"
    )

    ########################################
    # data module                          #
    ########################################
    dataset = get_dataset(cfg.dataset.name)

    # encode time of the day and use it as exogenous variable
    covariates = {'u': dataset.datetime_encoded('day').values}

    # get adjacency matrix
    adj = dataset.get_connectivity(**cfg.dataset.connectivity)

    torch_dataset = SpatioTemporalDataset(target=dataset.dataframe(),
                                          mask=dataset.mask,
                                          connectivity=adj,
                                          covariates=covariates,
                                          horizon=cfg.horizon,
                                          window=cfg.window,
                                          stride=cfg.stride,
                                          batch_delay=cfg.batch_opt.batch_delay,)
    transform = {
        'target': StandardScaler(axis=(0, 1, 2))
    }

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=transform,
        splitter=dataset.get_splitter(**cfg.dataset.splitting),
        batch_size=cfg.batch_size//cfg.batch_opt.batch_delay,
        workers=cfg.workers
    )
    dm.setup()

    ########################################
    # predictor                            #
    ########################################

    model_cls = get_model_class(cfg.model.name)

    model_kwargs = dict(n_nodes=torch_dataset.n_nodes,
                        input_size=1,
                        output_size=1,
                        # input_size=torch_dataset.n_channels,
                        # output_size=torch_dataset.n_channels,
                        horizon=torch_dataset.horizon,
                        exog_size=torch_dataset.input_map.u.shape[-2])

    model_cls.filter_model_args_(model_kwargs)
    model_kwargs.update(cfg.model.hparams)

    # loss_fn = torch_metrics.MaskedMAE(compute_on_step=True)
    #
    # loss_fn = torch_metrics.MaskedMSE(compute_on_step=True)
    #
    loss_fn = batch_opt(num_nodes=torch_dataset.n_nodes,
                        pred_len=torch_dataset.horizon,
                        delay=cfg.batch_opt.batch_delay,
                        train_L_batch=cfg.batch_opt.train_L_batch,
                        train_L_space=cfg.batch_opt.train_L_space,
                        train_L_time=cfg.batch_opt.train_L_time,)

    log_metrics = {'mae': torch_metrics.MaskedMAE(),
                   'mse': torch_metrics.MaskedMSE(),
                   'mape': torch_metrics.MaskedMAPE(),
                   'mae_at_15': torch_metrics.MaskedMAE(at=2),  # 3rd is 15 min
                   'mae_at_30': torch_metrics.MaskedMAE(at=5),  # 6th is 30 min
                   'mae_at_60': torch_metrics.MaskedMAE(at=11)}  # 12th is 1 h
    if cfg.lr_scheduler is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler, cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
    else:
        scheduler_class = scheduler_kwargs = None

    # setup predictor
    predictor = Predictor(
        model_class=model_cls,
        model_kwargs=model_kwargs,
        optim_class=getattr(torch.optim, cfg.optimizer.name),
        optim_kwargs=dict(cfg.optimizer.hparams),
        loss_fn=loss_fn,
        metrics=log_metrics,
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs
    )

    ########################################
    # logging options                      #
    ########################################

    # add tags
    tags = list(cfg.tags) + [cfg.model.name, cfg.dataset.name]

    if 'wandb' in cfg:
        exp_logger = WandbLogger(name=cfg.run.name,
                                 save_dir=cfg.run.dir,
                                 offline=cfg.wandb.offline,
                                 project=cfg.wandb.project)
    else:
        exp_logger = TensorBoardLogger(save_dir=cfg.run.dir,
                                       name=f'{cfg.run.name}_{"_".join(tags)}')

    ########################################
    # training                             #
    ########################################

    early_stop_callback = EarlyStopping(
        monitor='val_mae',
        patience=cfg.patience,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.run.dir,
        save_top_k=1,
        monitor='val_mae',
        mode='min',
    )

    trainer = Trainer(max_epochs=cfg.epochs,
                      default_root_dir=cfg.run.dir,
                      logger=exp_logger,
                      gpus=1 if torch.cuda.is_available() else None,
                      gradient_clip_val=cfg.grad_clip_val,
                      callbacks=[early_stop_callback, checkpoint_callback])

    trainer.fit(predictor, datamodule=dm)

    ########################################
    # testing                              #
    ########################################

    predictor.load_model(checkpoint_callback.best_model_path)

    predictor.freeze()
    trainer.test(predictor, datamodule=dm)

    output = trainer.predict(predictor, dataloaders=dm.test_dataloader())
    output = torch_to_numpy(output)
    y_hat, y_true, mask = output['y_hat'], \
        output['y'], \
        output.get('mask', None)

    y_true = y_true.reshape([y_true.shape[0]//cfg.batch_opt.batch_delay, cfg.batch_opt.batch_delay]+list(y_true.shape[1:]))
    y_true = y_true[:, 0, ...]
    y_hat = y_hat[:, :, :, 0, :]
    mask = mask[:, :, :, 0, :]

    res = dict(test_mae=numpy_metrics.mae(y_hat, y_true, mask),
               test_rmse=numpy_metrics.rmse(y_hat, y_true, mask),
               test_mape=numpy_metrics.mape(y_hat, y_true, mask))

    output = trainer.predict(predictor, dataloaders=dm.val_dataloader())
    output = torch_to_numpy(output)
    y_hat, y_true, mask = output['y_hat'], \
        output['y'], \
        output.get('mask', None)
    y_true = y_true.reshape([y_true.shape[0]//cfg.batch_opt.batch_delay, cfg.batch_opt.batch_delay]+list(y_true.shape[1:]))
    y_true = y_true[:, 0, ...]
    y_hat = y_hat[:, :, :, 0, :]
    mask = mask[:, :, :, 0, :]

    res.update(dict(val_mae=numpy_metrics.mae(y_hat, y_true, mask),
                    val_rmse=numpy_metrics.rmse(y_hat, y_true, mask),
                    val_mape=numpy_metrics.mape(y_hat, y_true, mask)))

    return res


if __name__ == '__main__':
    exp = Experiment(run_fn=run_traffic, config_path='config/traffic')
    res = exp.run()
    logger.info(res)
