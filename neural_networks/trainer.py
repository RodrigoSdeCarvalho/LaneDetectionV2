import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from neural_networks.loss import discriminative_loss
from neural_networks.enet import Enet
from data.datasets.lane_dataset import TuSimpleDataset
from utils.path import Path
from utils.logger import Logger


def init_weights(model):
    if type(model) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear]:
        torch.nn.init.xavier_uniform_(model.weight)
        if model.bias is not None:
            model.bias.data.fill_(0.01)


def train_enet(from_checkpoint: bool = False, model_name: str = 'enet-10-epochs'):
    # Initialize logger
    logger = Logger()
    
    # Constants
    VGG_MEAN = np.array([103.939, 116.779, 123.68]).astype(np.float32)
    VGG_MEAN = torch.from_numpy(VGG_MEAN).cuda().view([1, 3, 1, 1])
    batch_size = 16
    learning_rate = 1e-3
    num_steps = 2000000
    num_workers = 4
    ckpt_epoch_interval = 10
    val_step_interval = 50
    train_start_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        batch_size *= torch.cuda.device_count()
        logger.info(f"Using {torch.cuda.device_count()} GPUs!")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    logger.info(f"Batch size: {batch_size}")

    # Dataset setup
    data_dir = Path().train_data
    train_set = TuSimpleDataset(data_dir, 'train')
    val_set = TuSimpleDataset(data_dir, 'val')

    num_train = len(train_set)
    num_val = len(val_set)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloaders = {'train': train_loader, 'val': val_loader}
    logger.info(f'Data loaded from {data_dir}')

    # Tensorboard setup
    writer = SummaryWriter(log_dir=Path().get_summary(f'enet-{train_start_time}'))

    # Model setup
    net = Enet()
    net = nn.DataParallel(net)
    net.to(device)

    # Optimizer setup
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    MSELoss = nn.MSELoss()

    # Load checkpoint if provided
    if from_checkpoint:
        checkpoint = torch.load(Path().get_model(model_name))
        net.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step = 0
        epoch = 1
        loss = checkpoint['loss']
        logger.info('Checkpoint loaded.')
    else:
        net.apply(init_weights)
        step = 0
        epoch = 1
        logger.info('Network parameters initialized.')

    # Metrics tracking
    sum_bin_precision_train, sum_bin_precision_val = 0, 0
    sum_bin_recall_train, sum_bin_recall_val = 0, 0
    sum_bin_F1_train, sum_bin_F1_val = 0, 0

    # Training loop
    data_iter = {'train': iter(dataloaders['train']), 'val': iter(dataloaders['val'])}
    for step in range(step, num_steps):
        start_time = time.time()

        phase = 'train'
        net.train()
        if step % val_step_interval == 0:
            phase = 'val'
            net.eval()

        try:
            batch = next(data_iter[phase])
        except StopIteration:
            data_iter[phase] = iter(dataloaders[phase])
            batch = next(data_iter[phase])

            if phase == 'train':
                epoch += 1
                if epoch % ckpt_epoch_interval == 0:
                    ckpt_dir = Path().models
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)
                    ckpt_path = os.path.join(ckpt_dir, f'{model_name}-{epoch}-epochs.pt')
                    torch.save({
                        'epoch': epoch,
                        'step': step,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    }, ckpt_path)

                # Log training metrics
                avg_precision_bin_train = sum_bin_precision_train / num_train
                avg_recall_bin_train = sum_bin_recall_train / num_train
                avg_F1_bin_train = sum_bin_F1_train / num_train
                writer.add_scalar('Epoch_Precision_Bin-TRAIN', avg_precision_bin_train, step)
                writer.add_scalar('Epoch_Recall_Bin-TRAIN', avg_recall_bin_train, step)
                writer.add_scalar('Epoch_F1_Bin-TRAIN', avg_F1_bin_train, step)
                writer.add_text('Epoch', str(epoch), step)
                sum_bin_precision_train = 0
                sum_bin_recall_train = 0
                sum_bin_F1_train = 0

            elif phase == 'val':
                # Log validation metrics
                avg_precision_bin_val = sum_bin_precision_val / num_val
                avg_recall_bin_val = sum_bin_recall_val / num_val
                avg_F1_bin_val = sum_bin_F1_val / num_val
                writer.add_scalar('Epoch_Precision_Bin-VAL', avg_precision_bin_val, step)
                writer.add_scalar('Epoch_Recall_Bin-VAL', avg_recall_bin_val, step)
                writer.add_scalar('Epoch_F1_Bin-VAL', avg_F1_bin_val, step)
                sum_bin_precision_val = 0
                sum_bin_recall_val = 0
                sum_bin_F1_val = 0

        # Prepare data
        inputs = batch['input_tensor'].to(device)
        labels_bin = batch['binary_tensor'].to(device)
        labels_inst = batch['instance_tensor'].to(device)

        optimizer.zero_grad()

        # Forward pass
        embeddings, logit = net(inputs)

        # Compute losses
        preds_bin = torch.argmax(logit, dim=1, keepdim=True)
        preds_bin_expand = preds_bin.view(preds_bin.shape[0] * preds_bin.shape[1] * preds_bin.shape[2] * preds_bin.shape[3])
        labels_bin_expand = labels_bin.view(labels_bin.shape[0] * labels_bin.shape[1] * labels_bin.shape[2])

        # Dynamic loss weighting
        bin_count = torch.bincount(labels_bin_expand)
        bin_prop = bin_count.float() / torch.sum(bin_count)
        weight_bin = torch.tensor(1) / (bin_prop + 0.2)

        # Binary segmentation loss
        CrossEntropyLoss = nn.CrossEntropyLoss(weight=weight_bin)
        loss_bin = CrossEntropyLoss(logit, labels_bin)

        # Discriminative loss
        loss_disc, loss_v, loss_d, loss_r = discriminative_loss(
            embeddings, labels_inst,
            delta_v=0.2, delta_d=1,
            param_var=.5, param_dist=.5,
            param_reg=0.001
        )

        loss = loss_bin + loss_disc * 0.01

        # Backward pass
        if phase == 'train':
            loss.backward()
            optimizer.step()

        # Calculate metrics
        bin_TP = torch.sum((preds_bin_expand.detach() == labels_bin_expand.detach()) & (preds_bin_expand.detach() == 1))
        bin_precision = bin_TP.double() / (torch.sum(preds_bin_expand.detach() == 1).double() + 1e-6)
        bin_recall = bin_TP.double() / (torch.sum(labels_bin_expand.detach() == 1).double() + 1e-6)
        bin_F1 = 2 * bin_precision * bin_recall / (bin_precision + bin_recall)

        step_time = time.time() - start_time

        # Log metrics and save to JSON
        metrics = {
            'epoch': epoch,
            'step': step,
            'phase': phase,
            'total_loss': loss.item(),
            'binary_loss': loss_bin.item(),
            'discriminative_loss': loss_disc.item(),
            'variational_loss': loss_v.item(),
            'distance_loss': loss_d.item(),
            'regularization_loss': loss_r.item(),
            'binary_precision': bin_precision.item(),
            'binary_recall': bin_recall.item(),
            'binary_f1': bin_F1.item(),
            'learning_rate': learning_rate,
            'step_time': step_time
        }

        if phase == 'train':
            step += 1
            sum_bin_precision_train += bin_precision.detach() * preds_bin.shape[0]
            sum_bin_recall_train += bin_recall.detach() * preds_bin.shape[0]
            sum_bin_F1_train += bin_F1.detach() * preds_bin.shape[0]

            writer.add_scalar('learning_rate', learning_rate, step)
            writer.add_scalar('total_train_loss', loss.item(), step)
            writer.add_scalar('bin_train_loss', loss_bin.item(), step)
            writer.add_scalar('bin_train_F1', bin_F1, step)
            writer.add_scalar('disc_train_loss', loss_disc.item(), step)

            logger.info(f'{train_start_time}  \nEpoch:{epoch}  Step:{step}  '
                  f'TrainLoss:{loss.item():.5f}  Bin_Loss:{loss_bin.item():.5f}  '
                  f'BinRecall:{bin_recall.item():.5f}  BinPrec:{bin_precision.item():.5f}  '
                  f'F1:{bin_F1.item():.5f}  DiscLoss:{loss_disc.item():.5f}  '
                  f'vLoss:{loss_v.item():.5f}  dLoss:{loss_d.item():.5f}  '
                  f'rLoss:{loss_r.item():.5f}  Time:{step_time:.2f}')

        elif phase == 'val':
            sum_bin_precision_val += bin_precision.detach() * preds_bin.shape[0]
            sum_bin_recall_val += bin_recall.detach() * preds_bin.shape[0]
            sum_bin_F1_val += bin_F1.detach() * preds_bin.shape[0]

            writer.add_scalar('total_val_loss', loss.item(), step)
            writer.add_scalar('bin_val_loss', loss_bin.item(), step)
            writer.add_scalar('bin_val_F1', bin_F1, step)
            writer.add_scalar('disc_val_loss', loss_disc.item(), step)

            logger.info(f'\n{train_start_time}  \nEpoch:{epoch}  Step:{step}  '
                  f'ValidLoss:{loss.item():.5f}  BinLoss:{loss_bin.item():.5f}  '
                  f'BinRecall:{bin_recall.item():.5f}  BinPrec:{bin_precision.item():.5f}  '
                  f'F1:{bin_F1.item():.5f}  DiscLoss:{loss_disc.item():.5f}  '
                  f'vLoss:{loss_v.item():.5f}  dLoss:{loss_d.item():.5f}  '
                  f'rLoss:{loss_r.item():.5f}  Time:{step_time:.2f}')

            # Save sample images to tensorboard
            num_images = 3
            inputs_images = (inputs + VGG_MEAN / 255.)[:num_images, [2, 1, 0], :, :]
            writer.add_images('image', inputs_images, step)
            writer.add_images('Bin Pred', preds_bin[:num_images], step)
            labels_bin_img = labels_bin.view(labels_bin.shape[0], 1, labels_bin.shape[1], labels_bin.shape[2])
            writer.add_images('Bin Label', labels_bin_img[:num_images], step)
            embedding_img = F.normalize(embeddings[:num_images], 1, 1) / 2. + 0.5
            writer.add_images('Embedding', embedding_img, step)
