from essentials.add_module import set_working_directory
set_working_directory()

from neural_networks.trainer import train_enet


if __name__ == '__main__':
    model_name = 'enet-10-epochs'
    train_enet(model_name=model_name, from_checkpoint=True)
