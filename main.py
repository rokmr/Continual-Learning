import json
import argparse
from trainer import train
from evaluator import test

def main():
    args = setup_parser().parse_args() #Namespace(config='/home/rohitk/SLCA/exps/slca_cifar.json', test_only=False)
    param = load_json(args.config) #param: {'prefix': 'reproduce', 'dataset': 'cifar100_224', 'memory_size': 0, 'memory_per_class': 0, 'fixed_memory': False, 'shuffle': True, 'init_cls': 10, 'increment': 10, 'model_name': 'slca_cifar', 'model_postfix': '20e', 'convnet_type': 'vit-b-p16', 'device': ['0', '1'], 'seed': [1993, 1996, 1997], 'epochs': 2, 'ca_epochs': 5, 'ca_with_logit_norm': 0.1, 'milestones': [18]}
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json  #args: {'config': '/home/rohitk/SLCA/exps/slca_cifar.json', 'test_only': False, 'prefix': 'reproduce', 'dataset': 'cifar100_224', 'memory_size': 0, 'memory_per_class': 0, 'fixed_memory': False, 'shuffle': True, 'init_cls': 10, 'increment': 10, 'model_name': 'slca_cifar', 'model_postfix': '20e', 'convnet_type': 'vit-b-p16', 'device': ['0', '1'], 'seed': [1993, 1996, 1997], 'epochs': 2, 'ca_epochs': 5, 'ca_with_logit_norm': 0.1, 'milestones': [18]}
    if args['test_only']:
        test(args)
    else:
        train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='/home/rohitk/SLCA/exps/slca_cifar100-5%_buffer500.json',
                        help='Json file of settings.')
    parser.add_argument('--test_only', action='store_true')
    return parser


if __name__ == '__main__':
    main()
