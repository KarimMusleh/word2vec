import argparse
from src.train import Trainer

def train(config):
    os.makedirs(config['model_dir'], exist_ok=True)
    device = torch.device

    Trainer(config['skipgram_version'], config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to yaml config')
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    train(config)
