import yaml, os

class ConfigParser:
    def __init__(self, args):
        with open(args.config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        train_cfg_dir = os.path.join(os.path.dirname(args.config), self.config['_TRAIN_'])
        self.merge_from_yaml(train_cfg_dir)

        for arg in args.__dict__:
            self.config[arg] = args.__dict__[arg]

        self.config['model']['backbone']['input_size'] = self.config['data_loader']['input_size']
        self.config['model']['RPN']['input_size'] = self.config['data_loader']['input_size']

    def __getitem__(self, name):
        return self.config[name]

    def get_model_name(self):
        backbone_type = self.config['model']['backbone']['backbone_type']
        FPN_mode = '_fpn' if 'FPN' in self.config['model']['backbone'] else ''
        return backbone_type + FPN_mode

    def merge_from_yaml(self, dir):
        with open(dir) as f:
            tmp_config = yaml.load(f, Loader=yaml.FullLoader)
        for key in tmp_config:
            self.config[key] = tmp_config[key]


if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', default=None, type=str)
    args.add_argument('-d', '--device', default=None, type=str)
    args.add_argument('-r', '--resume', action='store_true')
    
    args = args.parse_args()

    args.config = "./conf/resnet_cfg.yaml"

    cp = ConfigParser(args)
