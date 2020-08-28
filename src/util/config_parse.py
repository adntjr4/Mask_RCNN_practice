import yaml

class ConfigParser:
    def __init__(self, args):
        with open(args.config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        for arg in args.__dict__:
            self.config[arg] = args.__dict__[arg]

    def __getitem__(self, name):
        return self.config[name]

    def get_model_name(self):
        return '%s'%(self.config['model']['backbone']['backbone_type'])
