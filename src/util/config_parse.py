import yaml

class ConfigParser:
    def __init__(self, args):
        self.args = args

        with open(args.config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.config['resume'] = args.resume
        self.config['device'] = args.device

    def __getitem__(self, name):
        return self.config[name]
