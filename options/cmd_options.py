from argparse import ArgumentParser

class CmdOptions:

    parser = None

    def __init__(self) -> None:
        self.parser = ArgumentParser()
        self.initialize()
        pass

    def initialize(self):
       self.parser.add_argument('-T', '--task', type=str, help='Task to execute') 
       self.parser.add_argument('-A', '--align', action="store_true", help='Align the images')
       
       self.parser.add_argument('-f', '--feature', type=str, help='feature to make')
       self.parser.add_argument('-g', '--gif', action="store_true", help='create latent animation')
       self.parser.add_argument('-i', '--latent-index', type=int, help='index of latent')
       self.parser.add_argument('-l', '--latent-file', type=str, help='filename of latent')
       self.parser.add_argument('-a', '--amount', type=int, help='amount of deviation of the direction modification.')
    
    def parse(self):
       opts = self.parser.parse_args()
       return opts 
    