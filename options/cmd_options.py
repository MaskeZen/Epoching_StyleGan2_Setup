from argparse import ArgumentParser

class CmdOptions:

    parser = None

    def __init__(self) -> None:
        self.parser = ArgumentParser()
        self.initialize()
        pass

    def initialize(self):
       self.parser.add_argument('-t', '--task', type=str, help='Task to execute') 
       self.parser.add_argument('-a', '--align', action="store_true", help='Align the images')
       self.parser.add_argument('-d', '--direction', action="store_true", help='run latent direction')
       self.parser.add_argument('-f', '--feature', type=str, help='feature to make')
       self.parser.add_argument('-l', '--latent-index', type=int, help='index of latent')
    
    def parse(self):
       opts = self.parser.parse_args()
       return opts 
    