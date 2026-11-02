"""Script adapter for Slime's Ray job launcher."""
from harness_s.train import train
from slime.utils.arguments import parse_args

if __name__=='__main__':
    train(parse_args())
