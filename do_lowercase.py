import argparse
from time import time

def do_lowercase(input, output):

    f = open(args.input).read()
    print('done reading & started writing')

    w = open(args.output, 'w')
    w.write(f.lower())
    w.close()
    
    return 0

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    default=None,
    type=str,
    required=True,
    help="Input path to be read and lowercased",
)
parser.add_argument(
    "--output",
    default=None,
    type=str,
    required=True,
    help="Output path to be written",
)
args = parser.parse_args()

# started lowercasing
st = time()
do_lowercase(args.input, args.output)
print(time() - st)