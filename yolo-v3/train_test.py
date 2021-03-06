import os
import sys


sys.path.insert(1, os.path.join(sys.path[0], '..'))


from util import setup_parser
from download import coco128


def train(epochs=100, batch_size=8):
	coco128()


if __name__ == '__main__':
	parser = setup_parser()

	args = parser.parse_args()

	if args.action == 'train':
		train(args.epochs, args.batch)