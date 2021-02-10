import argparse


def setup_parser():
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers(required=True)

	train_parser = subparsers.add_parser('train', help='train help')
	test_parser = subparsers.add_parser('test', help='test help')

	train_parser.add_argument('--epochs', type=int, default=100, help='epochs')
	train_parser.add_argument('batch', type=int, default=8, help='batch size')
	train_parser.add_argument(
		"--verbose",
		"-v",
		default=False,
		action='store_true'
	)

	return parser