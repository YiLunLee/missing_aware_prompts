import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mmimdb', type=str, help='Datasets.')
parser.add_argument('--root', default='./datasets', type=str, help='Root of datasets')
args = parser.parse_args()

if args.dataset.lower() == 'mmimdb':
    from vilt.utils.write_mmimdb import make_arrow
    make_arrow(f'{args.root}/mmimdb', './datasets/mmimdb')
    
elif args.dataset.lower() == 'food101':
    from vilt.utils.write_food101 import make_arrow
    make_arrow(f'{args.root}/Food101', './datasets/Food101')
    
elif args.dataset.lower() == 'hatefull_memes':
    from vilt.utils.write_hatememes import make_arrow
    make_arrow(f'{args.root}/Hatefull_Memes', './datasets/Hatefull_Memes')