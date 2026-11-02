"""Public command line for the AppWorld release."""
import argparse
import json
from .config import ROOT
from .pool import HarnessPool


def main():
    parser = argparse.ArgumentParser(prog='harness-s')
    sub = parser.add_subparsers(dest='action',required=True)
    for action in ('prepare','train'):
        item = sub.add_parser(action)
        item.add_argument('--config',required=True)
        item.add_argument('--run-dir',required=True)
        if action=='train':
            item.add_argument('--dry-run',action='store_true')
    item = sub.add_parser('evaluate')
    item.add_argument('--run-dir',required=True)
    item.add_argument('--checkpoint',choices=('best','final'),default='best')
    item.add_argument('--harness',choices=('h0','rank1'),default='h0')
    item.add_argument('--dry-run',action='store_true')
    sub.add_parser('check-pool')
    args = parser.parse_args()
    if args.action=='check-pool':
        print(json.dumps({'harnesses':HarnessPool(ROOT).verify()}))
        return
    if args.action=='evaluate':
        from .launch import launch
        launch(args.run_dir,args.dry_run,action='test',checkpoint=args.checkpoint,harness=args.harness)
        return
    from .prepare import prepare
    run = prepare(args.config,args.run_dir)
    if args.action=='prepare':
        print(run)
    else:
        from .launch import launch
        launch(run,dry_run=args.dry_run)


if __name__=='__main__':
    main()
