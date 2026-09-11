"""Register legacy episode references and inspect/select their pinned catalog."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fm_humanoid_bench.evaluation.reference_catalog import build_catalog, load_catalog, reference_options, select_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('create')
    create.add_argument('--bundle', type=Path, action='append', required=True)
    create.add_argument('--output', type=Path, required=True)
    for name in ('list', 'select'):
        command = sub.add_parser(name)
        command.add_argument('--catalog', type=Path, required=True)
        command.add_argument('--task', required=True)
        command.add_argument('--condition', choices=['none', 'state_action', 'images'], required=True)
        if name == 'select':
            command.add_argument('--reference-id')
            command.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'create':
        result = build_catalog(args.bundle, args.output)
    elif args.command == 'list':
        result = reference_options(load_catalog(args.catalog), args.task, args.condition)
    else:
        result = select_reference(load_catalog(args.catalog), task=args.task,
                                  condition=args.condition, reference_id=args.reference_id,
                                  log_path=args.receipt)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
