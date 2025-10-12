def main():
    import argparse
    import importlib
    from importlib import metadata
    import pkgutil
    import prismpyp.commands

    from prismpyp.utils import parse_args

    # create the top-level parser that allows users to type `prismpyp [command] ...` at the command line
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(title='Choose a sub-command', required=False)

    # enumerate the list of subcommands that can be called via `prismpyp [command] ...`
    subcommand_names = [f'prismpyp.commands.{modname}' for importer, modname, ispkg in pkgutil.iter_modules(prismpyp.commands.__path__)]

    for subcommand_name in subcommand_names:
        
        # create the parser for the subcommand
        subcommand_parser = subparsers.add_parser(name=f'{subcommand_name.split(".")[-1]}')
        # import the subcommand module
        subcommand_module = importlib.import_module(subcommand_name)
        # add the arg options of the subcommand module
        if f'{subcommand_name.split(".")[-1]}' in {'train','eval2d','eval3d'}:
            parse_args.add_args(subcommand_parser)
        else:
            subcommand_module.add_args(subcommand_parser)
        # define the entry point to the subcommand script
        subcommand_parser.set_defaults(func=subcommand_module.main)

    # parse all args, including defining which sub-command to execute
    args = parser.parse_args()
    if hasattr(args, 'func'):
        # execute the sub-command
        args.func(args)
    else:
        # no identifiable function given to tomodrgn; return help instead
        parser.print_help()

if __name__ == '__main__':
    main()