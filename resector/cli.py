# -*- coding: utf-8 -*-

"""Console script for resector."""
import sys
import click


@click.command()
@click.argument(
    'input-path',
)
def main(input_path):
    """Console script for resector."""
    import resector
    resector.resect(input_path)
    return 0


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    sys.exit(main())  # pragma: no cover
