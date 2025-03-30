from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from relseg import main_lrp_entry
__version__ = "0.0.1"

def main():
    # Create the main argument parser
    parser = main_lrp_entry.argparser()

    # Parse arguments and pass them to lrp.main()
    args = parser.parse_args()
    main_lrp_entry.main(args)

if __name__ == "__main__":
    main()