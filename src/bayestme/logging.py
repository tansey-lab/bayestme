import logging


def configure_logging(args):
    logger = logging.getLogger('bayestme')

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)



def add_logging_args(parser):
    parser.add_argument('-v', '--verbose',
                        default=False,
                        action='store_true',
                        help='Enable verbose logging')