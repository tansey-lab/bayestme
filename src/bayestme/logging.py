import logging


def configure_logging(args):
    logger = logging.getLogger('bayestme')

    ch = logging.StreamHandler()
    
    if args.verbose:
        ch.setLevel(level=logging.DEBUG)
    else:
        ch.setLevel(level=logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)



def add_logging_args(parser):
    parser.add_argument('-v', '--verbose',
                        default=False,
                        action='store_true',
                        help='Enable verbose logging')