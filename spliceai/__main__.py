import argparse
import sys
import pysam
from spliceai.utils import annotator, get_delta_scores
import logging

logger = logging.getLogger("SpliceAI")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
       '[%(asctime)s] %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# Account for python2/python3 differences
try:
    from sys.stdin import buffer as std_in
    from sys.stdout import buffer as std_out
except ImportError:
    from sys import stdin as std_in
    from sys import stdout as std_out


def get_options():

    parser = argparse.ArgumentParser()
    parser.add_argument('-I', nargs='?', default=std_in,
        help='path to the input VCF file, defaults to standard in')
    parser.add_argument('-O', nargs='?', default=std_out,
        help='path to the output VCF file, defaults to standard out')
    parser.add_argument('-R', required=True,
        help='path to the genome fasta file')
    parser.add_argument('-A',
        help='path to the gene annotations file, defaults to file in package')
    args = parser.parse_args()

    try:
        args.I = open(args.I, 'rt')
    except TypeError:
        pass

    try:
        args.O = open(args.O, 'wt')
    except TypeError:
        pass

    return args


def main():
    args = get_options()
    buffer_size = 10000 #TODO - make argument
    vcf = pysam.VariantFile(args.I)
    header = vcf.header
    header.add_line('##INFO=<ID=SpliceAI,Number=.,Type=String,Description="SpliceAI variant annotation. These include delta scores (DS) and delta positions (DP) for acceptor gain (AG), acceptor loss (AL), donor gain (DG), and donor loss (DL). Format: ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL">')
    output = pysam.VariantFile(args.O, mode='w', header=header)
    ann = annotator(args.R, args.A)
    cache = []
    for record in vcf:
        cache.append(record)
        if len(cache) >= buffer_size:
            process_cache(cache, output, ann)
            cache = []
    if cache:
        process_cache(cache, output, ann)

def process_cache(cache, output, ann):
    logger.info("Processing cache of {} variants in range".format(len(cache)) +
                " {}:{} - {}:{}".format(cache[0].chrom, cache[0].pos,
                                        cache[-1].chrom, cache[-1].pos))
    for record in get_delta_scores(cache, ann):
        output.write(record)

if __name__ == '__main__':
    main()

