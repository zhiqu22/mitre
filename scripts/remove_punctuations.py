import argparse
from string import punctuation

def len_no_punc(s, punc):
    return len([ch for ch in s if ch in punc])

def filter_overpunc(len_npunc, len_sen):
    return len_npunc < 0.5 * len_sen

def main(args):
    punc = punctuation + "—|–"
    print(f'Processing files {args.src_file} and {args.tgt_file}')
    
    with open(args.src_file, 'rt', encoding=args.encoding) as fsrc, \
         open(args.tgt_file, 'rt', encoding=args.encoding) as ftgt, \
         open(args.bitext + '.' + args.src_lang, 'wt', encoding=args.encoding) as fout_src, \
         open(args.bitext + '.' + args.tgt_lang, 'wt', encoding=args.encoding) as fout_tgt:
        
        for src, tgt in zip(fsrc, ftgt):
            src = src.strip()
            tgt = tgt.strip()
            
            nchar_npunc_src = len_no_punc(src, punc)
            nchar_npunc_tgt = len_no_punc(tgt, punc)
            
            if filter_overpunc(nchar_npunc_src, len(src)) and filter_overpunc(nchar_npunc_tgt, len(tgt)):
                fout_src.write(src + '\n')
                fout_tgt.write(tgt + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-file", required=True, type=str, help="Path to the source language file")
    parser.add_argument("--tgt-file", required=True, type=str, help="Path to the target language file")
    parser.add_argument('--encoding', default='utf-8', help='Character encoding for input/output')
    parser.add_argument('--bitext', type=str, required=True, help='Output file prefix')
    parser.add_argument('--src-lang', type=str, required=True, help='Source language code')
    parser.add_argument('--tgt-lang', type=str, required=True, help='Target language code')
    main(parser.parse_args())