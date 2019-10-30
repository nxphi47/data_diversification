"""

Written by: Xuan-Phi Nguyen (nxphi47)

"""

import torch
import os
import re
import argparse


def merge_nodup(src_ori, tgt_ori, src_hyps, tgt_hyps, **kwargs):
    sep = ' |||||||| '
    merge = [f'{x}{sep}{y}' for x, y in zip(src_ori, tgt_ori)]
    # ori_merge = set(ori_merge)
    for i, (src, tgt) in enumerate(zip(src_hyps, tgt_hyps)):
        merge += [f'{x}{sep}{y}' for x, y in zip(src, tgt)]

    merge = set(merge)
    out = [x.split(sep) for x in merge]
    print(f'Total size: {len(out)}')
    src = [x[0] for x in out]
    tgt = [x[1] for x in out]

    return src, tgt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='en', type=str)
    parser.add_argument('--tgt', default='de', type=str)
    parser.add_argument('--ori')
    parser.add_argument('--hypos')
    parser.add_argument('--dir')
    parser.add_argument('--out')

    args = parser.parse_args()

    ori_src_f = f'{args.ori}.{args.src}'
    ori_tgt_f = f'{args.ori}.{args.tgt}'
    hypos = [x for x in args.hypos.split(":") if x != ""]

    hypos_src_f = [f'{h}.{args.src}' for h in hypos]
    hypos_tgt_f = [f'{h}.{args.tgt}' for h in hypos]


    def read(fo):
        with open(fo, 'r') as f:
            out = f.read().strip().split('\n')
        return out

    ori_src = read(ori_src_f)
    ori_tgt = read(ori_tgt_f)
    hypos_src = [read(h) for h in hypos_src_f]
    hypos_tgt = [read(h) for h in hypos_tgt_f]
    assert len(hypos_src) == len(hypos_tgt)
    print(f'Merge size: {len(hypos_src)}')

    assert len(ori_src) == len(ori_tgt)
    for i, (hx, hy) in enumerate(zip(hypos_src, hypos_tgt)):
        assert len(hx) == len(hy), f'invalid len {i}'

    src, tgt = merge_nodup(ori_src, ori_tgt, hypos_src, hypos_tgt)
    os.makedirs(args.dir, exist_ok=True)
    src_out = os.path.join(args.dir, f'{args.out}.{args.src}')
    tgt_out = os.path.join(args.dir, f'{args.out}.{args.tgt}')
    print(f'src_out:{src_out}')
    print(f'tgt_out:{tgt_out}')
    with open(src_out, 'w') as f:
        f.write('\n'.join(src))
    with open(tgt_out, 'w') as f:
        f.write('\n'.join(tgt))









