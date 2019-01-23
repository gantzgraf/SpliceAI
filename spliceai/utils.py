from pkg_resources import resource_filename
import pandas as pd
import numpy as np
import re
import pyfasta
from keras.models import load_model
from collections import defaultdict
import operator

class annotator():

    def __init__(self, ref_fasta, annotations):

        if annotations is None:
            annotations = resource_filename(__name__, 'annotations/GENCODE.v24lift37')
        df = pd.read_csv(annotations, sep='\t')

        self.genes = df['#NAME'].get_values()
        self.chroms = df['CHROM'].get_values()
        self.strands = df['STRAND'].get_values()
        self.tx_starts = df['TX_START'].get_values()+1
        self.tx_ends = df['TX_END'].get_values()

        self.ref_fasta = pyfasta.Fasta(ref_fasta)

        paths = ('models/spliceai{}.h5'.format(x) for x in range(1, 6))
        self.models = [load_model(resource_filename(__name__, x)) for x in paths]

    def get_name_and_strand(self, chrom, pos):

        idxs = np.intersect1d(
                   np.nonzero(self.chroms == chrom)[0],
                   np.intersect1d(np.nonzero(self.tx_starts <= pos)[0],
                   np.nonzero(pos <= self.tx_ends)[0]))

        if len(idxs) >= 1:
            return (self.genes[idxs], self.strands[idxs], idxs)
        else:
            return ([], [], [])

    def get_pos_data(self, idx, pos):

        dist_tx_start = self.tx_starts[idx]-pos
        dist_tx_end = self.tx_ends[idx]-pos

        dist = (dist_tx_start, dist_tx_end)

        return dist


def one_hot_encode(seq):

    MAP = np.asarray([[0, 0, 0, 0],
                      [1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

    seq = seq.upper().replace('A', '\x01').replace('C', '\x02')
    seq = seq.replace('G', '\x03').replace('T', '\x04').replace('N', '\x00')

    return MAP[np.fromstring(seq, np.int8) % 5]


def get_delta_scores(records, ann, L=1001):

    W = 10000+L
    r = 0 #record index
    s = 0 #score index
    scored_genes = []
    scored_strands = []
    scored_alts = []
    scored_ref_lens = []
    scored_alt_lens = []
    scored_del_lens = []
    ref_to_score = []
    alt_to_score = []
    rec_to_scores = defaultdict(list)
    delta_scores = []
    for record in records:
        (genes, strands, idxs) = ann.get_name_and_strand(record.chrom, record.pos)
        for j in range(len(record.alts)):
            if len(idxs) == 0:
                delta_scores.append("{}|.|.|.|.|.|.|.|.|.".format(record.alts[j]))
                rec_to_scores[r].append(s)
                s += 1
            for i in range(len(idxs)):
                dist = ann.get_pos_data(idxs[i], record.pos)

                if len(record.ref) > 1 and len(record.alts[j]) > 1:
                    delta_scores.append("{}|.|.|.|.|.|.|.|.|.".format(record.alts[j]))
                    rec_to_scores[r].append(s)
                    s += 1
                    continue
                # Ignoring complicated INDELs
                
                pad_size = [max(W//2+dist[0], 0), max(W//2-dist[1], 0)]
                ref_len = len(record.ref)
                alt_len = len(record.alts[j])
                del_len = max(ref_len-alt_len, 0)
                
                seq = ann.ref_fasta[record.chrom][
                                    record.pos-W//2-1:record.pos+W//2]
                x_ref = 'N'*pad_size[0]+seq[pad_size[0]:W-pad_size[1]]\
                         +'N'*pad_size[1]
                x_alt = x_ref[:W//2]+str(record.alts[j])+x_ref[W//2+ref_len:]

                X_ref = one_hot_encode(x_ref)[None, :]
                X_alt = one_hot_encode(x_alt)[None, :]

                if strands[i] == '-':
                    X_ref = X_ref[:, ::-1, ::-1]
                    X_alt = X_alt[:, ::-1, ::-1]
                ref_to_score.append(X_ref)
                alt_to_score.append(X_alt)
                scored_ref_lens.append(ref_len)
                scored_alt_lens.append(alt_len)
                scored_del_lens.append(del_len)
                scored_strands.append(strands[i])
                scored_genes.append(genes[i])
                scored_alts.append(record.alts[j])
                rec_to_scores[r].append(s)
                s += 1
    #PROBLEM - this works only if all our np arrays are the same dimensions
    #          we need to split into batches with equal dimensions (i.e. 
    #          len(ref_to_score[n][0]) == len(ref_to_score[m][0]) ... )

    ref_Y0s = ann.models[0].predict_generator((x for x in ref_to_score),
                                              steps=len(ref_to_score))
    ref_Y1s = ann.models[1].predict_generator((x for x in ref_to_score),
                                              steps=len(ref_to_score))
    ref_Y2s = ann.models[2].predict_generator((x for x in ref_to_score),
                                              steps=len(ref_to_score))
    ref_Y3s = ann.models[3].predict_generator((x for x in ref_to_score),
                                              steps=len(ref_to_score))
    ref_Y4s = ann.models[4].predict_generator((x for x in ref_to_score),
                                              steps=len(ref_to_score))
    alt_Y0s = ann.models[0].predict_generator((x for x in alt_to_score),
                                              steps=len(alt_to_score))
    alt_Y1s = ann.models[1].predict_generator((x for x in alt_to_score),
                                              steps=len(alt_to_score))
    alt_Y2s = ann.models[2].predict_generator((x for x in alt_to_score),
                                              steps=len(alt_to_score))
    alt_Y3s = ann.models[3].predict_generator((x for x in alt_to_score),
                                              steps=len(alt_to_score))
    alt_Y4s = ann.models[4].predict_generator((x for x in alt_to_score),
                                              steps=len(alt_to_score))
    for i in range(len(ref_to_score)):
        Y_ref = (ref_Y0s[i]+ref_Y1s[i]+ref_Y2s[i]+ref_Y3s[i]+ref_Y4s[i])/5
        Y_alt = (alt_Y0s[i]+alt_Y1s[i]+alt_Y2s[i]+alt_Y3s[i]+alt_Y4s[i])/5
        if scored_strands[i] == '-':
            Y_ref = Y_ref[:, ::-1]
            Y_alt = Y_alt[:, ::-1]
        ref_len = scored_ref_lens[i]
        alt_len = scored_alt_lens[i]
        del_len = scored_del_lens[i]
        if ref_len > 1 and alt_len == 1:
            Y_alt = np.concatenate([Y_alt[:, :L//2+alt_len],
                                    np.zeros((1, del_len, 3)),
                                    Y_alt[:, L//2+alt_len:]], axis=1)
        elif ref_len == 1 and alt_len > 1:
            Y_alt = np.concatenate([
                Y_alt[:, :L//2],
                np.max(Y_alt[:, L//2:L//2+alt_len], axis=1)[:, None, :],
                Y_alt[:, L//2+alt_len:]], axis=1)
        Y = np.concatenate([Y_ref, Y_alt])

        idx_pA = (Y[1, :, 1]-Y[0, :, 1]).argmax()
        idx_nA = (Y[0, :, 1]-Y[1, :, 1]).argmax()
        idx_pD = (Y[1, :, 2]-Y[0, :, 2]).argmax()
        idx_nD = (Y[0, :, 2]-Y[1, :, 2]).argmax()

        delta_scores.append(
            "{}|{}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{}|{}|{}|{}".format(
            scored_alts[i],
            scored_genes[i],
            Y[1, idx_pA, 1]-Y[0, idx_pA, 1],
            Y[0, idx_nA, 1]-Y[1, idx_nA, 1],
            Y[1, idx_pD, 2]-Y[0, idx_pD, 2],
            Y[0, idx_nD, 2]-Y[1, idx_nD, 2],
            idx_pA-L//2,
            idx_nA-L//2,
            idx_pD-L//2,
            idx_nD-L//2))
    for i in range(len(records)):
        score_indices = rec_to_scores[i]
        scores = operator.itemgetter(score_indices)(delta_scores)
        if len(scores) > 0:
            records[i].info['SpliceAI'] = scores
    return records

