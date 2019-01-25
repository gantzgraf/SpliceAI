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

def _score_batches(refs, alts, ann):
    '''
        Puts refs and alts into lists with arrays of same lengths and
        score in batches. Returns two lists of scores in the same order
        as provided.
    '''
    batches = defaultdict(list)
    scores = defaultdict(list)
    ref_indices = dict()
    alt_indices = dict()
    i = 0
    for i in range(len(refs)):
        l = len(refs[i][0])
        batches[l].append(refs[i])
        ref_indices[i] = (l, len(batches[l]) - 1)
    for i in range(len(alts)):
        l = len(alts[i][0])
        batches[l].append(alts[i])
        alt_indices[i] = (l, len(batches[l]) - 1)
    for l,batch in batches.items():
        Y0s = ann.models[0].predict_generator((x for x in batch),
                                              steps=len(batch))
        Y1s = ann.models[1].predict_generator((x for x in batch),
                                              steps=len(batch))
        Y2s = ann.models[2].predict_generator((x for x in batch),
                                              steps=len(batch))
        Y3s = ann.models[3].predict_generator((x for x in batch),
                                              steps=len(batch))
        Y4s = ann.models[4].predict_generator((x for x in batch),
                                              steps=len(batch))
        scores[l] = list((Y0s[i]+Y1s[i]+Y2s[i]+Y3s[i]+Y4s[i])/5 for i in
                     range(len(Y0s)))
    ref_scores = []
    alt_scores = []
    for i in range(len(refs)):
        l,idx = ref_indices[i]
        ref_scores.append(np.asarray([scores[l][idx]]))
    for i in range(len(alts)):
        l,idx = alt_indices[i]
        alt_scores.append(np.asarray([scores[l][idx]]))
    return ref_scores,alt_scores




def get_delta_scores(records, ann, L=1001):

    W = 10000+L
    r = 0 #record index
    s = 0 #score index
    u = 0 #unscored index
    scored_genes = []
    scored_strands = []
    scored_alts = []
    scored_ref_lens = []
    scored_alt_lens = []
    scored_del_lens = []
    ref_to_score = []
    alt_to_score = []
    rec_to_scores = defaultdict(list)
    rec_to_unscored = defaultdict(list)
    delta_scores = []
    unscored = []
    for record in records:
        r += 1
        (genes, strands, idxs) = ann.get_name_and_strand(record.chrom, record.pos)
        for j in range(len(record.alts)):
            if len(idxs) == 0:
                unscored.append("{}|.|.|.|.|.|.|.|.|.".format(record.alts[j]))
                rec_to_unscored[r-1].append(u)
                u += 1
            for i in range(len(idxs)):
                dist = ann.get_pos_data(idxs[i], record.pos)
                if len(record.ref) > 1 and len(record.alts[j]) > 1:
                    unscored.append("{}|.|.|.|.|.|.|.|.|.".format(record.alts[j]))
                    rec_to_unscored[r-1].append(u)
                    u += 1
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
                rec_to_scores[r-1].append(s)
                s += 1
    #PROBLEM - we can only use precit_generator for np arrays of the same
    #          dimensions.
    #          We need to split into batches with equal dimensions (i.e. 
    #          len(ref_to_score[n][0]) == len(ref_to_score[m][0]) ... )
    Y_refs, Y_alts = _score_batches(ref_to_score, alt_to_score, ann)
    for i in range(len(ref_to_score)):
        Y_ref = Y_refs[i]
        Y_alt = Y_alts[i]
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
        scores = []
        if i in rec_to_scores:
            s = operator.itemgetter(*rec_to_scores[i])(delta_scores)
            if isinstance(s, str):
                scores.append(s)
            else: #if more than one result will be a tuple
                scores.extend(s)
        if i in rec_to_unscored:
            s = operator.itemgetter(*rec_to_unscored[i])(unscored)
            if isinstance(s, str):
                scores.append(s)
            else: #if more than one result will be a tuple
                scores.extend(s)
        if len(scores) > 0:
            records[i].info['SpliceAI'] = scores
    return records

