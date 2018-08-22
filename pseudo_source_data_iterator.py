import numpy

import gzip

import shuffle
from util import load_dict

def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)

class PseudoSourceTextIterator:
    """Bitext iterator processing parallel data,
    as well as parallel data with a pseudo-source."""
    def __init__(self, source, target,
                 pseudo_source, pseudo_target,
                 source_dicts, target_dict,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1,
                 n_words_target=-1,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=True,
                 use_factor=False,
                 maxibatch_size=20,
                 noise=False):
        if shuffle_each_epoch:
            self.source_orig = source
            self.target_orig = target
            self.pseudo_source_orig = pseudo_source
            self.pseudo_target_orig = pseudo_target
            self.source, self.target = shuffle.main([self.source_orig, self.target_orig], temporary=True)
            self.pseudo_source, self.pseudo_target = shuffle.main([self.pseudo_source_orig, self.pseudo_target_orig], temporary=True)
        else:
            self.source = fopen(source, 'r')
            self.target = fopen(target, 'r')
            self.pseudo_source = fopen(pseudo_source, 'r')
            self.pseudo_target = fopen(pseudo_target, 'r')
        self.source_dicts = []
        for source_dict in source_dicts:
            self.source_dicts.append(load_dict(source_dict))
        self.target_dict = load_dict(target_dict)

        # batch size in divided by two: one part
        # is for parallel data, the other is for
        # the pseudo-source data.
        self.batch_size = batch_size/2
        self.maxlen = maxlen
        self.skip_empty = skip_empty
        self.use_factor = use_factor
        self.pseudo_src_noise = noise

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target

        if self.n_words_source > 0:
            for d in self.source_dicts:
                for key, idx in d.items():
                    if idx >= self.n_words_source:
                        del d[key]

        if self.n_words_target > 0:
                for key, idx in self.target_dict.items():
                    if idx >= self.n_words_target:
                        del self.target_dict[key]

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.target_buffer = []
        self.pseudo_source_buffer = []
        self.pseudo_target_buffer = []
        self.k = batch_size * maxibatch_size
        

        self.end_of_data = False

    def __iter__(self):
        return self

    def __len__(self):
        return sum([1 for _ in self])
    
    def reset(self):
        # clear buffers for new epoch
        self.source_buffer = []
        self.target_buffer = []
        self.pseudo_source_buffer = []
        self.pseudo_target_buffer = []
        if self.shuffle:
            self.source, self.target = shuffle.main([self.source_orig, self.target_orig], temporary=True)
            self.pseudo_source, self.pseudo_target = shuffle.main([self.pseudo_source_orig, self.pseudo_target_orig], temporary=True)            
        else:
            self.source.seek(0)
            self.target.seek(0)
            self.pseudo_source.seek(0)
            self.pseudo_target.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []
        pseudo_source = []
        pseudo_target = []

        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch! (real data)'
        assert len(self.pseudo_source_buffer) == len(self.pseudo_target_buffer), 'Buffer size mismatch! (pseudo data)'        

        if len(self.source_buffer) == 0:
            for ss in self.source:
                ss = ss.split()
                tt = self.target.readline().split()
                
                if self.skip_empty and (len(ss) == 0 or len(tt) == 0):
                    continue
                if len(ss) > self.maxlen or len(tt) > self.maxlen:
                    continue

                self.source_buffer.append(ss)
                self.target_buffer.append(tt)
                if len(self.source_buffer) == self.k:
                    break

            if len(self.source_buffer) == 0 or len(self.target_buffer) == 0:
                self.end_of_data = False
                self.reset()
                raise StopIteration

            # sort by target buffer
            if self.sort_by_length:
                tlen = numpy.array([len(t) for t in self.target_buffer])
                tidx = tlen.argsort()

                _sbuf = [self.source_buffer[i] for i in tidx]
                _tbuf = [self.target_buffer[i] for i in tidx]

                self.source_buffer = _sbuf
                self.target_buffer = _tbuf

            else:
                self.source_buffer.reverse()
                self.target_buffer.reverse()

        if len(self.pseudo_source_buffer) == 0:
            for ss in self.pseudo_source:
                ss = ss.split()
                tt = self.pseudo_target.readline().split()
                
                if self.skip_empty and (len(ss) == 0 or len(tt) == 0):
                    continue
                if len(ss) > self.maxlen or len(tt) > self.maxlen:
                    continue

                self.pseudo_source_buffer.append(ss)
                self.pseudo_target_buffer.append(tt)
                if len(self.pseudo_source_buffer) == self.k:
                    break

            if len(self.pseudo_source_buffer) == 0 or len(self.pseudo_target_buffer) == 0:
                self.end_of_data = False
                self.reset()
                raise StopIteration

            # sort by target buffer
            if self.sort_by_length:
                tlen = numpy.array([len(t) for t in self.pseudo_target_buffer])
                tidx = tlen.argsort()

                _sbuf = [self.pseudo_source_buffer[i] for i in tidx]
                _tbuf = [self.pseudo_target_buffer[i] for i in tidx]

                self.pseudo_source_buffer = _sbuf
                self.pseudo_target_buffer = _tbuf

            else:
                self.pseudo_source_buffer.reverse()
                self.pseudo_target_buffer.reverse()


        try:
            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                tmp = []
                for w in ss:
                    if self.use_factor:
                        w = [self.source_dicts[i][f] if f in self.source_dicts[i] else 1 for (i,f) in enumerate(w.split('|'))]
                    else:
                        w = [self.source_dicts[0][w] if w in self.source_dicts[0] else 1]
                    tmp.append(w)
                ss = tmp

                # read from source file and map to word index
                tt = self.target_buffer.pop()
                tt = [self.target_dict[w] if w in self.target_dict else 1
                      for w in tt]
                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_target else 1 for w in tt]

                # read from source file and map to word index (pseudo source)
                try:
                    pss = self.pseudo_source_buffer.pop()
                except IndexError:
                    break

                if self.pseudo_src_noise:
                    # integrate noise in pseudo data (target copies)
                    # drop words
                    import random
                    p_drop = 0.1
                    pss = [s for s in pss if random.random() > p_drop]
                    # word permutations: a word can't be further
                    # than k words from its initial position.
                    k = 3
                    check_perm = [True for s in pss]
                    for i in range(len(pss)):
                        if check_perm[i] and random.random() < 0.1:
                            choose = [ii for ii in range(i-k, i+k+1) if ii >= 0 and ii < len(pss) and check_perm[ii]]
                            c = random.choice(choose)
                            # word permutation
                            pss[c], pss[i] = pss[i], pss[c]
                            check_perm[i] = check_perm[c] = False        

                tmp = []
                for w in pss:
                    w = [self.source_dicts[0][w] if w in self.source_dicts[0] else 1]
                    tmp.append(w)
                pss = tmp

                # read from source file and map to word index (pseudo target)
                ptt = self.pseudo_target_buffer.pop()
                ptt = [self.target_dict[w] if w in self.target_dict else 1
                      for w in ptt]
                if self.n_words_target > 0:
                    ptt = [w if w < self.n_words_target else 1 for w in ptt]

                source.append(ss)
                target.append(tt)
                pseudo_source.append(pss)
                pseudo_target.append(ptt)

                if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size or \
                        len(pseudo_source) >= self.batch_size or \
                        len(pseudo_target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True
            
        return source, target, pseudo_source, pseudo_target
