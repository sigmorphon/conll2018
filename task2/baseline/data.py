EOS = "<EOS>"
UNK = "<UNK>"
NONE = "_"

WF = 0
LEMMA = 1
MSD = 2


def count(fields,wf2id,lemma2id,char2id,msd2id):
    """ Add word forms, lemmas and MSDs to relevant id number maps. """
    wf, lemma, msd = fields
    wf2id.setdefault(wf,len(wf2id))
    lemma2id.setdefault(lemma,len(lemma2id))
    msd2id.setdefault(msd,len(msd2id))
    for c in wf+lemma:
        char2id.setdefault(c,len(char2id))

def read_dataset(fn):
    f = open(fn)

    wf2id = {EOS:0,UNK:1,NONE:2} 
    lemma2id = {EOS:0,UNK:1,NONE:2} 
    char2id = {EOS:0,UNK:1,NONE:2} 
    msd2id = {EOS:0,UNK:1,NONE:2} 

    data = [[]]
    for line in f:
        line = line.strip('\n')
        if line:
            wf, lemma, msd = line.split('\t')
            data[-1].append([wf,lemma,msd])
            count(data[-1][-1],wf2id,lemma2id,char2id,msd2id)
        else:
            data.append([])
    data = [s for s in data if s != []]

    return data, wf2id, lemma2id, char2id, msd2id
