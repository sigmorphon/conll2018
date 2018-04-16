from sys import argv, stdout
from random import random, seed, shuffle
from functools import wraps

sysargv = [a for a in argv]
import dynet as dy
seed(0)

from data import read_dataset, UNK, EOS, NONE, WF, LEMMA, MSD

LSTM_NUM_OF_LAYERS = 1
EMBEDDINGS_SIZE = 100
STATE_SIZE = 100
ATTENTION_SIZE = 100
WFDROPOUT=0.1
LSTMDROPOUT=0.3
# Every epoch, we train on a subset of examples from the train set,
# namely, 30% of them randomly sampled.
SAMPLETRAIN=0.3


def iscandidatemsd(msd):
    """ We only consider nouns, verbs and adjectives. """
    return msd.split(';')[0] in ['N','V','ADJ']

def init_model(wf2id,lemma2id,char2id,msd2id):
    global model, enc_fwd_lstm, enc_bwd_lstm, dec_lstm, character_lookup,\
    word_lookup, lemma_lookup, msd_lookup, attention_w1, attention_w2, \
    attention_v, decoder_w, decoder_b, output_lookup

    model = dy.Model()

    enc_fwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, 8*EMBEDDINGS_SIZE, STATE_SIZE, model)
    enc_bwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, 8*EMBEDDINGS_SIZE, STATE_SIZE, model)

    dec_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, STATE_SIZE*2+EMBEDDINGS_SIZE, STATE_SIZE, model)

    character_lookup = model.add_lookup_parameters((len(char2id), EMBEDDINGS_SIZE))
    word_lookup = model.add_lookup_parameters((len(wf2id), EMBEDDINGS_SIZE))
    lemma_lookup = model.add_lookup_parameters((len(lemma2id), EMBEDDINGS_SIZE))
    msd_lookup = model.add_lookup_parameters((len(msd2id), EMBEDDINGS_SIZE))

    attention_w1 = model.add_parameters( (ATTENTION_SIZE, STATE_SIZE*2))
    attention_w2 = model.add_parameters( (ATTENTION_SIZE, STATE_SIZE*LSTM_NUM_OF_LAYERS*2))
    attention_v = model.add_parameters( (1, ATTENTION_SIZE))
    decoder_w = model.add_parameters( (len(char2id), STATE_SIZE))
    decoder_b = model.add_parameters( (len(char2id)))
    output_lookup = model.add_lookup_parameters((len(char2id), EMBEDDINGS_SIZE))


def embed(lemma,context):
    """ Get word embedding and character based embedding for the input
        lemma. Concatenate the embeddings with a context representation. """
    lemma = [EOS] + list(lemma) + [EOS]
    lemma = [c if c in char2id else UNK for c in lemma]
    lemma = [char2id[c] for c in lemma]

    global character_lookup

    return [dy.concatenate([character_lookup[c], context])
            for c in lemma]

def run_lstm(init_state, input_vecs):
    s = init_state
    out_vectors = []
    for vector in input_vecs:
        s = s.add_input(vector)
        out_vector = s.output()
        out_vectors.append(out_vector)
    return out_vectors


def encode(embedded):
    embedded_rev = list(reversed(embedded))

    fwd_vectors = run_lstm(enc_fwd_lstm.initial_state(), embedded)
    bwd_vectors = run_lstm(enc_bwd_lstm.initial_state(), embedded_rev)
    bwd_vectors = list(reversed(bwd_vectors))
    vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]

    return vectors


def attend(input_mat, state, w1dt):
    global attention_w2
    global attention_v
    w2 = dy.parameter(attention_w2)
    v = dy.parameter(attention_v)

    # input_mat: (encoder_state x seqlen) => input vecs concatenated as cols
    # w1dt: (attdim x seqlen)
    # w2dt: (attdim x attdim)
    w2dt = w2*dy.concatenate(list(state.s()))
    # att_weights: (seqlen,) row vector
    unnormalized = dy.transpose(v * dy.tanh(dy.colwise_add(w1dt, w2dt)))
    att_weights = dy.softmax(unnormalized)
    # context: (encoder_state)
    context = input_mat * att_weights
    return context


def decode(vectors, output):
    output = [EOS] + list(output) + [EOS]
    output = [char2id[c] for c in output]
    w = dy.parameter(decoder_w)
    b = dy.parameter(decoder_b)
    w1 = dy.parameter(attention_w1)
    input_mat = dy.concatenate_cols(vectors)
    w1dt = None

    last_output_embeddings = output_lookup[char2id[EOS]]
    s = dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(STATE_SIZE*2), last_output_embeddings]))
    loss = []

    for char in output:
        # w1dt can be computed and cached once for the entire decoding phase
        w1dt = w1dt or w1 * input_mat
        vector = dy.concatenate([attend(input_mat, s, w1dt), last_output_embeddings])
        s = s.add_input(vector)
        out_vector = w * s.output() + b
        probs = dy.softmax(out_vector)
        last_output_embeddings = output_lookup[char]
        loss.append(-dy.log(dy.pick(probs, char)))
    loss = dy.esum(loss)
    return loss


def generate(i, s, id2char):
    """ Generate a word form for the lemma at position i in sentence s. """
    context = get_context(i,s)
    embedded = embed(s[i][LEMMA],context)
    encoded = encode(embedded)

    in_seq = s[i][LEMMA]
    w = dy.parameter(decoder_w)
    b = dy.parameter(decoder_b)
    w1 = dy.parameter(attention_w1)
    input_mat = dy.concatenate_cols(encoded)
    w1dt = None

    last_output_embeddings = output_lookup[char2id[EOS]]
    s = dec_lstm.initial_state().add_input(dy.concatenate(
            [dy.vecInput(STATE_SIZE * 2), last_output_embeddings]))

    out = ''
    count_EOS = 0
    for i in range(len(in_seq)*2):
        if count_EOS == 2: break
        # w1dt can be computed and cached once for the entire decoding phase
        w1dt = w1dt or w1 * input_mat
        vector = dy.concatenate([attend(input_mat, s, w1dt), 
                                 last_output_embeddings])
        s = s.add_input(vector)
        out_vector = w * s.output() + b
        probs = dy.softmax(out_vector).vec_value()
        next_char = probs.index(max(probs))
        last_output_embeddings = output_lookup[next_char]
        if id2char[next_char] == EOS:
            count_EOS += 1
            continue

        out += id2char[next_char]
    return out

def dropitem(item,item2id,training):
    return item2id[UNK if not item in item2id 
                   or training and random() < WFDROPOUT else
                   item]

def embed_context(prevword,prevlemma,prevmsd,lemma,
                  nextword,nextlemma,nextmsd):
    """ Emebed context elements. """
    return dy.concatenate([word_lookup[prevword], word_lookup[nextword],
                           lemma_lookup[prevlemma], lemma_lookup[nextlemma],
                           msd_lookup[prevmsd],msd_lookup[nextmsd],
                           lemma_lookup[lemma]])

def get_context(i,s,training=0):
    """ Embed context words, lemmas and MSDs. 
    
        The context of a lemma consists of the previous and following
        word forms, lemmas and MSDs as well as the MSD for the lemma
        in question.
    """
    prevword = s[i-1][WF] if i > 0 else EOS
    prevlemma = s[i-1][LEMMA] if i > 0 else EOS
    prevmsd = s[i-1][MSD] if i > 0 else EOS
    nextword = s[i+1][WF] if i + 1 < len(s) else EOS
    lemma = s[i][LEMMA] 
    nextlemma = s[i+1][LEMMA] if i + 1 < len(s) else EOS
    nextmsd = s[i+1][MSD] if i + 1 < len(s) else EOS

    prevword = dropitem(prevword,wf2id,training)
    nextword = dropitem(nextword,wf2id,training)
    prevlemma = dropitem(prevlemma,lemma2id,training)
    nextlemma = dropitem(nextlemma,lemma2id,training)
    prevmsd = dropitem(prevmsd,msd2id,training)
    nextmsd = dropitem(nextmsd,msd2id,training)
    lemma = dropitem(lemma,lemma2id,training)

    return embed_context(prevword,prevlemma,prevmsd,lemma,
                         nextword,nextlemma,nextmsd)

def get_loss(i, s):
    dy.renew_cg()  
    enc_fwd_lstm.set_dropout(LSTMDROPOUT)
    enc_bwd_lstm.set_dropout(LSTMDROPOUT)
    dec_lstm.set_dropout(LSTMDROPOUT)

    context = get_context(i,s,training=1)
    embedded = embed(s[i][LEMMA], context)
    encoded = encode(embedded)
    loss =  decode(encoded, s[i][WF])

    enc_fwd_lstm.set_dropout(0)
    enc_bwd_lstm.set_dropout(0)
    dec_lstm.set_dropout(0)
    
    return loss

def memolrec(func):
    """Memoizer for Levenshtein."""
    cache = {}
    @wraps(func)
    def wrap(sp, tp, sr, tr, cost):
        if (sr,tr) not in cache:
            res = func(sp, tp, sr, tr, cost)
            cache[(sr,tr)] = (res[0][len(sp):], res[1][len(tp):], res[4] - cost)
        return sp + cache[(sr,tr)][0], tp + cache[(sr,tr)][1], '', '', cost + cache[(sr,tr)][2]
    return wrap
                                                                
def levenshtein(s, t, inscost = 1.0, delcost = 1.0, substcost = 1.0):
    """Recursive implementation of Levenshtein, with alignments returned.
       Courtesy of Mans Hulden. """
    @memolrec
    def lrec(spast, tpast, srem, trem, cost):
        if len(srem) == 0:
            return spast + len(trem) * '_', tpast + trem, '', '', cost + len(trem)
        if len(trem) == 0:
            return spast + srem, tpast + len(srem) * '_', '', '', cost + len(srem)
        
        addcost = 0
        if srem[0] != trem[0]:
            addcost = substcost
            
        return min((lrec(spast + srem[0], tpast + trem[0], srem[1:], trem[1:], cost + addcost),
                    lrec(spast + '_', tpast + trem[0], srem, trem[1:], cost + inscost),
                    lrec(spast + srem[0], tpast + '_', srem[1:], trem, cost + delcost)),
                   key = lambda x: x[4])
        
    answer = lrec('', '', s, t, 0)
    return answer[0],answer[1],answer[4]
    
def eval(devdata,id2char,generating=1,outf=None):
    input, gold = devdata
    corr = 0.0
    lev=0.0
    tot = 0.0
    for n,s in enumerate(input):
        stdout.write("%u\r" % n)
        for i,fields in enumerate(s):
            wf, lemma, msd = fields            
            if msd == NONE and lemma != NONE:
                if generating:
                    wf = generate(i,s,id2char)
                if wf == gold[n][i][WF]:
                    corr += 1
                lev += levenshtein(wf,gold[n][i][WF])[2]
                tot += 1
            if outf:
                outf.write('%s\n' % '\t'.join([wf,lemma,msd]))
        if outf:
            outf.write('\n')
    return (0,0) if tot == 0 else (corr / tot, lev/tot)

def train(traindata,devdata,wf2id,lemma2id,char2id,id2char,msd2id,epochs=20):
    trainer = dy.AdamTrainer(model)
    for epoch in range(epochs):
        print("EPOCH %u" % (epoch + 1))
        shuffle(traindata)
        total_loss = 0
        for n,s in enumerate(traindata):
            for i,fields in enumerate(s):
                wf, lemma, msd = fields
                stdout.write("Example %u of %u\r" % 
                             (n+1,len(traindata)))
                if (iscandidatemsd(msd) or (msd == NONE and lemma != NONE))\
                   and random() < SAMPLETRAIN:
                    loss = get_loss(i, s)
                    loss_value = loss.value()
                    loss.backward()
                    trainer.update()
                    total_loss += loss_value
        print("\nLoss per sentence: %.3f" % (total_loss/len(traindata)))
        print("Example outputs:")
        for s in traindata[:5]:
            for i,fields in enumerate(s):
                wf, lemma, msd = fields
                if (iscandidatemsd(msd) or (msd == NONE and lemma != NONE))\
                   and random() < SAMPLETRAIN:
                    print("INPUT:", s[i][LEMMA], "OUTPUT:",
                          generate(i,s,id2char),
                          "GOLD:",wf)
                    break

        devacc, devlev = eval(devdata,id2char)
        print("Development set accuracy: %.2f" % (100*devacc))
        print("Development set avg. Levenshtein distance: %.2f" % devlev)
        print()


if __name__=='__main__':
    traindata, wf2id, lemma2id, char2id, msd2id = read_dataset(sysargv[1])
    devinputdata, _, _, _, _ = read_dataset(sysargv[2])
    devgolddata, _, _, _, _ = read_dataset(sysargv[3])

    id2char = {id:char for char,id in char2id.items()}
    init_model(wf2id,lemma2id,char2id,msd2id)
    train(traindata,[devinputdata,devgolddata],
          wf2id,lemma2id,char2id,id2char,msd2id,20)    
    eval([devinputdata,devgolddata],id2char,generating=1,
         outf=open("%s-out" % sysargv[2],"w"))
