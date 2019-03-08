import subprocess
import numpy as np
import os
import io
from rouge import Rouge
from constants import PAD, UNK, BOS, EOS

f = open('outfile.txt').read().splitlines()
cands = ''
outref = open('ref', 'w')
hyps = []
refs = []
for line in f:
    outref.write(line.split(';')[2].strip()+'\n')
    cands += line.split(';')[3].strip().replace(' '+PAD, '')+'\n'
    refs.append(line.split(';')[2].strip())
    hyps.append(line.split(';')[3].strip().replace(' '+EOS, '').replace(' '+PAD, '').replace(' '+UNK, ''))
outref.close()
rouge = Rouge()
scores = rouge.get_scores(hyps, refs, avg=True)
bleus = []
args = "./sentence-bleu ref".split()
popen = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(os.devnull, 'w'))
output, _ = popen.communicate(cands.encode('utf-8'))
for idx, o in enumerate(output.decode('utf-8').split('\n')):
    try:
        bleus.append(float(o))
    except:
        pass
print ('total evaluating {} sentences'.format(len(bleus)))
print ('bleu score mean is {}'.format(np.mean(bleus)))
print (scores)
