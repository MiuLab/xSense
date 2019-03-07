from collections import Counter

f = open('data/wiki_en_train.txt')
#f = open('../GoogleNews-vectors-negative300.bin.txt', encoding='utf-8')
voc = set()
for line in f:
    word = line.split(' ')[0]
    voc.add(word)
f.close()
f = open('data/s2s_sent.txt')
cnt = 0
tl = 0
oovs = Counter()
#wlist = ['a', 'of', 'and', 'to', 'an', 'is']
wlist = []
empty = 0
for idx, line in enumerate(f):
    line = line.strip()
    if idx%2==1:
        line = line.split(' ')
        tl += len(line)
        for w in line:
            if w == '':
                print (idx)
                empty += 1
            if w not in voc and w not in wlist:
                cnt += 1
                oovs[w] += 1
print (cnt/tl)
print (empty)
out = open('append_word.txt', 'w')
for w in oovs.most_common(1000):
    out.write('{}\n'.format(w[0]))
