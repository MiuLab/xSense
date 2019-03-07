from urllib.request import urlopen
from urllib.error import HTTPError
import re
from bs4 import BeautifulSoup
from collections import defaultdict
import json
import sys
import tqdm

def download_word(word):
    try:
        URL = 'https://en.oxforddictionaries.com/definition/' + word
        html = urlopen(URL).read().decode('utf-8')
    except HTTPError:
        print ('{} download failed'.format(word))
        print ('{} download failed'.format(word), file=sys.stderr)
        return {}
    if 'No exact matches found for' in html:
        print ('{} is oov'.format(word))
        print ('{} is oov'.format(word), file=sys.stderr)
        return {}
    try:
        parsed_html = BeautifulSoup(html, "lxml")
        for div in parsed_html.find_all("section", {'class':'etymology'}):
            div.decompose()
        html = str(parsed_html)
        pos = re.findall('<span class="pos">(.*?)</span>', html)
        ret_iter = []
        iteration_pattern = '<span class="iteration">(.*?)</span>'
        subIteration_pattern = '<span class="subsenseIteration">(.*?)</span>'
        for i in re.findall(iteration_pattern+'|'+subIteration_pattern, html):
            i = i[0] if len(i[0]) != 0 else i[1]
            if len(i) == 0:
                ret_iter.append('1')
            else:
                ret_iter.append(i)
                '''
                i = float(i)
                if int(i) == float(i):
                    ret_iter.append(int(i))
                else:
                    ret_iter.append(float(i))
                '''
        lemma = re.findall('<head><title>(.*?)'+ re.escape('| Definition of'), html)[0][:-1]
        exs = parsed_html.find_all(attrs={"class": "examples"})
        defs = parsed_html.find_all(attrs={"class": "gramb"})
        ret_def = []
        ret_ex = []
        for d in defs:
            semb = d.find_all(attrs={"class": "semb"})[0]
            ps = semb.find_all('li', "")
            for p in ps:
                if str(p)[:4] == '<li>':
                    trgs = p.find_all(attrs={"class": "trg"})
                    temp = str(trgs[0])
                    zs= re.split('<span class="ind">(.*?)</span>', temp)
                    for idx, z in enumerate(zs):
                        sents = re.findall('<em>(.*?)</em>', z)
                        if len(sents) != 0: # these are example sentences
                            definition = zs[idx-1]
                            try_def = re.findall('<div class="crossReference">(.*?)</div>', definition)
                            temp_def = []
                            temp_ex = []
                            if len(try_def) != 0:
                                try_def = try_def[0]
                                temp_def.append(re.sub('<(.*?)>', '', try_def))
                            else:
                                temp_def.append(definition)
                            for ss in sents:
                                temp_ex.append(ss[1:-1])
                            ret_def.append(temp_def)
                            ret_ex.append(temp_ex)
        
        # make ret
        ret_d = defaultdict(list)
        '''
        key: lemma
        value: dict key: iteration num
                    value: dict key: def
                                value: example sentences
        '''
        '''
        print (ret_iter)
        print (ret_def)
        print (pos)
        exit()
        '''
        sources = {}
        cnt = 0
        for idx, (it, defs, sents) in enumerate(zip(ret_iter, ret_def, ret_ex)):
            sources[it] = {defs[0]:sents}
            if idx < len(ret_iter) - 1 and float(ret_iter[idx+1]) == int(float(ret_iter[idx+1])) and float(it) >= float(ret_iter[idx+1]):
                add_sources = {}
                add_sources['pos'] = pos[cnt]
                cnt += 1
                add_sources['content'] = sources
                sources = {}
                ret_d[lemma].append(add_sources)
        if len(sources) != 0: # there may be some phrases
            add_sources = {}
            add_sources['pos'] = pos[cnt]
            add_sources['content'] = sources
        ret_d[lemma].append(add_sources)
        #print (json.dumps(ret_d, indent=4, sort_keys=True))
        return ret_d
    
    except:
        print ('{} processing failed'.format(word))
        print ('{} processing failed'.format(word), file=sys.stderr)
        return {}
    

all_d = {}
words = open('vocab.txt').read().splitlines()
for idx, word in enumerate(words):
    d = download_word(word)
    all_d.update(d)
    if idx % 2000 == 0: #checkpoint
        with open('data.txt', 'w') as outfile:
            json.dump(all_d, outfile, indent=4)
            print (idx)

with open('data.txt', 'w') as outfile:
    json.dump(all_d, outfile, indent=4)
#print (json.dumps(all_d, indent=4, sort_keys=True))
