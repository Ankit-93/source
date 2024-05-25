from source.utils.config import *



def expandContractions(text):
    c_re = re.compile('(%s)' % '|'.join(cList.keys()))
    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
    
def normalizeString(s):
    s = unicodeToAscii(expandContractions(s.lower().strip()))
    s = re.sub(r"([.!?,\"])", r" ", s)
    return s
    
def loaddata(filename):
    file = open(filename,'r',encoding="utf8")
    eng=[]
    beng = []
    for line in file.readlines():
        lang_pair = line.split('\t')
        lang_pair[0] = normalizeString(lang_pair[0])
        lang_pair[1] = normalizeString(lang_pair[1])
        eng.append(word_tokenize(lang_pair[0]))
        beng.append(word_tokenize(lang_pair[1]))
    file.close()
    return eng,beng