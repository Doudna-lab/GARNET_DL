import os
import operator
from functools import reduce

#===============================================================================
class AnsiColor:
    # normal; k:black;  lr:light red
    # to be added
    n = '\x1b[0;0m'
    k = '\x1b[0;30m'
    r = '\x1b[0;31m'
    g = '\x1b[0;32m'
    y = '\x1b[0;33m'
    b = '\x1b[0;34m'
    p = '\x1b[0;35m'
    c = '\x1b[0;36m'
    lk = '\x1b[1;30m'
    lr = '\x1b[1;31m'
    lg = '\x1b[1;32m'
    ly = '\x1b[1;33m'
    lb = '\x1b[1;34m'
    lp = '\x1b[1;35m'
    lc = '\x1b[1;36m'
    # background color
    _k = '\x1b[0;40m'
    _r = '\x1b[0;41m'
    _g = '\x1b[0;42m'
    _y = '\x1b[0;43m'
    _b = '\x1b[0;44m'
    _p = '\x1b[0;45m'
    _c = '\x1b[0;46m'


cl = AnsiColor()


#===============================================================================
def pager(texts):
    try:
        fp = os.popen('less -R','w')
        fp.writelines(texts)
        fp.close()
    except IOError:
        pass

#===============================================================================
# divide a list of text into blocks
# according to user-defined getkey()
# return a list containing subgroups
# apply for AAABBBBCC... key sequence
def divide(lines, getkey):
    keys = [getkey(a) for a in lines]
    mydiff = [b!=a for a,b in zip(keys[:-1],keys[1:])]
    idx = [i+1 for i,a in enumerate(mydiff) if a==1]
    start = [0]+idx
    end = idx+[len(lines)]
    return [lines[i:j] for i,j in zip(start,end)]

#===============================================================================
# divide a list of text into blocks
# according to user-definded isborder()
# return a list containing subgroups
# include: can be 'header', 'tailer' or None
# include='header': keep border as 1st line of each block; if 1st line of text
#                   is not border, then no border in this block; if last line
#                   of text is border, then this border is discarded
# include='tailer': keep border as the last line of each block; if the last line
#                   of text is not border, then no border in this block; if 1st
#                   line of text is border, then this border is discarded
# include = None:   discard border in each block
def partition(lines, isborder, include=None):
    if not lines:
        return []
    idx = [i for i,a in enumerate(lines) if isborder(a)]
    if not idx:
        return [lines]

    if include == 'header':
        if idx[0] != 0:
            idx = [0]+idx
        if idx[-1] != len(lines)-1:
            idx = idx+[len(lines)]
        start = idx[:-1]
        end = idx[1:]
        return [lines[i:j] for i,j in zip(start,end)]
    elif include == 'tailer':
        if idx[0] != 0:
            idx = [-1]+idx
        if idx[-1] != len(lines)-1:
            idx = idx+[len(lines)-1]
        start = idx[:-1]
        end = idx[1:]
        return [lines[i+1:j+1] for i,j in zip(start,end)]
    else:  # by default
        if idx[0] != 0:
            idx = [-1]+idx
        if idx[-1] != len(lines)-1:
            idx = idx+[len(lines)]
        start = idx[:-1]
        end = idx[1:]
        return [lines[i+1:j] for i,j in zip(start,end)]

#===============================================================================
def findcommon(a, b):
    if not isinstance(a, list):
        a = list(a)
    if not isinstance(b, list):
        b = list(b)
    c = list(set(a) & set(b))
    c.sort()
    idx1 = [a.index(item) for item in c]
    idx2 = [b.index(item) for item in c]
    return idx1,idx2

#===============================================================================
def range2list(mystr):
    fds = [list(map(int, fd.split('-'))) for fd in mystr.split(',')]
    mylist = reduce(operator.add, [list(range(fd[0], fd[-1]+1)) for fd in fds])
    return mylist

#===============================================================================
def alignstr(strs):
    str0 = strs[0]
    print(('%s'%str0))
    for str1 in strs[1:]:
        cs = [(c1 if c1!=c0 else '-') for c0,c1 in zip(str0, str1)]
        cs = ''.join(cs)
        print(('%s'%cs))
