from common.text import partition
#import pdb

class Entry:
    pass


class Amberlib:
    pass

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, fn=None):
        self.data = {}
        if fn:
            self.read(fn)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def read(self, fn):
        f = open(fn, 'rt')
        lines = f.readlines()
        f.close()

        self.data.clear()
        groups = partition(lines, lambda x: x[:6]=='!entry' and \
                           x.split('.')[3][:6]=='atoms ', include='header')
        idxstrs = [x.split('"')[1] for x in groups[0][1:]]
        groups = groups[1:]
        for group in groups:
            resn = group[0].split('.')[1]
            table = partition(group, lambda x: x[:1]=='!')[0]
#           pdb.set_trace()  #<====== for debug
            entry = Entry()
            entry.atnames = [item.split()[0].strip('"') for item in table]
            entry.charges = [float(item.split()[7]) for item in table]
            self.data[resn] = entry

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def writelib(self, fn):
        f = open(fn, 'wt')
        resns = list(self.data.keys())
        resns.sort()
        for resn in resns:
            res = self.data[resn]
#           f.write('RESI  %s  %s  aa\n'%(resn,resn.lower()))
            f.write('RESI  %s  %s  ion\n'%(resn,resn.lower()))
#           f.write('RESI  %s  %s  sol\n'%(resn,resn.lower()))
            for atname,charge in zip(res.atnames,res.charges):
                f.write(atname+'\n')
            f.write('\n')
        f.close()

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def writepar(self, fn):
        f = open(fn, 'wt')
        resns = list(self.data.keys())
        resns.sort()
        for resn in resns:
            res = self.data[resn]
            f.write('RESI  %s\n'%(resn.lower()))
            for atname,charge in zip(res.atnames,res.charges):
                f.write('%-4s  %9.6f\n'%(atname,charge))
            f.write('\n')
        f.close()
