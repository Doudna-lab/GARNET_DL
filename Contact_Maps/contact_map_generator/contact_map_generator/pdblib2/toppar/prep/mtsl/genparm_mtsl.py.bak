#!/usr/bin/python

from common.text import partition

lines = open('cml.lib').readlines()
lgrps = partition(lines, isborder=lambda x: x[:1]=='!')

print('RESI  mtsl')
for line in lgrps[1]:
    fds = line.split()
    name = fds[0][1:-1]
    charge = float(fds[7])
    print('%-4s  %9.6f'%(name,charge))
