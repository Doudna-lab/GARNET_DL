lines=open('parm.dat').readlines()

for i in range(10000):
#   filter(lambda x: x[:1]!='#', lines)
    [line for line in lines if line[:1]!='#']
