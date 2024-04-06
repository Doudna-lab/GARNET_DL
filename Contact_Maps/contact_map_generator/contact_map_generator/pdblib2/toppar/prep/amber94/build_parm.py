from amberlib import *

a = Amberlib('all_amino94.lib')
a.writepar('parm_aa.dat')

a = Amberlib('all_aminont94.lib')
a.writepar('parm_nt.dat')

a = Amberlib('all_aminoct94.lib')
a.writepar('parm_ct.dat')

a = Amberlib('ions94.lib')
a.writepar('parm_ion.dat')

a = Amberlib('solvents.lib')
a.writepar('parm_sol.dat')
