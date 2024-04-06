from amberlib import *

a = Amberlib('all_amino94.lib')
a.writelib('amber94_aa.lib')

a = Amberlib('all_ions94.lib')
a.writelib('amber94_ion.lib')

a = Amberlib('solvents.lib')
a.writelib('amber94_sol.lib')
