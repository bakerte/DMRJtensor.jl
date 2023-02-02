#########################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                              v0.8.3
#
#########################################################################
# Made by Thomas E. Baker (2018)
# See accompanying license with this program
# This code is native to the julia programming language (v1.1.1) or (v1.5)
#

using DMRJtensor

mpo = makeMPO(heisenbergMPO,2,100)
psi = randMPS(2,100)

#dense version
dmrg(psi,mpo,sweeps=300,m=45,cutoff=1E-9,method="twosite")

#symmetry version
@makeQNs "spin" U1
Qlabels = [spin(-2),spin(2)] #2 in units of 1/2 (so, two halves)

qpsi,qmpo = MPS(Qlabels,psi,mpo)
dmrg(qpsi,qmpo,sweeps=300,m=45,cutoff=1E-9,method="twosite")
