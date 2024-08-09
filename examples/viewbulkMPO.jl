###############################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                               v1.0
#
###############################################################################
# Made by Thomas E. Baker and « les qubits volants » (2024)
# See accompanying license with this program
# This code is native to the julia programming language (v1.10.0+)
#

#This example will make the bulk MPO for the XXZ model
#For viewing only


H = bulkMPO(5)

H[1,1] = H[end,end] = "I"

H[2,1] = "Sz"
H[3,1] = "Sp"
H[4,1] = "Sm"

H[end,2] = "Sz"
H[end,3] = half*"Sm"
H[end,4] = half*"Sp"

Ns = 10 #number of sites
bigH = prod(w->H,1:Ns)
display(bigH[end,1]) #Hamiltonian terms will accumulate in the lower left corner
