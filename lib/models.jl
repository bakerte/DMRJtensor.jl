#########################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                              v0.8
#
#########################################################################
# Made by Thomas E. Baker (2020)
# See accompanying license with this program
# This code is native to the julia programming language (v1.5.4+)
#

"""
    Module: models

Some MPO models for DMRG
"""

function heisenbergMPO(i::intType;spinmag::Number=0.5,Ops::Tuple=spinOps(spinmag))
  Sx,Sy,Sz,Sp,Sm,O,Id = Ops
  return [Id O O O O;
          Sp O O O O;
          Sm O O O O;
          Sz O O O O;
          O Sm/2 Sp/2 Sz Id]
end

function hubbardMPO(i::intType;t::Number=1.0,mu::Number=-2.0,HubU::Number=4.0,Ops::Tuple = fermionOps())
  Cup, Cdn, Nup, Ndn, Ndens, F, O, Id = Ops
  onsite = mu * Ndens + HubU * Nup * Ndn
    return [Id  O O O O O;
        -t*Cup' O O O O O;
        conj(t)*Cup  O O O O O;
        -t*Cdn' O O O O O;
        conj(t)*Cdn  O O O O O;
        onsite Cup*F Cup'*F Cdn*F Cdn'*F Id]
end