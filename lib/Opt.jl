#########################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                              v0.1
#
#########################################################################
# Made by Thomas E. Baker (2018)
# See accompanying license with this program
# This code is native to the julia programming language (v1.1.0) or (v1.5)
#

"""
    Module: Opt

Operator definitions for spin systems, fermions, and t-J model
"""
module Opt
import LinearAlgebra

  """
      spinOps()

  Make operators for a spin model (magnitude of size s)
    + equivalent: spinOps(s=0.5), spinOps(0.5)
  """
  function spinOps(;s=0.5)
    QS = convert(Int64,2*s+1) #number of quantum states

    O = zeros(Float64,QS,QS) #zero matrix
    Id = copy(O) + LinearAlgebra.I #identity matrix
    oz = copy(O) # z operator
    op = copy(O) # raising operator
    for (q,m) in enumerate(s:-1:-s) #counts from m to -m (all states)
      oz[q,q] = m
      if m+1 <= s
        op[q-1,q] = sqrt(s*(s+1)-m*(m+1))
      end
    end
    om = Array(op') # lowering operator
    ox = (op+om)/2 #x matrix
    oy = (op-om)/(2*im) #y matrix

    return ox,oy,oz,op,om,O,Id
  end
  function spinOps(a::Float64)
    return spinOps(s=a)
  end

  """
      fermionOps()
  Make fermion operators
  """
  function fermionOps()
    QS = 4 #fock space size
    O = zeros(Float64,QS,QS) #zero matrix
    Id = copy(O)+LinearAlgebra.I #identity

    Cup = copy(O) #annihilate (up)
    Cup[1,2] = 1.
    Cup[3,4] = 1.

    Cdn = copy(O) #annihilate (down)
    Cdn[1,3] = 1.
    Cdn[2,4] = -1.

    Nup = Cup' * Cup #density (up)
    Ndn = Cdn' * Cdn #density (down)
    Ndens = Nup + Ndn #density (up + down)

    F = copy(Id) #Jordan-Wigner string operator
    F[2,2] *= -1.
    F[3,3] *= -1.

    return Cup,Cdn,Nup,Ndn,Ndens,F,O,Id
  end

  """
      tJOps()
  Operators for a t-J model
  """
  function tJOps()
    #many of the Hubbard operators can be truncated
    Cup,Cdn,Nup,Ndn,Ndens,F,O,Id = fermionOps()
    QS = 3 #fock space size
    Cup = Cup[1:QS,1:QS]
    Cdn = Cdn[1:QS,1:QS]
    Nup = Nup[1:QS,1:QS]
    Ndn = Ndn[1:QS,1:QS]
    Ndens = Ndens[1:QS,1:QS]
    F = F[1:QS,1:QS]
    O = O[1:QS,1:QS]
    Id = Id[1:QS,1:QS]
  
    Sz = copy(O) #z-spin operator
    Sz[2,2] = 0.5
    Sz[3,3] = -0.5
  
    Sp = copy(O) #spin raising operator
    Sp[3,2] = 1.
    Sm = Sp' #spin lowering operator
  
   return Cup,Cdn,Nup,Ndn,Ndens,F,Sz,Sp,Sm,O,Id
  end

  export  spinOps,fermionOps,tJOps
end
