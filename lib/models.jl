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

function spinOps(;s=0.5)
  states = convert(Int64,2*s+1) #number of quantum states

  O = zeros(Float64,states,states) #zero matrix
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
export spinOps

"""
    fermionOps()
Make fermion operators
"""
function fermionOps()
  states = 4 #fock space size
  O = zeros(Float64,states,states) #zero matrix
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
export fermionOps

"""
    tJOps()
Operators for a t-J model
"""
function tJOps()
  #many of the Hubbard operators can be truncated
  Cup,Cdn,Nup,Ndn,Ndens,F,O,Id = fermionOps()
  states = 3 #fock space size
  s = states
  Cup = Cup[1:s,1:s]
  Cdn = Cdn[1:s,1:s]
  Nup = Nup[1:s,1:s]
  Ndn = Ndn[1:s,1:s]
  Ndens = Ndens[1:s,1:s]
  F = F[1:s,1:s]
  O = O[1:s,1:s]
  Id = Id[1:s,1:s]

  Sz = copy(O) #z-spin operator
  Sz[2,2] = 0.5
  Sz[3,3] = -0.5

  Sp = copy(O) #spin raising operator
  Sp[3,2] = 1/sqrt(2)
  Sm = Array(Sp') #spin lowering operator

  return Cup,Cdn,Nup,Ndn,Ndens,F,Sz,Sp,Sm,O,Id
end
export tJOps

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

function tjMPO(i::intType;t::Number=1.0,mu::Number=0.0,J::Number=1.0,Ops::Tuple = tJOps())
    Cup,Cdn,Nup,Ndn,Ndens,F,Sz,Sp,Sm,O,Id = Ops
    onsite = mu * Ndens #- Ne*exp(-abs(i-Ns/2)/2)*Ndens
    return [Id  O O O O O O O O O;
            Cup' O O O O O O O O O;
            Cup  O O O O O O O O O;
            Cdn' O O O O O O O O O;
            Cdn  O O O O O O O O O;
            Sp  O O O O O O O O O;
            Sm  O O O O O O O O O;
            Sz  O O O O O O O O O;
            Ndens O O O O O O O O O;
            onsite -t*F*Cup conj(t)*F*Cup' -t*F*Cdn conj(t)*F*Cdn' J*Sm J*Sp J*Sz -J*Ndens/4 Id]
end