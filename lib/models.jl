#########################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                              v0.8
#
#########################################################################
# Made by Thomas E. Baker and Jaimie Greasley (2020)
# See accompanying license with this program
# This code is native to the julia programming language (v1.5.4+)
#

"""
    Module: models

Some MPO models for DMRG
"""

function qubitOps(;d::intType=2,angle::W=pi/4,s::P=(d-1)/2) where {W <: Number, P <: Number}
  if d != 2
    println("WARNING: Hadamard gate not defined for more than 2 states")
  end

  s = (d-1)/2
  O = zeros(Float64,d,d) #zero matrix
  Id = makeArray(eye(2)) #identity matrix
  oz = copy(O) # z operator
  op = copy(O) # raising operator
  for (q,m) in enumerate(s:-1:-s) #counts from m to -m (all states)
    oz[q,q] = 2*m
    if m+1 <= s
      op[q-1,q] = sqrt(s*(s+1)-m*(m+1)) #Clebsch-Gordon coefficients
    end
  end
  om = Array(op') # lowering operator
  ox = (op+om) #x matrix
  oy = (om-op)*im #y matrix

  H = [1 1;1 -1]/sqrt(2)
  Rx = exp(-im*angle/2*ox)
  Ry = exp(-im*angle/2*oy)
  Rz = exp(-im*angle/2*oz)

  

  return ox,oy,oz,op,om,Rx,Ry,Rz,H,O,Id
end

function qubitOps(d::intType;angle::W=pi/4) where W <: Number
  return qubitOps(d=d,angle=angle)
end
export qubitOps

"""
  spinOps([,s=0.5])

Generates operators Sp,Sm,Sz,Sy,Sx,O,Id for a heisenberg model (spin-`s`, default 1/2)
"""
function spinOps(;s=0.5)
  states = convert(Int64,2*s+1) #number of quantum states

  ox,oy,oz,op,om,Rx,Ry,Rz,H,O,Id = qubitOps(d=states,s=s)

  sz = oz/2
  sx = ox/2
  sy = oy/2

  return op,om,sz,sy,sx,O,Id # equal to Sp,Sm,Sz,Sy,Sx,O,Id
end

function spinOps(a::Float64)
  return spinOps(s=a)
end
export spinOps

"""
    fermionOps()

Make fermion operators Cup,Cdn,F,Nup,Ndn,Ndens,O,Id
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

  return Cup,Cdn,F,Nup,Ndn,Ndens,O,Id
end
export fermionOps

"""
    tJOps()

Operators for a t-J model Cup,Cdn,F,Nup,Ndn,Ndens,Sp,Sm,Sz,O,Id
"""
function tJOps()
  #many of the Hubbard operators can be truncated
  Cup,Cdn,F,Nup,Ndn,Ndens,O,Id = fermionOps()
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
  Sp[3,2] = 1#/sqrt(2)
  Sm = Array(Sp') #spin lowering operator

  return Cup,Cdn,F,Nup,Ndn,Ndens,Sp,Sm,Sz,O,Id
end
export tJOps

function heisenbergMPO(i::intType;spinmag::Number=0.5,J::Number=0.5,Ops::Tuple=spinOps(spinmag))
  Sp,Sm,Sz,Sy,Sx,O,Id = Ops
  return [Id O O O O;
          J*Sm O O O O;
          J*Sp O O O O;
          Sz O O O O;
          O Sp Sm Sz Id]
end
export heisenbergMPO

XXZ = heisenbergMPO
export XXZ

function hubbardMPO(i::intType;t::Number=1.0,mu::Number=-2.0,HubU::Number=4.0,Ops::Tuple = fermionOps())
  Cup,Cdn,F,Nup,Ndn,Ndens,O,Id = Ops
  onsite = mu * Ndens + HubU * Nup * Ndn
    return [Id  O O O O O;
        -t*Cup' O O O O O;
        conj(t)*Cup  O O O O O;
        -t*Cdn' O O O O O;
        conj(t)*Cdn  O O O O O;
        onsite Cup*F Cup'*F Cdn*F Cdn'*F Id]
end
export hubbardMPO

function tjMPO(i::intType;t::Number=1.0,mu::Number=0.0,J::Number=1.0,Ops::Tuple = tJOps())
    Cup,Cdn,F,Nup,Ndn,Ndens,Sp,Sm,Sz,O,Id = Ops
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
export tjMPO
