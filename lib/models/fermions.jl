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
    Cup,Cdn,F,Nup,Ndn,Ndens,O,Id = fermionOps()

Make fermion operators Cup,Cdn,F,Nup,Ndn,Ndens,O,Id

  #Outputs:
  + `Cup`: spin-up annihilation operator
  + `Cdn`: spin-down annihilation operator
  + `F`: Jordan-Wigner Fermion string
  + `Nup`: spin-up number operator
  + `Ndn`: spin-dn number operator
  + `Ndens`: total-spin number operator
  + `O`: zero matrix
  + `Id`: identity matrix
"""
function fermionOps()
  states = 4 #fock space size
  O = zeros(Float64,states,states) #zero matrix
  Id = Array(tens(eye(states))) #copy(O)+LinearAlgebra.I #identity

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
    hubbardMPO(i[,t=1.0,mu=-2.0,HubU=4.0,Ops=fermionOps()])

Creates a bulk MPO of the Hubbard model for uniform kinetic energy `t`, and spin magnitude `spinmag`

See also: [`XXZ`](@ref)
"""
function hubbardMPO(i::intType;t::Number=1.0,mu::Number=-2.0,HubU::Number=4.0#=,Ops::Tuple = fermionOps()=#)
  Cup,Cdn,F,Nup,Ndn,Ndens,O,Id = fermionOps()
  onsite = mu * Ndens + HubU * Nup * Ndn
    return [Id  O O O O O;
        -t*Cup' O O O O O;
        conj(t)*Cup  O O O O O;
        -t*Cdn' O O O O O;
        conj(t)*Cdn  O O O O O;
        onsite Cup*F Cup'*F Cdn*F Cdn'*F Id]
end
export hubbardMPO


"""
    Cup,Cdn,F,Nup,Ndn,Ndens,Sp,Sm,Sz,O,Id = tJOps()

Operators for a t-J model

    #Outputs:
  + `Cup`: spin-up annihilation operator
  + `Cdn`: spin-down annihilation operator
  + `F`: Jordan-Wigner Fermion string
  + `Nup`: spin-up number operator
  + `Ndn`: spin-dn number operator
  + `Ndens`: total-spin number operator
  + `Sp`: spin-raising operator
  + `Sm`: spin-lowering operator
  + `Sz`: spin-z operator
  + `O`: zero matrix
  + `Id`: identity matrix
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

"""
    tjMPO(i[,t=1.0,mu=0.0,J=1.0])

Creates a bulk MPO of the t-J model for uniform kinetic energy `t`, onsite energy `mu`, and spin coupling `J`

See also: [`XXZ`](@ref)
"""
function tjMPO(i::intType;t::Number=1.0,mu::Number=0.0,J::Number=1.0#=,Ops::Tuple = tJOps()=#)
    Cup,Cdn,F,Nup,Ndn,Ndens,Sp,Sm,Sz,O,Id = tJOps()
    onsite = mu * Ndens #- Ne*exp(-abs(i-Ns/2)/2)*Ndens
    return [Id  O O O O O O O O O;
            Cup' O O O O O O O O O;
            Cup  O O O O O O O O O;
            Cdn' O O O O O O O O O;
            Cdn  O O O O O O O O O;
            J*Sp  O O O O O O O O O;
            J*Sm  O O O O O O O O O;
            J*Sz  O O O O O O O O O;
            J*Ndens O O O O O O O O O;
            onsite -t*F*Cup conj(t)*F*Cup' -t*F*Cdn conj(t)*F*Cdn' 0.5*Sm 0.5*Sp Sz -Ndens/4 Id]
end
export tjMPO

"""
    tJMPO(i[,t=1.0,mu=0.0,J=1.0])

Creates a bulk MPO of the t-J model for uniform kinetic energy `t`, onsite energy `mu`, and spin coupling `J`

See also: [`XXZ`](@ref)
"""
tJMPO = tjMPO
export tJMPO