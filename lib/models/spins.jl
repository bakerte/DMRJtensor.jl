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
    Sp,Sm,Sz,Sy,Sx,O,Id = spinOps([,s=0.5])

Generates operators for a heisenberg model (spin-`s`, default 1/2)

  #Outputs:
  + `Sx`: spin-x operator
  + `Sy`: spin-y operator
  + `Sz`: spin-z operator
  + `Sp`: raising operator
  + `Sm`: lowering operator
  + `H`: Hadamard gate
  + `O`: zero matrix
  + `Id`: identity matrix
"""
function spinOps(;s=0.5)
  states = convert(Int64,2*s+1) #number of quantum states

  ox,oy,oz,Id,op,om,Rx,Ry,Rz,H,O = qubitOps(d=states)

  sz = oz * s
  sx = ox * s
  sy = oy * s

  return op,om,sz,sy,sx,O,Id # equal to Sp,Sm,Sz,Sy,Sx,O,Id
end

"""
    Sp,Sm,Sz,Sy,Sx,O,Id = spinOps(s)

Generates operators for a heisenberg model (spin-`s`)

  #Outputs:
  + `Sx`: spin-x operator
  + `Sy`: spin-y operator
  + `Sz`: spin-z operator
  + `Sp`: raising operator
  + `Sm`: lowering operator
  + `H`: Hadamard gate
  + `O`: zero matrix
  + `Id`: identity matrix
"""
function spinOps(a::Float64)
  return spinOps(s=a)
end
export spinOps





"""
    heisenbergMPO(i[,spinmag=0.5,J=0.5,Ops=spinOps(spinmag)])

Creates a bulk MPO of the Heisenberg model for uniform coupling `J`, spin magnitude `spinmag`, and operator set `Ops`

  #Example:
  julia> En = Vector{Float64}(undef,8)
  julia> for Ns = 3:10
           mpo = makeMPO(XXZ,2,Ns)
           psi = randMPS(2,Ns)
           En[Ns-2] = dmrg(psi,mpo,sweeps=50,m=100,cutoff=1E-9)
         end


  #Expected outputs:
  3 -1.0
  4 -0.9571067811865475
  5 -1.9278862533179937
  6 -2.0019953568985334
  7 -2.836239680686649
  8 -3.3749325986878933
  9 -3.7363216980340077
  10 -4.258035204636598

See also: [`XXZ`](@ref)
"""
function heisenbergMPO(i::intType;spinmag::Number=0.5,J::Number=1.0)
  Sp,Sm,Sz,Sy,Sx,O,Id = spinOps(spinmag)
  return [Id O O O O;
          J*0.5*Sm O O O O;
          J*0.5*Sp O O O O;
          J*Sz O O O O;
          O Sp Sm Sz Id]
end
export heisenbergMPO

"""
    XXZ(i[,spinmag=0.5,J=0.5])

Creates a bulk MPO of the Heisenberg model for uniform coupling `J`, and spin magnitude `spinmag`

See also: [`XXZ`](@ref)
"""
XXZ = heisenbergMPO
export XXZ


