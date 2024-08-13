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

"""
    boundaryMove!(psi,i,mpo,Lenv,Renv)

Move orthogonality center of `psi` to site `i` and updates left and right environments `Lenv` and `Renv` using MPO `mpo`

See also: [`move!`](@ref)
"""
 function boundaryMove!(psi::MPS,i::Integer,Lenv::Env,Renv::Env,mpo::MPO...;mover::Function=move!)
  origoc = psi.oc
  if origoc < i
    mover(psi,i)
    for w = origoc:i-1
      Lupdate!(w,Lenv,psi,mpo...)
    end
  elseif origoc > i
    mover(psi,i)
    for w = origoc:-1:i+1
      Rupdate!(w,Renv,psi,mpo...)
    end
  end
  nothing
end

"""
    boundaryMove!(dualpsi,psi,i,mpo,Lenv,Renv)

Move orthogonality center of `psi` and `dualpsi` to site `i` and updates left and right environments `Lenv` and `Renv` using MPO `mpo`

See also: [`move!`](@ref)
"""
 function boundaryMove!(dualpsi::MPS,psi::MPS,i::Integer,Lenv::Env,Renv::Env,mpo::MPO...;mover::Function=move!)
  origoc = psi.oc
  if origoc < i
    mover(psi,i)
    mover(dualpsi,i)
    for w = origoc:i-1
      Lenv[w+1] = Lupdate(Lenv[w],dualpsi[w],psi[w],[mpo[a][w] for a = 1:length(mpo)]...)
    end
  elseif origoc > i
    mover(psi,i)
    mover(dualpsi,i)
    for w = origoc:-1:i+1
      Renv[w-1] = Rupdate(Renv[w],dualpsi[w],psi[w],[mpo[a][w] for a = 1:length(mpo)]...)
    end
  end
  nothing
end
export boundaryMove!

"""
    boundaryMove(dualpsi,psi,i,mpo,Lenv,Renv)

Copies `psi` and moves orthogonality center of `psi` and `dualpsi` to site `i` and updates left and right environments `Lenv` and `Renv` using MPO `mpo`

See also: [`move!`](@ref)
"""
 function boundaryMove(dualpsi::MPS,psi::MPS,i::Integer,Lenv::Env,Renv::Env,mpo::MPO...)
  newpsi = copy(psi)
  newdualpsi = copy(dualpsi)
  newLenv = copy(Lenv)
  newRenv = copy(Renv)
  boundaryMove!(newdualpsi,newpsi,i,newLenv,newRenv,mpo...)
  return newdualpsi,newpsi,newLenv,newRenv
end

"""
    boundaryMove(psi,i,mpo,Lenv,Renv)

Copies `psi` and moves orthogonality center of `psi` to site `i` and updates left and right environments `Lenv` and `Renv` using MPO `mpo`

See also: [`move!`](@ref)
"""
 function boundaryMove(psi::MPS,i::Integer,Lenv::Env,Renv::Env,mpo::MPO...)
  newpsi = copy(psi)
  newLenv = copy(Lenv)
  newRenv = copy(Renv)
  boundaryMove!(newpsi,newpsi,i,newLenv,newRenv,mpo...)
  return newpsi,newLenv,newRenv
end
export boundaryMove