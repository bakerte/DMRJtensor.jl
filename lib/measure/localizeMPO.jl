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

function localizeOp(psi::MPS,Op::Array{G,1},insites::Array{R,1};#=centerpsi::TensType=psi[psi.oc],=#trail::Tuple=()) where {G <: TensType, R <: Integer}

  #trail operations....
  isId = [isapprox(norm(trail[r])^2,size(trail[r],1)) for r = 1:length(trail)]
  if length(trail) > 0
    for r = 1:length(isId)
      index = 0
      @inbounds while isId[r] && index < size(trail[r],1)
        index += 1
        isId[r] &= searchindex(trail[r],index,index) == 1
      end
    end
  end

  if issorted(insites)
    Oparray = Op
    sites = insites
  else
    order = sortperm(insites)
    Oparray = Op[order]
    sites = insites[order]
  end


  #needs to incorporate trail operators between operators.

  minsite = minimum(sites)
  Lstart = minsite < psi.oc ? minsite : psi.oc
  Lenv = LinearAlgebra.Diagonal(ones(eltype(psi[Lstart]),size(psi[Lstart],1)))
  currOp = [0]
  for w = minsite:psi.oc-1
    thispsi = psi[w]
    if w in sites
      p = findall(r->r==w,sites)
      for k = 1:length(p)
        thispsi = contract([2,1,3],Oparray[p[k]],2,thispsi,2)
        currOp[1] += 1
      end
    end
    for g = currOp[1]+1:length(trail)
      thispsi = contract([2,1,3],trail[g],2,thispsi,2)
    end
    Lenv = Lupdate(Lenv,psi[w],thispsi)
  end

  maxsite = maximum(sites)
  Rstart = maxsite > psi.oc ? maxsite : psi.oc
  Renv = LinearAlgebra.Diagonal(ones(eltype(psi[Rstart]),size(psi[Rstart],3)))
  currOp = [length(sites)+1]
  for w = maxsite:-1:psi.oc+1
    thispsi = psi[w]
    if w in sites
      p = findall(r->r==w,sites)
      for k = 1:length(p)
        thispsi = contract([2,1,3],Oparray[p[k]],2,thispsi,2)
        currOp[1] -= 1
      end
    end
    for g = currOp[1]+1:length(trail)
      thispsi = contract([2,1,3],trail[g],2,thispsi,2)
    end
    Renv = Rupdate(Renv,psi[w],thispsi)
  end

  if psi.oc in sites
    p = findfirst(r->psi.oc==sites[r],1:length(sites))
    outOp = Oparray[p]
  else
    outOp = eye(eltype(psi[1]),size(psi[psi.oc],2))
  end
  return Lenv,outOp,Renv
end

function localizeOp(psi::MPS,mpo::MPO...;Hpsi::Function=singlesite_update)
  Lenv,Renv = makeEnv(psi,mpo...)
  nMPOs = length(mpo)
  nLsize = nMPOs+2

  Lpsi = contract(Lenv[psi.oc],ndims(Lenv[psi.oc]),psi[psi.oc],1)
  for w = 1:nMPOs
    Lpsi = contract(Lpsi,(2+nMPOs-w,ndims(Lpsi)-w),mpo[w][psi.oc],(1,2))
  end

  Ltup = (ntuple(w->w+1,ndims(Lpsi)-3)...,ndims(Lpsi))
  Rtup = ntuple(w->w,ndims(Renv[psi.oc])-1)

  return contract(Lpsi,Ltup,Renv[psi.oc],Rtup)
end
export localizeOp

