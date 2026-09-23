#=
function paulidecomposition(psi::Array{W,1};d::Integer=2,ops=qubitOps(d)) where W <: Number
  return paulidecomposition(psi,d,ops=ops)
end
=#
function paulidecomposition(psi::Union{Array{W,1},Array{W,2},tens{W}};d::Integer=2,ops=qubitOps(d)) where W <: Number
  ox,oy,oz,Id = qubitOps(d)
  checkId = eye(ops[1])

#  if ndims(psi) == 2
    Ns = round(Int64,log(d,size(psi,1)))
#  else
#    Ns = round(Int64,log(d,length(psi)))
#  end

  println(length(psi)," ",Ns," ",d)

  p = spzeros(ComplexF64,0,0)


  saveOps = Array{Array{typeof(p),1},1}(undef,Ns)
  for i = 1:Ns
    saveOps[i] = Array{typeof(p),1}(undef,4)
    saveOps[i][1] = sparse(kron(ox,i,Ns))
    saveOps[i][2] = sparse(kron(oy,i,Ns))
    saveOps[i][3] = sparse(kron(oz,i,Ns))
    saveOps[i][4] = sparse(kron(Id,i,Ns))
  end

#  saveOps = [(kron(ox,i,Ns),kron(oy,i,Ns),kron(oz,i,Ns),kron(Id,i,Ns)) for i = 1:Ns]

  psi = Array(psi) #sparse(psi) #to be implemented later for easy identification


#  Nops = [length(saveOps[w]) for w = 1:Ns]
  sizes = [length(saveOps[w]) for w = 1:Ns]
  nterms = prod(sizes)

#  println(sizes)

  values = Array{ComplexF64,1}(undef,nterms)
  weights = Array{intType,1}(undef,nterms)
  
  pos = makepos(Ns)
  for w = 1:nterms
    weights[w] = 0
    position_incrementer!(pos,sizes)
    paulistring = 1
    for x = 1:Ns #Ns:-1:1
#      println(x," ",length(saveOps))
#      println(pos[w]," ",length(pos))
      paulistring *= saveOps[x][pos[x]] #*paulistring
      if norm(ops[pos[x]]-checkId) > 1E-15 
#        println(w," ",pos)
        #!isapprox(norm(ops[pos[x]]-checkId),0)
        weights[w] += 1
      end
    end
    if ndims(psi) == 2
      values[w] = trace(psi*paulistring)
    else
      values[w] = psi'*paulistring*psi
    end
    values[w] /= 2^Ns
#    if abs(values[w]) > 1E-12
#      println(w," ",values[w]," ",weights[w]," ",pos)
#    end
  end

  if ndims(psi) == 2 && true #check
    checkmat = zeros(size(psi))
    pos = makepos(Ns)
    for w = 1:nterms
      position_incrementer!(pos,sizes)
      paulistring = 1
      for x = 1:Ns
        paulistring *= saveOps[x][pos[x]]
      end
      checkmat += values[w]*paulistring
    end
#    display(checkmat)
#    display(psi)
#    display(checkmat-psi)
    println("checking... ",norm(checkmat-psi))
  end

  return values,weights
end

function paulidecomposition(psi::MPS;ops=qubitOps(size(psi[1],2)))

  ops = ops[1:4]

  Ns = length(psi)
  Nops = length(ops)
  pos = makepos(Ns)
  sizes = ntuple(w->Nops,Ns)
  nterms = prod(sizes)

  checkId = eye(ops[1])

  paulistring = Array{eltype(ops),1}(undef,Ns)
  values = Array{ComplexF64,1}(undef,nterms)
  weights = Array{intType,1}(undef,nterms)
  for w = 1:nterms
    position_incrementer!(pos,sizes)
    weights[w] = 0
    for x = 1:Ns
      paulistring[x] = ops[pos[x]]
      if !isapprox(norm(ops[pos[x]]-checkId),0)
        weights[w] += 1
      end
    end
    mpo = MPO(paulistring)
    values[w] = expect(psi,mpo)
    println(w," ",values[w]," ",weights[w])
  end
  return values,weights
end
