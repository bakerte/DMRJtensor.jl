println("#            +--------------+")
println("#>-----------|  MPOauto.jl  |-----------<")
println("#            +--------------+")
fulltest = true

import LinearAlgebra

Ns = 10

Sp,Sm,Sz,Sy,Sx,O,Id = spinOps()

function checkqubitops(val)
  base = [Id for i = 1:Ns]
  testval = true
  for i = 1:Ns
    for j = i+1:Ns
      A = mpoterm(val,[Sp,Sm],[i,j],base)
      B = Array{typeof(Sp),1}(undef,Ns)
      for k = 1:Ns
        if k == i
          B[i] = Sp*val
        elseif k == j
          B[j] = Sm
        else
          B[k] = Id
        end
      end
      C = MPO(B)
      for k = 1:Ns
        testval &= isapprox(makeArray(A[k]),makeArray(C[k]))
      end
    end
  end
  return testval
end

testval = checkqubitops(0.5)
fulltest &= testfct(testval,"mpoterm(val,[operators],[indices],base)")

function checkqubitops_noval(val)
  base = [Id for i = 1:Ns]
  testval = true
  for i = 1:Ns
    for j = i+1:Ns
      A = mpoterm([Sp,Sm],[i,j],base)
      B = Array{typeof(Sp),1}(undef,Ns)
      for k = 1:Ns
        if k == i
          B[i] = Sp
        elseif k == j
          B[j] = Sm
        else
          B[k] = Id
        end
      end
      C = MPO(B)
      for k = 1:Ns
        testval &= isapprox(makeArray(A[k]),makeArray(C[k]))
      end
    end
  end
  return testval
end

testval = checkqubitops_noval(0.5)
fulltest &= testfct(testval,"mpoterm([operators],[indices],base)")

function checkqubitops_single(val)
  base = [Id for i = 1:Ns]
  testval = true
  for i = 1:Ns
    A = mpoterm(val,Sp,i,base)
    B = Array{typeof(Sp),1}(undef,Ns)
    for k = 1:Ns
      if k == i
        B[i] = Sp*val
      else
        B[k] = Id
      end
    end
    C = MPO(B)
    for k = 1:Ns
      testval &= isapprox(makeArray(A[k]),makeArray(C[k]))
    end
  end
  return testval
end

testval = checkqubitops_single(0.5)
fulltest &= testfct(testval,"mpoterm(val,operator,index,base)")

function checkqubitops_single_noval(val)
  base = [Id for i = 1:Ns]
  testval = true
  for i = 1:Ns
    A = mpoterm(Sp,i,base)
    B = Array{typeof(Sp),1}(undef,Ns)
    for k = 1:Ns
      if k == i
        B[i] = Sp
      else
        B[k] = Id
      end
    end
    C = MPO(B)
    for k = 1:Ns
      testval &= isapprox(makeArray(A[k]),makeArray(C[k]))
    end
  end
  return testval
end

testval = checkqubitops_single(0.5)
fulltest &= testfct(testval,"mpoterm(operator,index,base)")

@makeQNs "testB" U1

Qlabels = [testB(-1),testB(1)]
qSx,qSy,qSz,qSp,qSm,qO,qId = Qtens(Qlabels,Sx,Sy,Sz,Sp,Sm,O,Id)

function Qcheckqubitops(val)
  base = [qId for i = 1:Ns]
  testval = true
  for i = 1:Ns
    for j = i+1:Ns
      A = mpoterm(val,[qSp,qSm],[i,j],base)
      B = Array{typeof(qSp),1}(undef,Ns)
      for k = 1:Ns
        if k == i
          B[i] = qSp*val
        elseif k == j
          B[j] = qSm
        else
          B[k] = qId
        end
      end
      C = MPO(B)
      for k = 1:Ns
        testval &= isapprox(makeArray(A[k]),makeArray(C[k]))
      end
    end
  end
  return testval
end

testval = Qcheckqubitops(0.5)
fulltest &= testfct(testval,"mpoterm(val,[operators],[indices],base)")


vecQlabels = [Qlabels,inv.(Qlabels)]
function checkqubitops_string(val,vecQlabels)
  base = [Id for i = 1:Ns]
  testval = true
  for i = 1:Ns
    for j = i+1:Ns
      A = mpoterm(vecQlabels,val,[Sp,Sm],[i,j],base)
      B = Array{typeof(Sp),1}(undef,Ns)
      for k = 1:Ns
        if k == i
          B[i] = Sp*val
        elseif k == j
          B[j] = Sm
        else
          B[k] = Id
        end
      end
      C = MPO(B)
      for k = 1:Ns
        testval &= isapprox(makeArray(A[k]),makeArray(C[k]))
      end
    end
  end
  return testval
end

testval = checkqubitops_string(0.5,vecQlabels)
fulltest &= testfct(testval,"mpoterm(Qlabels,val,[operators],[indices],base)")


vecQlabels = [Qlabels,inv.(Qlabels)]
function checkqubitops_string_noval(vecQlabels)
  base = [Id for i = 1:Ns]
  testval = true
  for i = 1:Ns
    for j = i+1:Ns
      A = mpoterm(vecQlabels,[Sp,Sm],[i,j],base)
      B = Array{typeof(Sp),1}(undef,Ns)
      for k = 1:Ns
        if k == i
          B[i] = Sp
        elseif k == j
          B[j] = Sm
        else
          B[k] = Id
        end
      end
      C = MPO(B)
      for k = 1:Ns
        testval &= isapprox(makeArray(A[k]),makeArray(C[k]))
      end
    end
  end
  return testval
end

testval = checkqubitops_string_noval(vecQlabels)
fulltest &= testfct(testval,"mpoterm(Qlabels,[operators],[indices],base)")

println()

C = [heisenbergMPO(i) for i = 1:Ns]
mpo = makeMPO(C,2)

function bigOp(X,i,Ns,Id)
  lId = 1
  for k = 1:i-1
    lId = kron(lId,Id)
  end
  rId = 1
  for k = Ns:-1:i+1
    rId = kron(Id,rId)
  end
  return kron(lId,X,rId)
end

function test(Sp,Sm,Sz,Id,Ns)
  base = [Id for i = 1:Ns]

  Ham = bigOp(Sz,1,Ns,Id)*bigOp(Sz,2,Ns,Id)
  mpo = mpoterm(1.,[Sz,Sz],[1,2],base)

  for i = 2:Ns-1
    Ham += bigOp(Sz,i,Ns,Id)*bigOp(Sz,i+1,Ns,Id)
    mpo += mpoterm(1.0,[Sz,Sz],[i,i+1],base)
  end

  for i = 1:Ns-1
    Ham += bigOp(Sp,i,Ns,Id)*bigOp(Sm,i+1,Ns,Id)/2
    Ham += bigOp(Sm,i,Ns,Id)*bigOp(Sp,i+1,Ns,Id)/2

    mpo += mpoterm(0.5,[Sp,Sm],[i,i+1],base)
    mpo += mpoterm(0.5,[Sm,Sp],[i,i+1],base)
  end
  
  return Ham,mpo
end

Ham,mpo = test(Sp,Sm,Sz,Id,Ns)

testval = isapprox(makeArray(Ham),makeArray(fullH(mpo)))
D,U = LinearAlgebra.eigen(Ham)
testval &= isapprox(D[1],-4.258035204637)

#i = 1
fulltest &= testfct(testval,"Hamiltonian: Sz_i Sz_(i+1)")

println()



psi = makeMPS(U[:,1],2)

offsetval = -0.1
#expect(psi,mpo+offsetval)

testval = isapprox(expect(psi,mpo),D[1])
testval &= isapprox(expect(psi,mpo+offsetval)-expect(psi,mpo),offsetval)
fulltest &= testfct(testval,"+(mpo,number)")

println()

testval = isapprox(expect(psi,mpo),D[1])
testval &= isapprox(expect(psi,mpo-offsetval)-expect(psi,mpo),-offsetval)
fulltest &= testfct(testval,"-(mpo,number)")

println()


function another_test(Sp,Sm,Sz,Id,Ns)
  base = [Id for i = 1:Ns]

#  Ham = zeros(2^Ns,2^Ns)#bigOp(Sz,1,Ns,Id)*bigOp(Sz,2,Ns,Id)
#=
  mpo_tup_Sz = [mpoterm([Sz,Sz],[i,i+1],base) for i = 1:Ns-1]
  mpo_tup_Sp = [mpoterm(0.5,[Sp,Sm],[i,i+1],base) for i = 1:Ns-1]
  mpo_tup_Sm = [mpoterm(0.5,[Sm,Sp],[i,i+1],base) for i = 1:Ns-1]
  =#
  mpo_tup_Sz = ntuple(i->mpoterm([Sz,Sz],[i,i+1],base),Ns-1)
  mpo_tup_Sp = ntuple(i->mpoterm(0.5,[Sp,Sm],[i,i+1],base),Ns-1)
  mpo_tup_Sm = ntuple(i->mpoterm(0.5,[Sm,Sp],[i,i+1],base),Ns-1)

  mpo = +(mpo_tup_Sz...,mpo_tup_Sp...,mpo_tup_Sm...)
  return mpo
end

mpo = another_test(Sp,Sm,Sz,Id,Ns)

testval = isapprox(makeArray(Ham),makeArray(fullH(mpo)))
D,U = LinearAlgebra.eigen(Ham)
testval &= isapprox(D[1],-4.258035204637)
fulltest &= testfct(testval,"+(mpo...) and add!(mpo,mpo)")

println()

println("deparallelize! implicit in the preceeding...can comment out to check separately")

println()

cmpo = compressMPO!(mpo)

testval = isapprox(D[1],expect(psi,cmpo))
fulltest &= testfct(testval,"compressMPO!(mpo)")

println()

#reorder!()


Ns = 100

psi = randMPS(2,Ns)
mpo = makeMPO(heisenbergMPO,2,Ns)
dmrg(psi,mpo,sweeps=300,goal=1E-8,cutoff=1E-9,m=100,method="twosite",silent=true)

println("STARTING ALGORITHM")
true_energy = expect(psi,mpo)
@time expect(psi,mpo)


C = [nametens(conj(psi[i]), ["a$(i-1)", "b$i", "a$i"]) for i in range(1,Ns)]
D = [nametens(mpo[i], ["h$(i-1)", "c$i", "b$(i)", "h$i"]) for i in range(1,Ns)]
E = [nametens(psi[i], ["f$(i-1)", "c$i", "f$i"]) for i in range(1,Ns)]

F = vcat(E,C,D)

G = network(F)
answer = contract(G)

#@time contract(G)

testval = abs(true_energy - answer.N.T[1]) < 1E-6
fulltest &= testfct(testval,"contract(network)")
