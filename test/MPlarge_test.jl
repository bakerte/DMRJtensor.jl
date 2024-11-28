
println("#            +-------------+")
println("#>-----------|  MPlarge.jl |-----------<")
println("#            +-------------+")
fulltest = true

A = rand(20,30,40)

testval = DMRjulia.file_extension == ".dmrjulia"
fulltest &= testfct(testval,".dmrjulia")

println()

DMRjulia.tensor2disc("del",A,ext=DMRjulia.file_extension)
B = DMRjulia.tensorfromdisc("del",ext=DMRjulia.file_extension)
testval = isapprox(A,B)
fulltest &= testfct(testval,"tensor2disc and tensorfromdisc")



println()

Ns = 10
psi = MPS(2,Ns)
A = largeMPS(psi)
testval = typeof(A.A) <: Array{String,1}
testval &= A.oc == psi.oc
testval &= A.type == eltype(psi)
fulltest &= testfct(testval,"largeMPS(mps)")

psi = A

A = largeMPS(Ns)
testval = typeof(A.A) <: Array{String,1}
testval &= A.oc == psi.oc
testval &= A.type == eltype(psi)
fulltest &= testfct(testval,"largeMPS(integer)")

A = largeMPS(ComplexF64,Ns)
testval = typeof(A.A) <: Array{String,1}
testval &= A.oc == psi.oc
testval &= A.type == ComplexF64
fulltest &= testfct(testval,"largeMPS(type,integer)")

println()

msize = 4
C = [rand(i == 1 ? 1 : msize,2,2,i == Ns ? 1 : msize) for i = 1:Ns]

mpo = MPO(C)
A = largeMPO(mpo)
testval = typeof(A.H) <: Array{String,1}
testval &= A.type == eltype(mpo)
fulltest &= testfct(testval,"largeMPS(mps)")

mpo = A

A = largeMPO(Ns)
testval = typeof(A.H) <: Array{String,1}
testval &= A.type == eltype(mpo)
fulltest &= testfct(testval,"largeMPS(integer)")

A = largeMPO(ComplexF64,Ns)
testval = typeof(A.H) <: Array{String,1}
testval &= A.type == ComplexF64
fulltest &= testfct(testval,"largeMPS(integer)")


println()

Lenv = environment(C)
A = largeLenv(Lenv)
testval = typeof(A.V) <: Array{String,1}
testval &= A.type == eltype(C[1])
fulltest &= testfct(testval,"largeLenv(Env)")

A = largeLenv(Ns)
testval = typeof(A.V) <: Array{String,1}
testval &= A.type == eltype(C[1])
fulltest &= testfct(testval,"largeLenv(integer)")

A = largeLenv(ComplexF64,Ns)
testval = typeof(A.V) <: Array{String,1}
testval &= A.type == ComplexF64
fulltest &= testfct(testval,"largeLenv(type,integer)")

println()


Renv = environment(C)
A = largeRenv(Renv)
testval = typeof(A.V) <: Array{String,1}
testval &= A.type == eltype(C[1])
fulltest &= testfct(testval,"largeRenv(Env)")

A = largeRenv(Ns)
testval = typeof(A.V) <: Array{String,1}
testval &= A.type == eltype(C[1])
fulltest &= testfct(testval,"largeRenv(integer)")

A = largeRenv(ComplexF64,Ns)
testval = typeof(A.V) <: Array{String,1}
testval &= A.type == ComplexF64
fulltest &= testfct(testval,"largeRenv(type,integer)")


println()


A,B = largeEnv(Lenv,Renv)
testval = typeof(A.V) <: Array{String,1}
testval &= A.type == eltype(C[1])
fulltest &= testfct(testval,"largeEnv(Env)")

A,B = largeEnv(Ns)
testval = typeof(A.V) <: Array{String,1}
testval &= A.type == eltype(C[1])
fulltest &= testfct(testval,"largeEnv(integer)")

A,B = largeEnv(ComplexF64,Ns)
testval = typeof(A.V) <: Array{String,1}
testval &= A.type == ComplexF64
fulltest &= testfct(testval,"largeEnv(type,integer)")

Lenv,Renv = A,B

println()

try
  psi[3]
  mpo[3]
  Lenv[3]
  Renv[3]
  global testval = true
catch
  global testval = true
end
fulltest &= testfct(testval,"getindex([largeMPS,largeMPO,largeEnv])")

println()

X = loadMPS(length(psi))
testval = isapprox(Array(X[4]),Array(psi[4]))
fulltest &= testfct(testval,"loadMPS(integer)")

println()

Y = loadMPO(length(mpo))
testval = isapprox(Array(Y[4]),Array(mpo[4]))
fulltest &= testfct(testval,"loadMPO(integer)")

println()

Y = loadLenv(length(mpo))
testval = isapprox(Array(Y[4]),Array(Lenv[4]))
fulltest &= testfct(testval,"loadLenv(integer)")

println()

Y = loadRenv(length(mpo))
testval = isapprox(Array(Y[4]),Array(Renv[4]))
fulltest &= testfct(testval,"loadRenv(integer)")

println()

Y,Z = loadEnv(length(mpo))
testval = isapprox(Array(Y[4]),Array(Lenv[4]))
testval &= isapprox(Array(Z[4]),Array(Renv[4]))
fulltest &= testfct(testval,"loadEnv(integer)")

println()

C = rand(20,4,20)
psi[3] = C
testval = isapprox(psi[3],C)
fulltest &= testfct(testval,"setindex(largeMPS)")


mpo[3] = C
testval = isapprox(mpo[3],C)
fulltest &= testfct(testval,"setindex(largeMPO)")

Lenv[3] = C
testval = isapprox(Lenv[3],C)
fulltest &= testfct(testval,"setindex(largeEnv)")

println()

testval = length(psi) == Ns
fulltest &= testfct(testval,"length(largeMPS)")

testval = length(mpo) == Ns
fulltest &= testfct(testval,"length(largeMPO)")

testval = length(Lenv) == Ns
fulltest &= testfct(testval,"length(largeEnv)")

println()

testval = eltype(psi) == Float64
fulltest &= testfct(testval,"eltype(largeMPS)")

testval = eltype(mpo) == Float64
fulltest &= testfct(testval,"eltype(largeMPO)")

testval = eltype(Lenv) == ComplexF64
fulltest &= testfct(testval,"eltype(largeEnv)")

println()

try
  copy(["altmps_$i" for i = 1:length(psi)],psi)
  global testval = true
catch
  global testval = false
end
fulltest &= testfct(testval,"copy(Array{String},largeMPS)")

try
  copy("alt",psi)
  global testval = true
catch
  global testval = false
end
fulltest &= testfct(testval,"copy(String,largeMPS)")


try
  copy(["altmpo_$i" for i = 1:length(mpo)],mpo)
  global testval = true
catch
  global testval = false
end
fulltest &= testfct(testval,"copy(Array{String},largeMPO)")

try
  copy("alt",mpo)
  global testval = true
catch
  global testval = false
end
fulltest &= testfct(testval,"copy(String,largeMPO)")


try
  copy(["altenv_$i" for i = 1:length(Lenv)],Lenv)
  global testval = true
catch
  global testval = false
end
fulltest &= testfct(testval,"copy(Array{String},largeEnv)")

try
  copy("alt",Lenv)
  global testval = true
catch
  global testval = false
end
fulltest &= testfct(testval,"copy(String,largeEnv)")



rm("del"*DMRjulia.file_extension)
for i = 1:Ns
  rm("mps_$i"*DMRjulia.file_extension)
  rm("mpo_$i"*DMRjulia.file_extension)
  rm("Lenv_$i"*DMRjulia.file_extension)
  rm("Renv_$i"*DMRjulia.file_extension)

  rm("altLenv_$i"*DMRjulia.file_extension)

  rm("altmps_$i"*DMRjulia.file_extension)
  rm("altmpo_$i"*DMRjulia.file_extension)
  rm("altenv_$i"*DMRjulia.file_extension)
#  rm("altRenv_$i.dmrjulia")
end
