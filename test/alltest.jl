
include("../DMRjulia.jl")

function testfct(test::Bool,message::String)
#  try test
  if test
    printstyled("PASS",color=:green)
  else
    printstyled("FAIL ",color=:red)
  end
  println(" "*message)
#  catch
#    error(message)
#  end
  return test
end

function checkall(fulltestrecord::Array{Bool,1},i::Integer,fulltest::Bool)
  fulltestrecord[i] = fulltest
  print("All tests passed? ")
  if fulltest
    printstyled(fulltest,color=:green)
  else
    printstyled(fulltest,color=:red)
  end
  println()
end

tests = [
"tensor_test.jl",
"QN_test.jl",
"Qtensor_test.jl",
"libalg_test.jl",
"dense_contract_test.jl",
"dense_decomposition_test.jl",
"Qlinearalgebra.jl",
"contract_time.jl",
"svdtest.jl",
"MPtask_test.jl",
"MPmeas_test.jl",
"MPlarge_test.jl",
"MPOauto_test.jl",
"model_test.jl",
"tensornetwork_test.jl"#,
######
######
######
#"dmrg_test.jl"
]

fulltestrecord = Array{Bool,1}(undef,length(tests))

for i = 1:length(tests)
  @time include(tests[i])
  checkall(fulltestrecord,i,fulltest)
end

println()

for i = 1:length(tests)
  if fulltestrecord[i]
    printstyled(fulltestrecord[i],color=:green)
  else
    printstyled(fulltestrecord[i],color=:red)
  end
  println("    ",i,"   ",tests[i])
end

println()

if sum(fulltestrecord) == length(tests)
  println("All passed. Good work. We happy :^)")
else
  println("These passed:")
  printstyled(tests[fulltestrecord],color=:green)
  println()
  println()
  println("These did not pass:")
  notfulltestrecord = [!fulltestrecord[w] for w = 1:length(tests)]
  printstyled(tests[notfulltestrecord],color=:red)
end
