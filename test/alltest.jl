
include("../DMRjulia.jl")

function testfct(test::Bool,message::String)
#  try test
  if test
    println("PASS "*message)
  else
    println("FAIL "*message)
  end
#  catch
#    error(message)
#  end
  return test
end

function checkall(fulltestrecord::Array{Bool,1},i::Integer,fulltest::Bool)
  fulltestrecord[i] = fulltest
  println("All tests passed? ",fulltest)
end

tests = [
"tensor_test.jl",
"QN_test.jl",
"Qtensor_test.jl",
"libalg_test.jl",
"dense_contract_test.jl",
"dense_decomposition_test.jl",
"contract_time.jl",
"svdtest.jl",
"MPtask_test.jl",
"MPmeas_test.jl",
"MPlarge_test.jl",
"MPOauto_test.jl",
"model_test.jl",
"Qlinearalgebra.jl",
"tensornetwork_test.jl"]

fulltestrecord = Array{Bool,1}(undef,length(tests))

for i = 1:length(tests)
  @time include(tests[i])
  checkall(fulltestrecord,i,fulltest)
end

println()

for i = 1:length(tests)
  println(i," ",fulltestrecord[i],"     ",tests[i])
end

println()

if sum(fulltestrecord) == length(tests)
  println("All passed. Good work. We happy :^)")
else
  println("These passed:")
  println(tests[fulltestrecord])
  println()
  println("These did not pass:")
  notfulltestrecord = [!fulltestrecord[w] for w = 1:length(tests)]
  println(tests[notfulltestrecord])
end
