
println("#            +-------------+")
println("#>-----------|    QNs.jl   |-----------<")
println("#            +-------------+")
fulltest = true

@makeQNs "testQN" U1 U1 Zn{3}


oneQN = testQN()
testval = typeof(oneQN.val1) == U1 && typeof(oneQN.val2) == U1 && typeof(oneQN.val3) == Zn{3}
fulltest &= testfct(testval,"@composeQNs")

println()

testval = oneQN[1] == 0 && oneQN[2] == 0 && oneQN[3] == 0
fulltest &= testfct(testval,"getindex(QNs")

println()

twoQN = testQN(3,1,2)
checkadd = oneQN + twoQN

testval = checkadd[1] == 3 && checkadd[2] == 1 && checkadd[3] == 2
fulltest &= testfct(testval,"add(QNs")

println()

checkinv = inv(twoQN)
testval = checkinv[1] == -3 && checkinv[2] == -1 && checkinv[3] == 1
fulltest &= testfct(testval,"inv(QNs")

println()

testval = oneQN != twoQN && oneQN < twoQN
fulltest &= testfct(testval,"<(QNs)")

println()

testval = oneQN != twoQN && inv(oneQN) > inv(twoQN)
fulltest &= testfct(testval,">(QNs)")

println("All tests passed? ",testval)
