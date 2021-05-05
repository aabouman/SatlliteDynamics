function computeCost(ctrl::MPCController{OSQP.Model}, X::Vector, U::Vector)
    pW = ctrl.Q[1:3,1:3]; qW = mean(ctrl.Q[4,4], ctrl.Q[5,5], ctrl.Q[6,6], ctrl.Q[7,7])
    vW = ctrl.Q[8:10,8:10]; wW = ctrl.Q[11:13,11:13]

    pWf = ctrl.Qf[1:3,1:3]; qWf = mean(ctrl.Qf[4,4], ctrl.Qf[5,5], ctrl.Qf[6,6], ctrl.Qf[7,7])
    vWf = ctrl.Qf[8:10,8:10]; qWf = ctrl.Qf[11:13,11:13]

    R = ctrl.R

    distGeo(q₁, q₂) = min(1 - q₁' * q₂, 1 + q₁' * q₂)

    (X[i][1:3]' * pW * X[i][1:3] + qW * distGeo(ctrl.Xref[i][], q₂)  +
     X[i][8:10]' * vW * X[i][8:10] + X[i][11:13]' * wW * X[i][11:13])
end
