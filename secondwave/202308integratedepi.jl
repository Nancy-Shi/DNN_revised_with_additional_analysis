##
# Loading Packages and setup random seeds
using Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, Random, Plots
using Flux
using DataFrames
using CSV
using ComponentArrays
using OptimizationOptimisers
rng = Random.default_rng()
Random.seed!(rng, 1024)

# load training dta

data = DataFrame(CSV.File("./output/datasmoothing.csv"))
choosentime = range(366, 425)
choosencolumn = [3, 15] # vaccined individuals, cases
trainingdata = Array(data[choosentime, choosencolumn])'
datascale = Array(data[choosentime, 2:end])'

dailycase=Array(data[choosentime, 2])
plot(dailycase)

dailyvac=Array(data[choosentime, 14])
display(plot(dailyvac))

# initial value of M1,C1,M2,C2
intercolumn = [10, 11] # M, C score vaccine
interdata = Array(data[choosentime, intercolumn])'
m10, c10 = Float32.(Array(interdata[:, 1]))

vaccolumn = [12, 13] # M, C score vaccine
vacdata = Array(data[choosentime, vaccolumn])'
m20, c20 = Float32.(Array(vacdata[:, 1]))





# reload M1,C1,M2,C2
using BSON: @load
@load "./output/anninter_second.bson" dudt2
anninter = dudt2
@load "./output/anninterpara_second.bson" psave
pintersave = psave
pinter, stinter = Lux.setup(rng, anninter)
pinterinit = ComponentArray(pinter)
pinterfinal = ComponentArray(pintersave, getaxes(pinterinit))

@load "./output/annvac_second.bson" dudt2
annvac = dudt2
@load "./output/annvacpara_second.bson" psave
pvacsave = psave
pvac, stvac = Lux.setup(rng, annvac)
pvacinit = ComponentArray(pvac)
pvacfinal = ComponentArray(pvacsave, getaxes(pvacinit))

# set up neural differential equation models
N = 38250000.0f0
σ1 = 0.19f0
γ1 = 0.1f0
hi0 = Float32(trainingdata[1, 1])
hv0 = Float32(trainingdata[2, 1])
v0 = Float32(trainingdata[2, 1])
e0 = Float32(datascale[1, 1] / σ1)
i0 = Float32(datascale[1, 1] / γ1)
u0 = [N, v0, e0, i0, hi0, hv0, m10, c10, m20, c20]
datasize = 60
tspan = (0.0f0, 60.0f0)
tsteps = range(tspan[1], tspan[2], length=datasize)
ann1 = Flux.Chain(Flux.Dense(2, 64, tanh), Flux.Dense(64, 1))
ann2 = Flux.Chain(Flux.Dense(2, 64, tanh), Flux.Dense(64, 1))
p1, re1 = Flux.destructure(ann1)
p2, re2 = Flux.destructure(ann2)
p1 = Float32.(p1)
p2 = Float32.(p2)
re1(p1)([m10, c10])
re2(p2)([m20, c20])

function SVEIR_nn(du, u, p, t)
    S, V, E, I, HI, HV, M1, C1, M2, C2 = u
    N = 38250000.0f0
    ϵ = 0.8f0
    σ1 = 0.19f0
    γ1 = 0.1f0
    β = min(5.0f0, abs(re1(p[1:length(p1)])([M1, C1])[1]))
    ν = abs(re2(p[(length(p1)+1):end])([M2, C2])[1])
    du[1] = -β * I * S / N - ν * S
    du[2] = ν * S - (1.0f0 - ϵ) * β * I * V / N
    du[3] = β * I * S / N + (1.0f0 - ϵ) * β * I * V / N - σ1 * E
    du[4] = σ1 * E - γ1 * I
    du[5] = β * I * S / N + (1.0f0 - ϵ) * β * I * V / N
    du[6] = ν * S
    du[7] = anninter([M1, C1], pinterfinal, stinter)[1][1]
    du[8] = anninter([M1, C1], pinterfinal, stinter)[1][2]
    du[9] = annvac([M2, C2], pvacfinal, stvac)[1][1]
    du[10] = annvac([M2, C2], pvacfinal, stvac)[1][2]
end
prob_neuralode = ODEProblem(SVEIR_nn, u0, tspan, [p1; p2])


# simulate the neural differential equation models
function predict_neuralode(θ)
    #Array(prob_neuralode(u0, p, st)[1])
    prob = remake(prob_neuralode, p=θ)
    Array(solve(prob, Vern7(), saveat=tsteps))
end
p = [p1; p2]
predict_neuralode(p)[5:6, :]
size(predict_neuralode(p)[5:6, :]) == size(trainingdata)

sol = predict_neuralode(p)
pltinter = scatter(tsteps, interdata[1, :], label="Minter data")
scatter!(tsteps, interdata[2, :], label="Cinter data")
plot!(pltinter, tsteps, sol[7, :], label="Minter prediction")
plot!(pltinter, tsteps, sol[8, :], label="Cinter prediction")
display(plot(pltinter))
pltvac = scatter(tsteps, vacdata[1, :], label="Mvac data")
scatter!(tsteps, vacdata[2, :], label="Cvac data")
plot!(pltvac, tsteps, sol[9, :], label="Mvac prediction")
plot!(pltvac, tsteps, sol[10, :], label="Cvac prediction")
display(plot(pltvac))

# loss function and callbacks

function loss_neuralode(p)
    pred = predict_neuralode(p)[5:6, :]
    loss = sum(abs2, log.(trainingdata) .- log.(pred))
    return loss, pred
end

loss_neuralode(p)


callback = function (p, l, pred; doplot=false)
    println(l)
    # plot current prediction against data
    if doplot
        plt = scatter(tsteps, trainingdata[1, :], label="Accumulated cases")
        scatter!(tsteps, trainingdata[2, :], label="Accumulated vaccinated individuals")
        plot!(plt, tsteps, pred[1, :], label="Predicted accumulated cases")
        plot!(plt, tsteps, pred[2, :], label="Predicted accumulated vaccinated individuals")
        display(plot(plt))
    end
    return false
end

pinit = [p1; p2]
callback(pinit, loss_neuralode(pinit)...; doplot=true)

# use Optimization.jl to solve the problem
##
@load "./output/ann12epipfinal_second.bson" psave
pinit=psave
adtype = Optimization.AutoForwardDiff()

optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

result_neuralode2 = Optimization.solve(optprob,
    OptimizationOptimisers.ADAM(0.001),
    callback=callback,
    maxiters=10)
optprob2 = remake(optprob, u0=result_neuralode2.u)

result_neuralode2 = Optimization.solve(optprob2,
    Optim.BFGS(initial_stepnorm=0.001),
    callback=callback)




optprob2 = remake(optprob2, u0=result_neuralode2.u)

result_neuralode2 = Optimization.solve(optprob2,
    Optimisers.ADAM(0.0001),
    maxiters
    =3000,
    callback=callback,
    allow_f_increases=false)

optprob2 = remake(optprob2, u0=result_neuralode2.u)
result_neuralode2 = Optimization.solve(optprob2,
    Optim.BFGS(initial_stepnorm=0.01),
    callback=callback,
    maxiters=50,
    allow_f_increases=false)

optprob2 = remake(optprob2, u0=result_neuralode2.u)

result_neuralode2 = Optimization.solve(optprob2,
    Optimisers.ADAM(0.000001),
    maxiters
    =300,
    callback=callback,
    allow_f_increases=false)

optprob2 = remake(optprob2, u0=result_neuralode2.u)
result_neuralode2 = Optimization.solve(optprob2,
    Optim.BFGS(initial_stepnorm=0.01),
    callback=callback,
    maxiters=50,
    allow_f_increases=false)
optprob2 = remake(optprob2, u0=result_neuralode2.u)
result_neuralode2 = Optimization.solve(optprob2,
    Optimisers.ADAM(0.000001),
    maxiters
    =10000,
    callback=callback,
    allow_f_increases=false)

optprob2 = remake(optprob2, u0=result_neuralode2.u)
result_neuralode2 = Optimization.solve(optprob2,
    Optim.BFGS(initial_stepnorm=0.0001),
    callback=callback,
    maxiters=50,
    allow_f_increases=false)

optprob2 = remake(optprob, u0=result_neuralode2.u)

result_neuralode2 = Optimization.solve(optprob2,
    Optim.LBFGS(),
    callback=callback,
    allow_f_increases=false)
optprob2 = remake(optprob, u0=result_neuralode2.u)

result_neuralode2 = Optimization.solve(optprob2,
    Optimisers.ADAM(0.001),
    maxiters=300,
    callback=callback,
    allow_f_increases=false)
optprob2 = remake(optprob, u0=result_neuralode2.u)

result_neuralode2 = Optimization.solve(optprob2,
    Optim.LBFGS(),
    callback=callback,
    allow_f_increases=false)

pfinal = result_neuralode2.u

callback(pfinal, loss_neuralode(pfinal)...; doplot=true)

##
# beta(t), nu(t) function
probfinal = remake(prob_neuralode, p=pfinal)
solfinal = solve(probfinal, Vern7(), saveat=tsteps)
β(t) = min(5.0f0, abs(re1(pfinal[1:length(p1)])([solfinal(t)[7], solfinal(t)[8]])[1]))
ν(t) = abs(re2(pfinal[(length(p1)+1):end])([solfinal(t)[9], solfinal(t)[10]])[1])


#β(t) = min(5.0f0, abs(re1(pfinal[1:length(p1)])([t])[1]))
#ν(t) = abs(re2(pfinal[(length(p1)+1):end])([t])[1])



display(plot(tsteps, β.(tsteps)))



display(plot(tsteps, ν.(tsteps)))
##
# Save neural network architechtures and 
using BSON: @save
@save "./output/ann1epi_second.bson" ann1
@save "./output/ann2epi_second.bson" ann2
psave = pfinal
@save "./output/ann12epipfinal_second.bson" psave
pred = predict_neuralode(pfinal)[5:6, :]
plt = scatter(tsteps, trainingdata[1, :], label="Accumulated cases")
plot!(plt, tsteps, pred[1, :], label="Predicted accumulated cases")
display(plot(plt))
savefig("./output/ann12epicase_second.png")
plt = scatter(tsteps, trainingdata[2, :], label="Accumulated vaccinated individuals")
#scatter!(tsteps, trainingdata[3, :], label="Minter data")
#scatter!(tsteps, trainingdata[4, :], label="Cinter data")

plot!(plt, tsteps, pred[2, :], label="Predicted accumulated vaccinated individuals")
#plot!(plt, tsteps, pred[3, :], label="Minter prediction")
#plot!(plt, tsteps, pred[4, :], label="Cinter prediction")
display(plot(plt))
savefig("./output/ann12epivac_second.png")

neuralepi = DataFrame()
neuralepi[!, "date"] = data[choosentime, 1]
neuralepi[!, "Case"] = trainingdata[1, :]
neuralepi[!, "Vaccine"] = trainingdata[2, :]
neuralepi[!, "PredCase"] = pred[1, :]
neuralepi[!, "PredVaccine"] = pred[2, :]
neuralepi[!, "beta"] = β.(tsteps)
neuralepi[!, "nu"] = ν.(tsteps)
CSV.write("./output/neuralepi12_second.csv", neuralepi)

##