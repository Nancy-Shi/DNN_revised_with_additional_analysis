##
# Loading Packages and setup random seeds
using Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, Random, Plots
using DataFrames
using CSV
using ComponentArrays
rng = Random.default_rng()
Random.seed!(rng, 131)
# load training dta

data = DataFrame(CSV.File("./output/datasmoothing.csv"))
data = DataFrame(CSV.File("./output/data.csv"))

choosentime = range(366, 425)

choosencolumn = [10, 11] # M, C score intervention

trainingdata = Array(data[choosentime, choosencolumn])'



# set up neural differential equation models
u0 = Array(trainingdata[:, 1])
datasize = length(choosentime)
tspan = (0.0f0, Float32(datasize))
tsteps = range(tspan[1], tspan[2], length=datasize)
dudt2 = Lux.Chain(Lux.Dense(2, 32, tanh), Lux.Dense(32, 2))
p, st = Lux.setup(rng, dudt2)
#prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat=tsteps)
prob_neuralode = ODEProblem((u, p, t) -> dudt2(u, p, st)[1], u0, tspan, ComponentArray(p))


# simulate the neural differential equation models
function predict_neuralode(θ)
    #Array(prob_neuralode(u0, p, st)[1])
    prob = remake(prob_neuralode, p=θ)
    Array(solve(prob, Tsit5(), saveat=tsteps))
end

predict_neuralode(p)
size(predict_neuralode(p)) == size(trainingdata)
# loss function and callbacks

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, trainingdata[:,1:4:end] .- pred[:,1:4:end])
    return loss, pred
end

loss_neuralode(p)

callback = function (p, l, pred; doplot=false)
    println(l)
    # plot current prediction against data
    if doplot
        plt = scatter(tsteps, trainingdata[1, :], label="Minter data")
        scatter!(tsteps, trainingdata[2, :], label="Cinter data")
        plot!(plt, tsteps, pred[1, :], label="Minter prediction")
        plot!(plt, tsteps, pred[2, :], label="Cinter prediction")
        display(plot(plt))
    end
    return false
end

pinit = ComponentArray(p)
callback(pinit, loss_neuralode(pinit)...; doplot=true)

# use Optimization.jl to solve the problem
##
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

using OptimizationOptimisers

result_neuralode = Optimization.solve(optprob,
    Optimisers.ADAM(0.01),
    callback=callback,
    maxiters=500)

optprob2 = remake(optprob, u0=result_neuralode.u)

result_neuralode2 = Optimization.solve(optprob2,
    Optim.BFGS(initial_stepnorm=0.01),
    callback=callback,
    allow_f_increases=false)

optprob2 = remake(optprob, u0=result_neuralode2.u)

result_neuralode2 = Optimization.solve(optprob2,
    Optimisers.ADAM(0.001),
    maxiters=100000,
    callback=callback,
    allow_f_increases=false)


optprob2 = remake(optprob, u0=result_neuralode2.u)

result_neuralode2 = Optimization.solve(optprob2,
    Optim.BFGS(initial_stepnorm=0.01),
    maxiters=300,
    callback=callback,
    allow_f_increases=false)

optprob2 = remake(optprob, u0=result_neuralode2.u)


result_neuralode2 = Optimization.solve(optprob2,
    Optimisers.ADAM(0.005),
    maxiters=300,
    callback=callback,
    allow_f_increases=false)

optprob2 = remake(optprob, u0=result_neuralode2.u)

result_neuralode2 = Optimization.solve(optprob2,
    Optim.BFGS(initial_stepnorm=0.01),
    callback=callback,
    allow_f_increases=false)

pfinal = result_neuralode2.u

callback(pfinal, loss_neuralode(pfinal)...; doplot=true)

##

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, trainingdata .- pred)
    return loss, pred
end
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, result_neuralode2.u)
using OptimizationOptimisers

result_neuralode = Optimization.solve(optprob,
    Optimisers.ADAM(0.01),
    callback=callback,
    maxiters=500)

optprob2 = remake(optprob, u0=result_neuralode.u)

result_neuralode2 = Optimization.solve(optprob2,
    Optim.BFGS(initial_stepnorm=0.01),
    callback=callback,
    allow_f_increases=false)

optprob2 = remake(optprob, u0=result_neuralode2.u)

result_neuralode2 = Optimization.solve(optprob2,
    Optimisers.ADAM(0.001),
    maxiters=10000,
    callback=callback,
    allow_f_increases=false)
pfinal = result_neuralode2.u
callback = function (p, l, pred; doplot=false)
    println(l)
    # plot current prediction against data
    if doplot
        plt = scatter(tsteps, trainingdata[1, :], label="Minter data")
        scatter!(tsteps, trainingdata[2, :], label="Cinter data")
        plot!(plt, tsteps, pred[1, :], label="Minter prediction")
        plot!(plt, tsteps, pred[2, :], label="Cinter prediction")
        display(plot(plt))
    end
    return false
end

callback(pfinal, loss_neuralode(pfinal)...; doplot=true)
##
using BSON: @save
@save "./output/anninter_second.bson" dudt2
psave = collect(pfinal)
@save "./output/anninterpara_second.bson" psave
pred = predict_neuralode(pfinal)
plt = scatter(tsteps, trainingdata[1, :], label="Minter Score")
scatter!(tsteps, trainingdata[2, :], label="Cinter Score")
#scatter!(tsteps, trainingdata[3, :], label="Minter data")
#scatter!(tsteps, trainingdata[4, :], label="Cinter data")
plot!(plt, tsteps, pred[1, :], label="Minter Score prediction")
plot!(plt, tsteps, pred[2, :], label="Cinter Score prediction")
#plot!(plt, tsteps, pred[3, :], label="Minter prediction")
#plot!(plt, tsteps, pred[4, :], label="Cinter prediction")
display(plot(plt))
savefig("./output/anninter_second.png")
neuralinter = DataFrame()
neuralinter[!, "date"] = data[choosentime, 1]
neuralinter[!, "MScoreInter"] = trainingdata[1, :]
neuralinter[!, "CScoreInter"] = trainingdata[2, :]
neuralinter[!, "PredMScoreInter"] = pred[1, :]
neuralinter[!, "PredCScoreInter"] = pred[2, :]
CSV.write("./output/neuralintervention_second.csv", neuralinter)
##