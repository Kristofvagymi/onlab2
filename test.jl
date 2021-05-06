using Plots
using Flux, DiffEqFlux
using DifferentialEquations

x = 0:0.01:10
y = sin.(x) .* x
z = sin.(x) .* x .^  2

plot(y)
plot!(z)



using Flux, DiffEqFlux
using DifferentialEquations

function lotka_volterra(du,u,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end
u0 = [1.0,1.0]
tspan = (0.0,10.0)
p = [1.5,1.0,3.0,1.0]
prob = ODEProblem(lotka_volterra,u0,tspan,p)

sol = solve(prob,Tsit5(),saveat=0.1)

plot(sol)


using Flux, DiffEqFlux
p = [2.2, 1.0, 2.0, 0.4] # Initial Parameter Vector
params = Flux.params(p)

function predict_rd() # Our 1-layer "neural network"
  solve(prob,Tsit5(),p=p,saveat=0.1)[1,:] # override with new parameters
end

loss_rd() = sum(abs2,x-1 for x in predict_rd()) # loss function

data = Iterators.repeated((), 100)
opt = ADAM(0.1)
cb = function () #callback function to observe training
  display(loss_rd())
end

# Display the ODE with the initial parameter values.
cb()

Flux.train!(loss_rd, params, data, opt, cb = cb)


u0 = Float32[2.; 0.]
datasize = 30
tspan = (0.0f0,1.5f0)

function trueODEfunc(du,u,p,t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end
t = range(tspan[1],tspan[2],length=datasize)

prob = ODEProblem(trueODEfunc,u0,tspan)
ode_data = solve(prob,Tsit5(),saveat=t)


function predict_n_ode()
  n_ode(u0)
end

cb = function () #callback function to observe training
  display(loss_n_ode())
  cur_pred = predict_n_ode()
  pl = scatter(ode_data[1,:],label="data")
  scatter!(pl,cur_pred[1,:],label="prediction")
  display(plot(pl))
end

dudt = Chain(x -> x.^3,Dense(2,50,tanh), Dense(50,2))

n_ode = NeuralODE(dudt,tspan,Tsit5(),saveat=t, reltol=1e-7,abstol=1e-9)

n_ode(u0)

ps = Flux.params(n_ode)
data = Iterators.repeated((), 100)
opt = ADAM(0.1)
loss_n_ode() = sum(abs2,ode_data .- predict_n_ode())

Flux.train!(loss_n_ode, ps, data, opt, cb = cb)
