import pymc3 as pm
import numpy as np
import theano.tensor as tt
import matplotlib
import matplotlib.pyplot as plt
import arviz as az
import scipy.io as spio


class LogLike(tt.Op): # ????

    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, data, sigma):

        #Initialise the Op with various things that our log-likelihood function requires.

        # add inputs as class attributes
        self.likelihood = loglike
        self.data = data
        self.sigma = sigma

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (conductivity,) = inputs  # this will contain my variables ???

        # call the log-likelihood function
        logl = self.likelihood(conductivity, self.data, self.sigma)

        outputs[0][0] = np.array(logl)  # output the log-likelihood

def my_model(conductivity):

    import matlab.engine

    eng = matlab.engine.start_matlab()
    test_cond = matlab.double(conductivity.tolist()) #converts input into usable matlab format
    voltages = eng.forward_solver(test_cond) #uses forward_solver.m file in matlab, in github ; must be in python project folder
    eng.quit()

    # return the numpy array
    return voltages


# define your really-complicated likelihood function that uses loads of external codes
def my_loglike(conductivity, data, sigma):

    voltages = my_model(conductivity)
    loglik = np.log(sigma) - .5 * np.log(2 * 3.14159) - .5 * np.sum((data - voltages) ** 2)

    return loglik



mat2 = spio.loadmat('C:/Users/krait/volt2data.mat', squeeze_me=True) #file output from matlab with "save", in github, replace with correct path
data = mat2['volt2']
data = data.tolist()
sigma = 1

# create our Op
logl = LogLike(my_loglike, data, sigma)

# use PyMC3 to sample from log-likelihood
with pm.Model():
    cond = pm.Normal("cond", mu=0, sigma=1, shape=64)

    # convert conductivity to a tensor vector
    parameter = tt.as_tensor_variable(cond)

    # use a DensityDist (use a lamdba function to "call" the Op) ????
    pm.DensityDist("likelihood", lambda v: logl(v), observed={"v": parameter})

    trace = pm.sample(50, tune=25, discard_tuned_samples=True)

# plot the traces
_ = pm.traceplot(trace)

# put the chains in an array (for later!)
samples_pymc3 = np.vstack((trace["m"], trace["c"])).T


# sigma = pm.HalfNormal("sigma", sigma=1)
# will eventually have sigma as random variable, for now just constant