import pymc3 as pm
import numpy as np
import theano.tensor as tt
import matplotlib
import matplotlib.pyplot as plt
import arviz as az
import scipy.io as spio
import warnings
import matlab.engine

engine = matlab.engine.start_matlab()


class LogLikeWithGrad(tt.Op): # ????

    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, data, sigma, engine):

        #Initialise the Op with various things that our log-likelihood function requires.

        # add inputs as class attributes
        self.likelihood = loglike
        self.data = data
        self.sigma = sigma
        self.engine = engine

        # initialise the gradient Op (below)
        self.logpgrad = LogLikeGrad(self.likelihood, self.data, self.sigma, self.engine)

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (conductivity,) = inputs  # this will contain my variables ???

        # call the log-likelihood function
        logl = self.likelihood(conductivity, self.data, self.sigma, self.engine)

        outputs[0][0] = np.array(logl)  # output the log-likelihood

    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values
        (conductivity,) = inputs  # our parameters
        return [g[0] * self.logpgrad(conductivity)]

class LogLikeGrad(tt.Op):
    """
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.
    """

    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def __init__(self, loglike, data, sigma, engine):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that out function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.data = data
        self.sigma = sigma
        self.engine = engine

    def perform(self, node, inputs, outputs):
        (conductivity,) = inputs

        # define version of likelihood function to pass to derivative function
        def lnlike(values):
            return self.likelihood(values, self.data, self.sigma, self.engine)

        # calculate gradients
        grads = gradients(conductivity, lnlike)

        outputs[0][0] = grads

def my_model(conductivity, engine):

    eng = engine
    test_cond = matlab.double(conductivity.tolist()) #converts input into usable matlab format
    forward = eng.forward_solver(test_cond) #uses forward_solver.m file in matlab, in github ; must be in python project folder
    voltages = forward['meas']
    print(voltages)

    # return the numpy array
    return voltages

def gradients(vals, func, releps=1e-3, abseps=None, mineps=1e-9, reltol=1e-3,
              epsscale=0.5):
    """
    Calculate the partial derivatives of a function at a set of values. The
    derivatives are calculated using the central difference, using an iterative
    method to check that the values converge as step size decreases.

    Parameters
    ----------
    vals: array_like
        A set of values, that are passed to a function, at which to calculate
        the gradient of that function
    func:
        A function that takes in an array of values.
    releps: float, array_like, 1e-3
        The initial relative step size for calculating the derivative.
    abseps: float, array_like, None
        The initial absolute step size for calculating the derivative.
        This overrides `releps` if set.
        `releps` is set then that is used.
    mineps: float, 1e-9
        The minimum relative step size at which to stop iterations if no
        convergence is achieved.
    epsscale: float, 0.5
        The factor by which releps if scaled in each iteration.

    Returns
    -------
    grads: array_like
        An array of gradients for each non-fixed value.
    """

    grads = np.zeros(len(vals))

    # maximum number of times the gradient can change sign
    flipflopmax = 10.

    # set steps
    if abseps is None:
        if isinstance(releps, float):
            eps = np.abs(vals)*releps
            eps[eps == 0.] = releps  # if any values are zero set eps to releps
            teps = releps*np.ones(len(vals))
        elif isinstance(releps, (list, np.ndarray)):
            if len(releps) != len(vals):
                raise ValueError("Problem with input relative step sizes")
            eps = np.multiply(np.abs(vals), releps)
            eps[eps == 0.] = np.array(releps)[eps == 0.]
            teps = releps
        else:
            raise RuntimeError("Relative step sizes are not a recognised type!")
    else:
        if isinstance(abseps, float):
            eps = abseps*np.ones(len(vals))
        elif isinstance(abseps, (list, np.ndarray)):
            if len(abseps) != len(vals):
                raise ValueError("Problem with input absolute step sizes")
            eps = np.array(abseps)
        else:
            raise RuntimeError("Absolute step sizes are not a recognised type!")
        teps = eps

    # for each value in vals calculate the gradient
    count = 0
    for i in range(len(vals)):
        # initial parameter diffs
        leps = eps[i]
        cureps = teps[i]

        flipflop = 0

        # get central finite difference
        fvals = np.copy(vals)
        bvals = np.copy(vals)

        # central difference
        fvals[i] += 0.5*leps  # change forwards distance to half eps
        bvals[i] -= 0.5*leps  # change backwards distance to half eps
        cdiff = (func(fvals)-func(bvals))/leps

        while 1:
            fvals[i] -= 0.5*leps  # remove old step
            bvals[i] += 0.5*leps

            # change the difference by a factor of two
            cureps *= epsscale
            if cureps < mineps or flipflop > flipflopmax:
                # if no convergence set flat derivative (TODO: check if there is a better thing to do instead)
                warnings.warn("Derivative calculation did not converge: setting flat derivative.")
                grads[count] = 0.
                break
            leps *= epsscale

            # central difference
            fvals[i] += 0.5*leps  # change forwards distance to half eps
            bvals[i] -= 0.5*leps  # change backwards distance to half eps
            cdiffnew = (func(fvals)-func(bvals))/leps

            if cdiffnew == cdiff:
                grads[count] = cdiff
                break

            # check whether previous diff and current diff are the same within reltol
            rat = (cdiff/cdiffnew)
            if np.isfinite(rat) and rat > 0.:
                # gradient has not changed sign
                if np.abs(1.-rat) < reltol:
                    grads[count] = cdiffnew
                    break
                else:
                    cdiff = cdiffnew
                    continue
            else:
                cdiff = cdiffnew
                flipflop += 1
                continue

        count += 1

    return grads

# define your really-complicated likelihood function that uses loads of external codes
def my_loglike(conductivity, data, sigma, engine):

    voltages = my_model(conductivity, engine)
    loglik = -.5*np.log(sigma) - .5 * np.log(2 * 3.14159) - .5 * np.sum((data - voltages) ** 2) #log likelihood equation

    return loglik


mat2 = spio.loadmat('C:/Users/krait/volt2data.mat', squeeze_me=True) #file output from matlab with "save", in github, replace with correct path
data = mat2['volt2']
#data = data.tolist()
sigma = 1.2

# create our Op
logl = LogLikeWithGrad(my_loglike, data, sigma, engine)
basic_model = pm.Model()
# use PyMC3 to sample from log-likelihood
with basic_model:
    cond = pm.Normal("cond", mu=1, sigma=1, shape=64)

    # convert conductivity to a tensor vector
    parameter = tt.as_tensor_variable(cond)

    # use a DensityDist (use a lamdba function to "call" the Op) ????
    pm.DensityDist("likelihood", lambda v: logl(v), observed={"v": parameter})

    #trace = pm.sample(10) #draw samples from posterior


map_estimate = pm.find_MAP(model=basic_model)
map_estimate
eng.quit()

# plot the traces
_ = pm.traceplot(trace)

# put the chains in an array (for later!)
#samples_pymc3 = np.vstack((trace["m"], trace["c"])).T


# sigma = pm.HalfNormal("sigma", sigma=1)
# will eventually have sigma as random variable, for now just constant