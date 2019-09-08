import torch
from scipy.optimize import root_scalar
from scipy.stats import gaussian_kde


class Loss:
    terminal_layer = None
    N_OUTPUTS = None

    def __repr__(self):
        classname = self.__class__.__name__
        s = """<lifelike.%s, n_outputs=%d>""" % (
            classname, self.N_OUTPUTS
        )
        return s

    def cumulative_hazard(self, params, t):
        # must override this or survival_function
        return -torch.log(self.survival_function(params, t))

    def survival_function(self, params, t):
        # must override this or cumulative_hazard
        return torch.exp(-self.cumulative_hazard(params, t))

    def hazard(self, params, t):
        return self.cumulative_hazard(params, t)

    def log_hazard(self, params, t):
        return torch.log(torch.clamp(self.hazard(params, t), 1e-30))

    def inform(self, **kwargs):
        pass


class GeneralizedGamma(Loss):

    N_OUTPUTS = 3

    def __init__(self, topology):
        self.terminal_layer = [stax.Dense(self.N_OUTPUTS)]
        raise NotImplementedError("Jax still needs to have support for incomplete gamma function")

    def cumulative_hazard(self, params, t):
        pass

    def log_hazard(self, params, t):
        pass



class ParametricMixture(Loss):
    """

    ::math

        S(t | x) = p_1(x) S_{Weibull}(t | x) + p_2(x) S_{LogLogistic}(t | x) + p3(x)


    """
    N_OUTPUTS = 3 + 2 + 2

    def __init__(self):
        self.terminal_layer = [
            stax.Dense(self.N_OUTPUTS, W_init=stax.randn(1e-10), b_init=stax.randn(1e-10))
        ]

    def cumulative_hazard(self, params, t):
        # weights
        ln_p = params[:3] - torch.logsumexp(params[:3], -1)
        ln_p1, ln_p2, ln_p3 = ln_p
        # p1, p2, p3 = torch.clamp(torch.softmax(params[:3]), 1e-25)

        # weibull params
        lambda_, rho_ = torch.exp(params[3]), torch.exp(params[4])

        # loglogistic params
        # alpha_, beta_ = torch.exp(params[5]), torch.exp(params[6])
        ln_alpha, ln_beta = params[5:7]
        term2 = torch.log(t) - ln_alpha

        v = -torch.logsumexp(
            torch.hstack(
                (
                    ln_p1 - (t / lambda_) ** rho_,
                    ln_p2 - torch.logsumexp(torch.stack((0, torch.exp(ln_beta) * term2))),
                    ln_p3,
                )
            )
        )
        return v


class PiecewiseConstant(Loss):
    def __init__(self, breakpoints):
        self.N_OUTPUTS = len(breakpoints) + 1
        self.breakpoints = torch.hstack(([0], breakpoints, torch.tensor([np.inf])))
        self.terminal_layer = [
            stax.Dense(
                self.N_OUTPUTS, W_init=stax.randn(1e-7), b_init=stax.randn(1e-7)
            ),
            stax.Exp,
        ]

    def __repr__(self):
        try:
            classname = self.__class__.__name__
            s = """<lifelike.%s, breakpoints=%s>""" % (
                classname, self.breakpoints
            )
        except:
            s = """<lifelike.%s>""" % classname
        return s

    def cumulative_hazard(self, params, t):
        M = torch.minimum(self.breakpoints, t)
        M = torch.diff(M)
        return (M * params).sum()

    """
    def hazard(self, params, t):
        ix = torch.searchsorted(self.breakpoints, t)
        or
        ix = 0
        for tau in self.breakpoints:
            if t < tau:
                break
            ix += 1
        return params[ix]
    """


class NonParametric(PiecewiseConstant):
    """
    We create the concentration of breakpoints in proportional to the number of subjects that died around that time.
    See blog post at https://dataorigami.net/blogs/napkin-folding/non-parametric-survival-function-prediction
    """

    def __init__(self, n_breakpoints=None):
        self.n_breakpoints = n_breakpoints

    def inform(self, **kwargs):
        T = kwargs.pop("T")
        E = kwargs.pop("E")

        # first take a look at T, and create a KDE around the deaths
        breakpoints = self.create_breakpoints(T[E.astype(bool)])
        super(NonParametric, self).__init__(breakpoints)

    def create_breakpoints(self, observed_event_times):
        def solve_inverse_cdf_problem(f, fprime=None, starting_point=0):
            return root_scalar(f, x0=starting_point, fprime=fprime).root

        n_obs = observed_event_times.shape[0]
        dist = gaussian_kde(observed_event_times)

        if self.n_breakpoints is None:
            n_breakpoints = min(int(torch.sqrt(n_obs) / 2), torch.unique(observed_event_times).shape[0])
        else:
            n_breakpoints = self.n_breakpoints

        breakpoints = torch.empty(n_breakpoints)

        # We scale our pdf/cdf by CDF(max observed time) so that we will
        # never have breakpoints greater than the max observed time.
        # call this cdf'
        MAX = observed_event_times.max()
        CDF_M =  dist.integrate_box_1d(0, MAX)

        sol = 0
        for i, p in enumerate(torch.linspace(0, 1, n_breakpoints + 2)[1:-1]):
            # solve the following simple root problem:
            # cdf'(x) = p
            # cdf(x)/cdf(M) = p
            # cdf(x) = p * cdf(M)
            # cdf(x) - p*cdf(M) = 0
            sol = solve_inverse_cdf_problem(
                f=lambda x: dist.integrate_box_1d(0, x) / CDF_M - p,
                fprime=lambda x: dist(x) / CDF_M,
                starting_point=sol)
            breakpoints[i] = sol
        return breakpoints
