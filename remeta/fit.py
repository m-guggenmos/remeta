import timeit
import warnings
from functools import partial
from itertools import product
from multiprocessing import cpu_count

import numpy as np
try:  # only necessary if multiple cores should be used
    from multiprocessing_on_dill.pool import Pool as DillPool
except ModuleNotFoundError:
    pass
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize, Bounds, basinhopping
from scipy.optimize._constraints import old_bound_to_new  # noqa
from .util import TAB


_slsqp_epsilon = np.sqrt(np.finfo(float).eps)  # scipy's default value for the SLSQP epsilon parameter


def loop(fun, params, args, gridsize, verbose, param_id):
    # t0 = timeit.default_timer()
    if param_id and verbose and (np.mod(param_id, 1000) == 0):
        print(f'{TAB}{TAB}Grid iteration {param_id} / {gridsize}')
    result = fun(params[param_id], *args)
    # print(timeit.default_timer() - t0)
    return result


def fgrid(fun, valid, args, grid_multiproc, verbose=True):
    if grid_multiproc:
        with DillPool(cpu_count() - 1 or 1) as pool:
            ll_grid = pool.map(partial(loop, fun, valid, args, len(valid), verbose), range(len(valid)))
    else:
        ll_grid = [None] * len(valid)
        for i, param in enumerate(valid):
            ll_grid[i] = loop(fun, valid, args, len(valid), verbose, i)
    return ll_grid


def grid_search(x0, fun, args, param_set, ll_grid, valid, n_grid_candidates, n_grid_iter, grid_multiproc, verbose=True):
    n_grid_candidates_init = 3 * n_grid_candidates
    gx0 = x0
    gll_min_grid = np.min(ll_grid)
    grid_candidates = [valid[i] for i in np.argsort(ll_grid)[:n_grid_candidates_init]]
    previous_grid_range = [param_set.grid_range for _ in range(n_grid_candidates_init)]
    candidate_ids = list(range(n_grid_candidates_init))
    counter = 0
    for i in range(n_grid_iter):
        if verbose:
            print(f'{TAB}Grid iteration {i + 1} / {n_grid_iter}')
        gvalid = []
        valid_candidate_ids = []
        grid_range = [None] * len(grid_candidates) * 2
        for j, grid_candidate in enumerate(grid_candidates):
            grid_range[j] = [None] * param_set.nparams
            for k, p in enumerate(grid_candidate):
                ind = np.where(previous_grid_range[candidate_ids[j]][k] == p)[0][0]
                lb = previous_grid_range[candidate_ids[j]][k][max(0, ind - 1)]  # noqa
                ub = previous_grid_range[candidate_ids[j]][k][  # noqa
                    min(len(previous_grid_range[candidate_ids[j]][k]) - 1, ind + 1)]  # noqa
                grid_range[j][k] = np.around(np.linspace(lb, ub, 5), decimals=10)  # noqa
            gvalid_candidate = [p for p in product(*grid_range[j]) if  # noqa
                                np.all([con['fun'](p) >= 0 for con in param_set.constraints])]
            gvalid += gvalid_candidate
            valid_candidate_ids += [j] * len(gvalid_candidate)
            if verbose:
                print(f'{TAB}{TAB}Candidate {j + 1}: {[(p[0], p[-1]) for p in grid_range[j]]}')  # noqa
        gll_grid = fgrid(fun, gvalid, args, grid_multiproc, verbose=False)
        counter += len(gvalid)

        min_id = np.argmin(gll_grid)
        if gll_grid[min_id] < gll_min_grid:
            gll_min_grid = gll_grid[min_id]
            gx0 = gvalid[min_id]

        if verbose:
            print(f'{TAB}{TAB}Best fit: {gx0} (LL={gll_min_grid})')

        if i != n_grid_iter - 1:
            grid_candidates, candidate_ids = [], []
            for j in np.argsort(gll_grid):
                if gvalid[j] not in grid_candidates:
                    grid_candidates += [gvalid[j]]
                    candidate_ids += [valid_candidate_ids[j]]
                    if len(candidate_ids) == n_grid_candidates:
                        break
            previous_grid_range = grid_range
    fit = OptimizeResult(success=True, x=gx0, fun=gll_min_grid, nfev=counter)

    return fit


def fmincon(fun, param_set, args, gridsearch=False, grid_multiproc=True,
            gradient_free=False, global_minimization=False, fine_gridsearch=False, verbose=True,
            n_grid_candidates=10, n_grid_iter=3, gradient_method='slsqp', slsqp_epsilon=_slsqp_epsilon,
            init_nelder_mead=False, guess=None):

    t0 = timeit.default_timer()

    if verbose:
        negll_initial_guess = fun(param_set.guess, *args)
        print(f'Initial guess (neg. LL: {negll_initial_guess:.2f})')
        for i, p in enumerate(param_set.names):
            print(f'{TAB}[guess] {p}: {param_set.guess[i]:.4g}')

    bounds = Bounds(*old_bound_to_new(param_set.bounds), keep_feasible=True)
    if gridsearch:
        if len(param_set.constraints):
            valid = [p for p in product(*param_set.grid_range) if
                     np.all([con['fun'](p) >= 0 for con in param_set.constraints])]
        else:
            valid = list(product(*param_set.grid_range))
        if verbose:
            print(f"Grid search activated (grid size = {len(valid)})")
        t0 = timeit.default_timer()
        ll_grid = fgrid(fun, valid, args, grid_multiproc, verbose=verbose)
        x0 = valid[np.argmin(ll_grid)]
        ll_min_grid = np.min(ll_grid)
        grid_time = timeit.default_timer() - t0
        if verbose:
            for i, p in enumerate(param_set.names):
                print(f'{TAB}[grid] {p}: {x0[i]:.4g}')
            print(f"Grid neg. LL: {ll_min_grid:.1f}")
            print(f"Grid runtime: {grid_time:.2f} secs")
        fit_grid = OptimizeResult(success=True, x=x0, fun=ll_min_grid, nfev=len(valid))
    else:
        if guess is None:
            x0 = param_set.guess
        else:
            x0 = guess
        fit_grid = OptimizeResult(success=True, x=x0, fun=fun(x0, *args), nfev=1)

    if fine_gridsearch:
        x0 = grid_search(x0, fun, args, param_set, ll_grid, valid, n_grid_candidates, n_grid_iter,  # noqa
                         grid_multiproc, verbose=verbose).x

    if gradient_free:
        if global_minimization:
            if verbose:
                print('Performing global optimization')
            fit = basinhopping(fun, x0,
                               take_step=RandomDisplacementBoundsConstraints(bounds, param_set.constraints),
                               accept_test=BoundsConstraints(bounds, param_set.constraints),
                               minimizer_kwargs=dict(method='Nelder-Mead', args=tuple(args)))
        else:
            if verbose:
                print('Performing local optimization')

            fit_nm = minimize(fun, fit_grid.x, args=tuple(args), method='Nelder-Mead')
            if verbose:
                print(f'LL Nelder_mead: {fit_nm.fun:.3f}')
            use_grid_guess = False
            if fit_grid.fun < fit_nm.fun:
                use_grid_guess = True
                warnings.warn(f'Grid guess (LL={fit_grid.fun:.3f}) superior to minimize (LL={fit_nm.fun:.3f})')
            bds = [(fit_nm.x[i] >= b[0]) & (fit_nm.x[i] <= b[1]) for i, b in enumerate(param_set.bounds)]
            cns = [con['fun'](fit_nm.x) >= 0 for con in param_set.constraints]
            if not (np.all(cns) and np.all(bds)):
                use_grid_guess = True
                warnings.warn(f'Nelder-Mead estimate {fit_nm.x} violates constraints {not np.all(cns)} and/or '
                              f'bounds {not np.all(bds)}')
            if use_grid_guess:
                fit = fit_grid
            else:
                fit = fit_nm
    else:
        if global_minimization:
            if verbose:
                print('Performing global optimization')
            fit = basinhopping(fun, x0, take_step=RandomDisplacementBoundsConstraints(bounds, param_set.constraints),
                               accept_test=BoundsConstraints(bounds, param_set.constraints),
                               minimizer_kwargs=dict(method='Nelder-Mead', args=tuple(args)))
        else:
            if verbose:
                print('Performing local optimization')

            if init_nelder_mead:
                fit_nelder = minimize(fun, x0, bounds=bounds, args=tuple(args), constraints=param_set.constraints,
                                      method='nelder-mead')
                for i, b in enumerate(param_set.bounds):
                    fit_nelder.x[i] = min(max(fit_nelder.x[i], b[0]), b[1])
                if fun(fit_nelder.x, *args) < fun(x0, *args):
                    x0 = fit_nelder.x
            gradient_method = [gradient_method] if isinstance(gradient_method, str) else gradient_method
            min_fun = fun(x0, *args)
            fit = fit_grid
            for method in gradient_method:
                if method == 'slsqp':
                    slsqp_epsilon_ = slsqp_epsilon if hasattr(slsqp_epsilon, '__len__') else [slsqp_epsilon]
                    for eps in slsqp_epsilon_:
                        fit_ = minimize(fun, x0, bounds=bounds, args=tuple(args), constraints=param_set.constraints,
                                        method='slsqp', options=dict(eps=eps))
                        if fit_.fun < min_fun:
                            min_fun = fit_.fun
                            fit = fit_
                            # x0 = fit_.x  # risk of local minima?
                else:
                    fit_ = minimize(fun, x0, bounds=bounds, args=tuple(args), constraints=param_set.constraints,
                                    method=method)
                    if fit_.fun < min_fun:
                        min_fun = fit_.fun
                        fit = fit_
                        # x0 = fit_.x  # risk of local minima?
    fit.execution_time = timeit.default_timer() - t0  # noqa

    return fit


class RandomDisplacementBoundsConstraints(object):
    def __init__(self, bounds, constraints, stepsize=0.5):
        self.xmin = bounds.lb
        self.xmax = bounds.ub
        self.constraints = constraints
        self.stepsize = stepsize

    def __call__(self, x):
        while True:
            xnew = x + np.random.uniform(-self.stepsize, self.stepsize, np.shape(x))
            if np.all(xnew < self.xmax) and np.all(xnew > self.xmin) and np.all(
                    [con['fun'](xnew) >= 0 for con in self.constraints]):
                break
        return xnew


class BoundsConstraints(object):

    def __init__(self, bounds, constraints):
        self.xmin = bounds.lb
        self.xmax = bounds.ub
        self.constraints = constraints

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        cons = bool(np.all([con['fun'](x) >= 0 for con in self.constraints]))
        return tmax and tmin and cons
