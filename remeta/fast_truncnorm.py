
# This is the truncnorm implementation of scipy 1.3.3, which has much faster
# broadcasting than the implementation of scipy 1.4.0 and even more so than
# scipy 1.5.2. Use, until a newer version of scipy is more or less back to
# the speed of version 1.3.3.

# ToDo: strip this module of unnessacary code (there should be a lot of it)

#
# Author:  Travis Oliphant  2002-2011 with contributions from
#          SciPy Developers 2004-2011
#
from __future__ import division, print_function, absolute_import

import inspect
import keyword
import re
import types
from collections import namedtuple

import numpy as np
import scipy.special as sc
from numpy import (arange, putmask, ravel, ones, shape, ndarray, zeros, floor,
                   logical_and, log, sqrt, place, argmax, vectorize, asarray,
                   nan, inf, empty)
from scipy import integrate
from scipy import optimize
from scipy._lib import doccer
from scipy._lib._util import check_random_state
from scipy.misc import derivative
from scipy.special import (entr)
from scipy.stats._constants import _XMAX


def instancemethod(func, obj, cls):
    return types.MethodType(func, obj)

parse_arg_template = """
def _parse_args(self, %(shape_arg_str)s %(locscale_in)s):
    return (%(shape_arg_str)s), %(locscale_out)s

def _parse_args_rvs(self, %(shape_arg_str)s %(locscale_in)s, size=None):
    return self._argcheck_rvs(%(shape_arg_str)s %(locscale_out)s, size=size)

def _parse_args_stats(self, %(shape_arg_str)s %(locscale_in)s, moments='mv'):
    return (%(shape_arg_str)s), %(locscale_out)s, moments
"""

ArgSpec = namedtuple('ArgSpec', ['args', 'varargs', 'keywords', 'defaults'])

def _getargspec(func):
    """inspect.getargspec replacement using inspect.signature.

    inspect.getargspec is deprecated in python 3. This is a replacement
    based on the (new in python 3.3) `inspect.signature`.

    Parameters
    ----------
    func : callable
        A callable to inspect

    Returns
    -------
    argspec : ArgSpec(args, varargs, varkw, defaults)
        This is similar to the result of inspect.getargspec(func) under
        python 2.x.
        NOTE: if the first argument of `func` is self, it is *not*, I repeat
        *not* included in argspec.args.
        This is done for consistency between inspect.getargspec() under
        python 2.x, and inspect.signature() under python 3.x.
    """
    sig = inspect.signature(func)
    args = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    varargs = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.VAR_POSITIONAL
    ]
    varargs = varargs[0] if varargs else None
    varkw = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.VAR_KEYWORD
    ]
    varkw = varkw[0] if varkw else None
    defaults = [
        p.default for p in sig.parameters.values()
        if (p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and
           p.default is not p.empty)
    ] or None
    return ArgSpec(args, varargs, varkw, defaults)


# This should be rewritten
def argsreduce(cond, *args):
    """Return the sequence of ravel(args[i]) where ravel(condition) is
    True in 1D.

    Examples
    --------
    >>> import numpy as np
    >>> rand = np.random.random_sample
    >>> A = rand((4, 5))
    >>> B = 2
    >>> C = rand((1, 5))
    >>> cond = np.ones(A.shape)
    >>> [A1, B1, C1] = argsreduce(cond, A, B, C)
    >>> B1.shape
    (20,)
    >>> cond[2,:] = 0
    >>> [A2, B2, C2] = argsreduce(cond, A, B, C)
    >>> B2.shape
    (15,)

    """
    newargs = np.atleast_1d(*args)
    if not isinstance(newargs, (list, tuple)):
        newargs = [newargs, ]
    expand_arr = (cond == cond)
    return [np.extract(cond, arr1 * expand_arr) for arr1 in newargs]


# Frozen RV class
class rv_frozen(object):

    def __init__(self, dist, *args, **kwds):
        self.args = args
        self.kwds = kwds

        # create a new instance
        self.dist = dist.__class__(**dist._updated_ctor_param())

        # a, b may be set in _argcheck, depending on *args, **kwds. Ouch.
        shapes, _, _ = self.dist._parse_args(*args, **kwds)
        self.dist._argcheck(*shapes)
        self.a, self.b = self.dist.a, self.dist.b

    @property
    def random_state(self):
        return self.dist._random_state

    @random_state.setter
    def random_state(self, seed):
        self.dist._random_state = check_random_state(seed)

    def pdf(self, x):   # raises AttributeError in frozen discrete distribution
        return self.dist.pdf(x, *self.args, **self.kwds)

    def logpdf(self, x):
        return self.dist.logpdf(x, *self.args, **self.kwds)

    def cdf(self, x):
        return self.dist.cdf(x, *self.args, **self.kwds)

    def logcdf(self, x):
        return self.dist.logcdf(x, *self.args, **self.kwds)

    def ppf(self, q):
        return self.dist.ppf(q, *self.args, **self.kwds)

    def isf(self, q):
        return self.dist.isf(q, *self.args, **self.kwds)

    def rvs(self, size=None, random_state=None):
        kwds = self.kwds.copy()
        kwds.update({'size': size, 'random_state': random_state})
        return self.dist.rvs(*self.args, **kwds)

    def sf(self, x):
        return self.dist.sf(x, *self.args, **self.kwds)

    def logsf(self, x):
        return self.dist.logsf(x, *self.args, **self.kwds)

    def stats(self, moments='mv'):
        kwds = self.kwds.copy()
        kwds.update({'moments': moments})
        return self.dist.stats(*self.args, **kwds)

    def median(self):
        return self.dist.median(*self.args, **self.kwds)

    def mean(self):
        return self.dist.mean(*self.args, **self.kwds)

    def var(self):
        return self.dist.var(*self.args, **self.kwds)

    def std(self):
        return self.dist.std(*self.args, **self.kwds)

    def moment(self, n):
        return self.dist.moment(n, *self.args, **self.kwds)

    def entropy(self):
        return self.dist.entropy(*self.args, **self.kwds)

    def pmf(self, k):
        return self.dist.pmf(k, *self.args, **self.kwds)

    def logpmf(self, k):
        return self.dist.logpmf(k, *self.args, **self.kwds)

    def interval(self, alpha):
        return self.dist.interval(alpha, *self.args, **self.kwds)

    def expect(self, func=None, lb=None, ub=None, conditional=False, **kwds):
        # expect method only accepts shape parameters as positional args
        # hence convert self.args, self.kwds, also loc/scale
        # See the .expect method docstrings for the meaning of
        # other parameters.
        a, loc, scale = self.dist._parse_args(*self.args, **self.kwds)
        if isinstance(self.dist, rv_discrete):
            return self.dist.expect(func, a, loc, lb, ub, conditional, **kwds)
        else:
            return self.dist.expect(func, a, loc, scale, lb, ub,
                                    conditional, **kwds)

    def support(self):
        return self.dist.support(*self.args, **self.kwds)


class rv_generic(object):
    """Class which encapsulates common functionality between rv_discrete
    and rv_continuous.

    """
    def __init__(self, seed=None):
        super(rv_generic, self).__init__()

        # figure out if _stats signature has 'moments' keyword
        sign = _getargspec(self._stats)
        self._stats_has_moments = ((sign[2] is not None) or
                                   ('moments' in sign[0]))
        self._random_state = check_random_state(seed)

    @property
    def random_state(self):
        """ Get or set the RandomState object for generating random variates.

        This can be either None or an existing RandomState object.

        If None (or np.random), use the RandomState singleton used by np.random.
        If already a RandomState instance, use it.
        If an int, use a new RandomState instance seeded with seed.

        """
        return self._random_state

    @random_state.setter
    def random_state(self, seed):
        self._random_state = check_random_state(seed)

    def __getstate__(self):
        return self._updated_ctor_param(), self._random_state

    def __setstate__(self, state):
        ctor_param, r = state
        self.__init__(**ctor_param)
        self._random_state = r
        return self

    def _construct_argparser(
            self, meths_to_inspect, locscale_in, locscale_out):
        """Construct the parser for the shape arguments.

        Generates the argument-parsing functions dynamically and attaches
        them to the instance.
        Is supposed to be called in __init__ of a class for each distribution.

        If self.shapes is a non-empty string, interprets it as a
        comma-separated list of shape parameters.

        Otherwise inspects the call signatures of `meths_to_inspect`
        and constructs the argument-parsing functions from these.
        In this case also sets `shapes` and `numargs`.
        """

        if self.shapes:
            # sanitize the user-supplied shapes
            if not isinstance(self.shapes, str):
                raise TypeError('shapes must be a string.')

            shapes = self.shapes.replace(',', ' ').split()

            for field in shapes:
                if keyword.iskeyword(field):
                    raise SyntaxError('keywords cannot be used as shapes.')
                if not re.match('^[_a-zA-Z][_a-zA-Z0-9]*$', field):
                    raise SyntaxError(
                        'shapes must be valid python identifiers')
        else:
            # find out the call signatures (_pdf, _cdf etc), deduce shape
            # arguments. Generic methods only have 'self, x', any further args
            # are shapes.
            shapes_list = []
            for meth in meths_to_inspect:
                shapes_args = _getargspec(meth)   # NB: does not contain self
                args = shapes_args.args[1:]       # peel off 'x', too

                if args:
                    shapes_list.append(args)

                    # *args or **kwargs are not allowed w/automatic shapes
                    if shapes_args.varargs is not None:
                        raise TypeError(
                            '*args are not allowed w/out explicit shapes')
                    if shapes_args.keywords is not None:
                        raise TypeError(
                            '**kwds are not allowed w/out explicit shapes')
                    if shapes_args.defaults is not None:
                        raise TypeError('defaults are not allowed for shapes')

            if shapes_list:
                shapes = shapes_list[0]

                # make sure the signatures are consistent
                for item in shapes_list:
                    if item != shapes:
                        raise TypeError('Shape arguments are inconsistent.')
            else:
                shapes = []

        # have the arguments, construct the method from template
        shapes_str = ', '.join(shapes) + ', ' if shapes else ''  # NB: not None
        dct = dict(shape_arg_str=shapes_str,
                   locscale_in=locscale_in,
                   locscale_out=locscale_out,
                   )
        ns = {}
        exec(parse_arg_template % dct, ns)
        # NB: attach to the instance, not class
        for name in ['_parse_args', '_parse_args_stats', '_parse_args_rvs']:
            setattr(self, name,
                    instancemethod(ns[name], self, self.__class__)
                    )

        self.shapes = ', '.join(shapes) if shapes else None
        if not hasattr(self, 'numargs'):
            # allows more general subclassing with *args
            self.numargs = len(shapes)

    def _construct_doc(self, docdict, shapes_vals=None):
        """Construct the instance docstring with string substitutions."""
        tempdict = docdict.copy()
        tempdict['name'] = self.name or 'distname'
        tempdict['shapes'] = self.shapes or ''

        if shapes_vals is None:
            shapes_vals = ()
        vals = ', '.join('%.3g' % val for val in shapes_vals)
        tempdict['vals'] = vals

        tempdict['shapes_'] = self.shapes or ''
        if self.shapes and self.numargs == 1:
            tempdict['shapes_'] += ','

        if self.shapes:
            tempdict['set_vals_stmt'] = '>>> %s = %s' % (self.shapes, vals)
        else:
            tempdict['set_vals_stmt'] = ''

        if self.shapes is None:
            # remove shapes from call parameters if there are none
            for item in ['default', 'before_notes']:
                tempdict[item] = tempdict[item].replace(
                    "\n%(shapes)s : array_like\n    shape parameters", "")
        for i in range(2):
            if self.shapes is None:
                # necessary because we use %(shapes)s in two forms (w w/o ", ")
                self.__doc__ = self.__doc__.replace("%(shapes)s, ", "")
            try:
                self.__doc__ = doccer.docformat(self.__doc__, tempdict)
            except TypeError as e:
                raise Exception("Unable to construct docstring for distribution \"%s\": %s" % (self.name, repr(e)))

        # correct for empty shapes
        self.__doc__ = self.__doc__.replace('(, ', '(').replace(', )', ')')

    def freeze(self, *args, **kwds):
        """Freeze the distribution for the given arguments.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution.  Should include all
            the non-optional arguments, may include ``loc`` and ``scale``.

        Returns
        -------
        rv_frozen : rv_frozen instance
            The frozen distribution.

        """
        return rv_frozen(self, *args, **kwds)

    def __call__(self, *args, **kwds):
        return self.freeze(*args, **kwds)
    __call__.__doc__ = freeze.__doc__

    # The actual calculation functions (no basic checking need be done)
    # If these are defined, the others won't be looked at.
    # Otherwise, the other set can be defined.
    def _stats(self, *args, **kwds):
        return None, None, None, None

    #  Central moments
    def _munp(self, n, *args):
        # Silence floating point warnings from integration.
        olderr = np.seterr(all='ignore')
        vals = self.generic_moment(n, *args)
        np.seterr(**olderr)
        return vals

    def _argcheck_rvs(self, *args, **kwargs):
        # Handle broadcasting and size validation of the rvs method.
        # Subclasses should not have to override this method.
        # The rule is that if `size` is not None, then `size` gives the
        # shape of the result (integer values of `size` are treated as
        # tuples with length 1; i.e. `size=3` is the same as `size=(3,)`.)
        #
        # `args` is expected to contain the shape parameters (if any), the
        # location and the scale in a flat tuple (e.g. if there are two
        # shape parameters `a` and `b`, `args` will be `(a, b, loc, scale)`).
        # The only keyword argument expected is 'size'.
        size = kwargs.get('size', None)
        all_bcast = np.broadcast_arrays(*args)

        def squeeze_left(a):
            while a.ndim > 0 and a.shape[0] == 1:
                a = a[0]
            return a

        # Eliminate trivial leading dimensions.  In the convention
        # used by numpy's random variate generators, trivial leading
        # dimensions are effectively ignored.  In other words, when `size`
        # is given, trivial leading dimensions of the broadcast parameters
        # in excess of the number of dimensions  in size are ignored, e.g.
        #   >>> np.random.normal([[1, 3, 5]], [[[[0.01]]]], size=3)
        #   array([ 1.00104267,  3.00422496,  4.99799278])
        # If `size` is not given, the exact broadcast shape is preserved:
        #   >>> np.random.normal([[1, 3, 5]], [[[[0.01]]]])
        #   array([[[[ 1.00862899,  3.00061431,  4.99867122]]]])
        #
        all_bcast = [squeeze_left(a) for a in all_bcast]
        bcast_shape = all_bcast[0].shape
        bcast_ndim = all_bcast[0].ndim

        if size is None:
            size_ = bcast_shape
        else:
            size_ = tuple(np.atleast_1d(size))

        # Check compatibility of size_ with the broadcast shape of all
        # the parameters.  This check is intended to be consistent with
        # how the numpy random variate generators (e.g. np.random.normal,
        # np.random.beta) handle their arguments.   The rule is that, if size
        # is given, it determines the shape of the output.  Broadcasting
        # can't change the output size.

        # This is the standard broadcasting convention of extending the
        # shape with fewer dimensions with enough dimensions of length 1
        # so that the two shapes have the same number of dimensions.
        ndiff = bcast_ndim - len(size_)
        if ndiff < 0:
            bcast_shape = (1,)*(-ndiff) + bcast_shape
        elif ndiff > 0:
            size_ = (1,)*ndiff + size_

        # This compatibility test is not standard.  In "regular" broadcasting,
        # two shapes are compatible if for each dimension, the lengths are the
        # same or one of the lengths is 1.  Here, the length of a dimension in
        # size_ must not be less than the corresponding length in bcast_shape.
        ok = all([bcdim == 1 or bcdim == szdim
                  for (bcdim, szdim) in zip(bcast_shape, size_)])
        if not ok:
            raise ValueError("size does not match the broadcast shape of "
                             "the parameters.")

        param_bcast = all_bcast[:-2]
        loc_bcast = all_bcast[-2]
        scale_bcast = all_bcast[-1]

        return param_bcast, loc_bcast, scale_bcast, size_

    ## These are the methods you must define (standard form functions)
    ## NB: generic _pdf, _logpdf, _cdf are different for
    ## rv_continuous and rv_discrete hence are defined in there
    def _argcheck(self, *args):
        """Default check for correct values on args and keywords.

        Returns condition array of 1's where arguments are correct and
         0's where they are not.

        """
        cond = 1
        for arg in args:
            cond = logical_and(cond, (asarray(arg) > 0))
        return cond

    def _get_support(self, *args):
        """Return the support of the (unscaled, unshifted) distribution.

        *Must* be overridden by distributions which have support dependent
        upon the shape parameters of the distribution.  Any such override
        *must not* set or change any of the class members, as these members
        are shared amongst all instances of the distribution.

        Parameters
        ----------
        arg1, arg2, ... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        Returns
        -------
        a, b : numeric (float, or int or +/-np.inf)
            end-points of the distribution's support for the specified
            shape parameters.
        """
        return self.a, self.b

    def _support_mask(self, x, *args):
        a, b = self._get_support(*args)
        return (a <= x) & (x <= b)

    def _open_support_mask(self, x, *args):
        a, b = self._get_support(*args)
        return (a < x) & (x < b)

    def _rvs(self, *args):
        # This method must handle self._size being a tuple, and it must
        # properly broadcast *args and self._size.  self._size might be
        # an empty tuple, which means a scalar random variate is to be
        # generated.

        ## Use basic inverse cdf algorithm for RV generation as default.
        U = self._random_state.random_sample(self._size)
        Y = self._ppf(U, *args)
        return Y

    def _logcdf(self, x, *args):
        return log(self._cdf(x, *args))

    def _sf(self, x, *args):
        return 1.0-self._cdf(x, *args)

    def _logsf(self, x, *args):
        return log(self._sf(x, *args))

    def _ppf(self, q, *args):
        return self._ppfvec(q, *args)

    def _isf(self, q, *args):
        return self._ppf(1.0-q, *args)  # use correct _ppf for subclasses

    # These are actually called, and should not be overwritten if you
    # want to keep error checking.
    def rvs(self, *args, **kwds):
        """
        Random variates of given type.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).
        scale : array_like, optional
            Scale parameter (default=1).
        size : int or tuple of ints, optional
            Defining number of random variates (default is 1).
        random_state : None or int or ``np.random.RandomState`` instance, optional
            If int or RandomState, use it for drawing the random variates.
            If None, rely on ``self.random_state``.
            Default is None.

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of given `size`.

        """
        discrete = kwds.pop('discrete', None)
        rndm = kwds.pop('random_state', None)
        args, loc, scale, size = self._parse_args_rvs(*args, **kwds)
        cond = logical_and(self._argcheck(*args), (scale >= 0))
        if not np.all(cond):
            raise ValueError("Domain error in arguments.")

        if np.all(scale == 0):
            return loc*ones(size, 'd')

        # extra gymnastics needed for a custom random_state
        if rndm is not None:
            random_state_saved = self._random_state
            self._random_state = check_random_state(rndm)

        # `size` should just be an argument to _rvs(), but for, um,
        # historical reasons, it is made an attribute that is read
        # by _rvs().
        self._size = size
        vals = self._rvs(*args)

        vals = vals * scale + loc

        # do not forget to restore the _random_state
        if rndm is not None:
            self._random_state = random_state_saved

        # Cast to int if discrete
        if discrete:
            if size == ():
                vals = int(vals)
            else:
                vals = vals.astype(int)

        return vals

    def stats(self, *args, **kwds):
        """
        Some statistics of the given RV.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional (continuous RVs only)
            scale parameter (default=1)
        moments : str, optional
            composed of letters ['mvsk'] defining which moments to compute:
            'm' = mean,
            'v' = variance,
            's' = (Fisher's) skew,
            'k' = (Fisher's) kurtosis.
            (default is 'mv')

        Returns
        -------
        stats : sequence
            of requested moments.

        """
        args, loc, scale, moments = self._parse_args_stats(*args, **kwds)
        # scale = 1 by construction for discrete RVs
        loc, scale = map(asarray, (loc, scale))
        args = tuple(map(asarray, args))
        cond = self._argcheck(*args) & (scale > 0) & (loc == loc)
        output = []
        default = np.full(shape(cond), self.badvalue)

        # Use only entries that are valid in calculation
        if np.any(cond):
            goodargs = argsreduce(cond, *(args+(scale, loc)))
            scale, loc, goodargs = goodargs[-2], goodargs[-1], goodargs[:-2]

            if self._stats_has_moments:
                mu, mu2, g1, g2 = self._stats(*goodargs,
                                              **{'moments': moments})
            else:
                mu, mu2, g1, g2 = self._stats(*goodargs)
            if g1 is None:
                mu3 = None
            else:
                if mu2 is None:
                    mu2 = self._munp(2, *goodargs)
                if g2 is None:
                    # (mu2**1.5) breaks down for nan and inf
                    mu3 = g1 * np.power(mu2, 1.5)

            if 'm' in moments:
                if mu is None:
                    mu = self._munp(1, *goodargs)
                out0 = default.copy()
                place(out0, cond, mu * scale + loc)
                output.append(out0)

            if 'v' in moments:
                if mu2 is None:
                    mu2p = self._munp(2, *goodargs)
                    if mu is None:
                        mu = self._munp(1, *goodargs)
                    mu2 = mu2p - mu * mu
                    if np.isinf(mu):
                        # if mean is inf then var is also inf
                        mu2 = np.inf
                out0 = default.copy()
                place(out0, cond, mu2 * scale * scale)
                output.append(out0)

            if 's' in moments:
                if g1 is None:
                    mu3p = self._munp(3, *goodargs)
                    if mu is None:
                        mu = self._munp(1, *goodargs)
                    if mu2 is None:
                        mu2p = self._munp(2, *goodargs)
                        mu2 = mu2p - mu * mu
                    with np.errstate(invalid='ignore'):
                        mu3 = (-mu*mu - 3*mu2)*mu + mu3p
                        g1 = mu3 / np.power(mu2, 1.5)
                out0 = default.copy()
                place(out0, cond, g1)
                output.append(out0)

            if 'k' in moments:
                if g2 is None:
                    mu4p = self._munp(4, *goodargs)
                    if mu is None:
                        mu = self._munp(1, *goodargs)
                    if mu2 is None:
                        mu2p = self._munp(2, *goodargs)
                        mu2 = mu2p - mu * mu
                    if mu3 is None:
                        mu3p = self._munp(3, *goodargs)
                        with np.errstate(invalid='ignore'):
                            mu3 = (-mu * mu - 3 * mu2) * mu + mu3p
                            mu3 = mu3p - 3 * mu * mu2 - mu**3
                    with np.errstate(invalid='ignore'):
                        mu4 = ((-mu**2 - 6*mu2) * mu - 4*mu3)*mu + mu4p
                        g2 = mu4 / mu2**2.0 - 3.0
                out0 = default.copy()
                place(out0, cond, g2)
                output.append(out0)
        else:  # no valid args
            output = [default.copy() for _ in moments]

        if len(output) == 1:
            return output[0]
        else:
            return tuple(output)


    def median(self, *args, **kwds):
        """
        Median of the distribution.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            Location parameter, Default is 0.
        scale : array_like, optional
            Scale parameter, Default is 1.

        Returns
        -------
        median : float
            The median of the distribution.

        See Also
        --------
        rv_discrete.ppf
            Inverse of the CDF

        """
        return self.ppf(0.5, *args, **kwds)

    def mean(self, *args, **kwds):
        """
        Mean of the distribution.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        mean : float
            the mean of the distribution

        """
        kwds['moments'] = 'm'
        res = self.stats(*args, **kwds)
        if isinstance(res, ndarray) and res.ndim == 0:
            return res[()]
        return res

    def var(self, *args, **kwds):
        """
        Variance of the distribution.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        var : float
            the variance of the distribution

        """
        kwds['moments'] = 'v'
        res = self.stats(*args, **kwds)
        if isinstance(res, ndarray) and res.ndim == 0:
            return res[()]
        return res

    def std(self, *args, **kwds):
        """
        Standard deviation of the distribution.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        std : float
            standard deviation of the distribution

        """
        kwds['moments'] = 'v'
        res = sqrt(self.stats(*args, **kwds))
        return res

    def interval(self, alpha, *args, **kwds):
        """
        Confidence interval with equal areas around the median.

        Parameters
        ----------
        alpha : array_like of float
            Probability that an rv will be drawn from the returned range.
            Each value should be in the range [0, 1].
        arg1, arg2, ... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            location parameter, Default is 0.
        scale : array_like, optional
            scale parameter, Default is 1.

        Returns
        -------
        a, b : ndarray of float
            end-points of range that contain ``100 * alpha %`` of the rv's
            possible values.

        """
        alpha = asarray(alpha)
        if np.any((alpha > 1) | (alpha < 0)):
            raise ValueError("alpha must be between 0 and 1 inclusive")
        q1 = (1.0-alpha)/2
        q2 = (1.0+alpha)/2
        a = self.ppf(q1, *args, **kwds)
        b = self.ppf(q2, *args, **kwds)
        return a, b

    def support(self, *args, **kwargs):
        """
        Return the support of the distribution.

        Parameters
        ----------
        arg1, arg2, ... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            location parameter, Default is 0.
        scale : array_like, optional
            scale parameter, Default is 1.
        Returns
        -------
        a, b : float
            end-points of the distribution's support.

        """
        args, loc, scale = self._parse_args(*args, **kwargs)
        _a, _b = self._get_support(*args)
        return _a * scale + loc, _b * scale + loc


class rv_continuous(rv_generic):
    """
    A generic continuous random variable class meant for subclassing.

    `rv_continuous` is a base class to construct specific distribution classes
    and instances for continuous random variables. It cannot be used
    directly as a distribution.

    Parameters
    ----------
    momtype : int, optional
        The type of generic moment calculation to use: 0 for pdf, 1 (default)
        for ppf.
    a : float, optional
        Lower bound of the support of the distribution, default is minus
        infinity.
    b : float, optional
        Upper bound of the support of the distribution, default is plus
        infinity.
    xtol : float, optional
        The tolerance for fixed point calculation for generic ppf.
    badvalue : float, optional
        The value in a result arrays that indicates a value that for which
        some argument restriction is violated, default is np.nan.
    name : str, optional
        The name of the instance. This string is used to construct the default
        example for distributions.
    longname : str, optional
        This string is used as part of the first line of the docstring returned
        when a subclass has no docstring of its own. Note: `longname` exists
        for backwards compatibility, do not use for new subclasses.
    shapes : str, optional
        The shape of the distribution. For example ``"m, n"`` for a
        distribution that takes two integers as the two shape arguments for all
        its methods. If not provided, shape parameters will be inferred from
        the signature of the private methods, ``_pdf`` and ``_cdf`` of the
        instance.
    extradoc :  str, optional, deprecated
        This string is used as the last part of the docstring returned when a
        subclass has no docstring of its own. Note: `extradoc` exists for
        backwards compatibility, do not use for new subclasses.
    seed : None or int or ``numpy.random.RandomState`` instance, optional
        This parameter defines the RandomState object to use for drawing
        random variates.
        If None (or np.random), the global np.random state is used.
        If integer, it is used to seed the local RandomState instance.
        Default is None.

    Methods
    -------
    rvs
    pdf
    logpdf
    cdf
    logcdf
    sf
    logsf
    ppf
    isf
    moment
    stats
    entropy
    expect
    median
    mean
    std
    var
    interval
    __call__
    fit
    fit_loc_scale
    nnlf
    support

    Notes
    -----
    Public methods of an instance of a distribution class (e.g., ``pdf``,
    ``cdf``) check their arguments and pass valid arguments to private,
    computational methods (``_pdf``, ``_cdf``). For ``pdf(x)``, ``x`` is valid
    if it is within the support of the distribution.
    Whether a shape parameter is valid is decided by an ``_argcheck`` method
    (which defaults to checking that its arguments are strictly positive.)

    **Subclassing**

    New random variables can be defined by subclassing the `rv_continuous` class
    and re-defining at least the ``_pdf`` or the ``_cdf`` method (normalized
    to location 0 and scale 1).

    If positive argument checking is not correct for your RV
    then you will also need to re-define the ``_argcheck`` method.

    For most of the scipy.stats distributions, the support interval doesn't
    depend on the shape parameters. ``x`` being in the support interval is
    equivalent to ``self.a <= x <= self.b``.  If either of the endpoints of
    the support do depend on the shape parameters, then
    i) the distribution must implement the ``_get_support`` method; and
    ii) those dependent endpoints must be omitted from the distribution's
    call to the ``rv_continuous`` initializer.

    Correct, but potentially slow defaults exist for the remaining
    methods but for speed and/or accuracy you can over-ride::

      _logpdf, _cdf, _logcdf, _ppf, _rvs, _isf, _sf, _logsf

    The default method ``_rvs`` relies on the inverse of the cdf, ``_ppf``,
    applied to a uniform random variate. In order to generate random variates
    efficiently, either the default ``_ppf`` needs to be overwritten (e.g.
    if the inverse cdf can expressed in an explicit form) or a sampling
    method needs to be implemented in a custom ``_rvs`` method.

    If possible, you should override ``_isf``, ``_sf`` or ``_logsf``.
    The main reason would be to improve numerical accuracy: for example,
    the survival function ``_sf`` is computed as ``1 - _cdf`` which can
    result in loss of precision if ``_cdf(x)`` is close to one.

    **Methods that can be overwritten by subclasses**
    ::

      _rvs
      _pdf
      _cdf
      _sf
      _ppf
      _isf
      _stats
      _munp
      _entropy
      _argcheck
      _get_support

    There are additional (internal and private) generic methods that can
    be useful for cross-checking and for debugging, but might work in all
    cases when directly called.

    A note on ``shapes``: subclasses need not specify them explicitly. In this
    case, `shapes` will be automatically deduced from the signatures of the
    overridden methods (`pdf`, `cdf` etc).
    If, for some reason, you prefer to avoid relying on introspection, you can
    specify ``shapes`` explicitly as an argument to the instance constructor.


    **Frozen Distributions**

    Normally, you must provide shape parameters (and, optionally, location and
    scale parameters to each call of a method of a distribution.

    Alternatively, the object may be called (as a function) to fix the shape,
    location, and scale parameters returning a "frozen" continuous RV object:

    rv = generic(<shape(s)>, loc=0, scale=1)
        `rv_frozen` object with the same methods but holding the given shape,
        location, and scale fixed

    **Statistics**

    Statistics are computed using numerical integration by default.
    For speed you can redefine this using ``_stats``:

     - take shape parameters and return mu, mu2, g1, g2
     - If you can't compute one of these, return it as None
     - Can also be defined with a keyword argument ``moments``, which is a
       string composed of "m", "v", "s", and/or "k".
       Only the components appearing in string should be computed and
       returned in the order "m", "v", "s", or "k"  with missing values
       returned as None.

    Alternatively, you can override ``_munp``, which takes ``n`` and shape
    parameters and returns the n-th non-central moment of the distribution.

    Examples
    --------
    To create a new Gaussian distribution, we would do the following:

    >>> from scipy.stats import rv_continuous
    >>> class gaussian_gen(rv_continuous):
    ...     "Gaussian distribution"
    ...     def _pdf(self, x):
    ...         return np.exp(-x**2 / 2.) / np.sqrt(2.0 * np.pi)
    >>> gaussian = gaussian_gen(name='gaussian')

    ``scipy.stats`` distributions are *instances*, so here we subclass
    `rv_continuous` and create an instance. With this, we now have
    a fully functional distribution with all relevant methods automagically
    generated by the framework.

    Note that above we defined a standard normal distribution, with zero mean
    and unit variance. Shifting and scaling of the distribution can be done
    by using ``loc`` and ``scale`` parameters: ``gaussian.pdf(x, loc, scale)``
    essentially computes ``y = (x - loc) / scale`` and
    ``gaussian._pdf(y) / scale``.

    """
    def __init__(self, momtype=1, a=None, b=None, xtol=1e-14,
                 badvalue=None, name=None, longname=None,
                 shapes=None, extradoc=None, seed=None):

        super(rv_continuous, self).__init__(seed)

        # save the ctor parameters, cf generic freeze
        self._ctor_param = dict(
            momtype=momtype, a=a, b=b, xtol=xtol,
            badvalue=badvalue, name=name, longname=longname,
            shapes=shapes, extradoc=extradoc, seed=seed)

        if badvalue is None:
            badvalue = nan
        if name is None:
            name = 'Distribution'
        self.badvalue = badvalue
        self.name = name
        self.a = a
        self.b = b
        if a is None:
            self.a = -inf
        if b is None:
            self.b = inf
        self.xtol = xtol
        self.moment_type = momtype
        self.shapes = shapes
        self._construct_argparser(meths_to_inspect=[self._pdf, self._cdf],
                                  locscale_in='loc=0, scale=1',
                                  locscale_out='loc, scale')

        # nin correction
        self._ppfvec = vectorize(self._ppf_single, otypes='d')
        self._ppfvec.nin = self.numargs + 1
        self.vecentropy = vectorize(self._entropy, otypes='d')
        self._cdfvec = vectorize(self._cdf_single, otypes='d')
        self._cdfvec.nin = self.numargs + 1

        self.extradoc = extradoc
        if momtype == 0:
            self.generic_moment = vectorize(self._mom0_sc, otypes='d')
        else:
            self.generic_moment = vectorize(self._mom1_sc, otypes='d')
        # Because of the *args argument of _mom0_sc, vectorize cannot count the
        # number of arguments correctly.
        self.generic_moment.nin = self.numargs + 1

        if longname is None:
            if name[0] in ['aeiouAEIOU']:
                hstr = "An "
            else:
                hstr = "A "
            longname = hstr + name

    def _updated_ctor_param(self):
        """ Return the current version of _ctor_param, possibly updated by user.

            Used by freezing and pickling.
            Keep this in sync with the signature of __init__.
        """
        dct = self._ctor_param.copy()
        dct['a'] = self.a
        dct['b'] = self.b
        dct['xtol'] = self.xtol
        dct['badvalue'] = self.badvalue
        dct['name'] = self.name
        dct['shapes'] = self.shapes
        dct['extradoc'] = self.extradoc
        return dct

    def _ppf_to_solve(self, x, q, *args):
        return self.cdf(*(x, )+args)-q

    def _ppf_single(self, q, *args):
        left = right = None
        _a, _b = self._get_support(*args)
        if _a > -np.inf:
            left = _a
        if _b < np.inf:
            right = _b

        factor = 10.
        if not left:  # i.e. self.a = -inf
            left = -1.*factor
            while self._ppf_to_solve(left, q, *args) > 0.:
                right = left
                left *= factor
            # left is now such that cdf(left) < q
        if not right:  # i.e. self.b = inf
            right = factor
            while self._ppf_to_solve(right, q, *args) < 0.:
                left = right
                right *= factor
            # right is now such that cdf(right) > q

        return optimize.brentq(self._ppf_to_solve,
                               left, right, args=(q,)+args, xtol=self.xtol)

    # moment from definition
    def _mom_integ0(self, x, m, *args):
        return x**m * self.pdf(x, *args)

    def _mom0_sc(self, m, *args):
        _a, _b = self._get_support(*args)
        return integrate.quad(self._mom_integ0, _a, _b,
                              args=(m,)+args)[0]

    # moment calculated using ppf
    def _mom_integ1(self, q, m, *args):
        return (self.ppf(q, *args))**m

    def _mom1_sc(self, m, *args):
        return integrate.quad(self._mom_integ1, 0, 1, args=(m,)+args)[0]

    def _pdf(self, x, *args):
        return derivative(self._cdf, x, dx=1e-5, args=args, order=5)

    ## Could also define any of these
    def _logpdf(self, x, *args):
        return log(self._pdf(x, *args))

    def _cdf_single(self, x, *args):
        _a, _b = self._get_support(*args)
        return integrate.quad(self._pdf, _a, x, args=args)[0]

    def _cdf(self, x, *args):
        return self._cdfvec(x, *args)

    ## generic _argcheck, _logcdf, _sf, _logsf, _ppf, _isf, _rvs are defined
    ## in rv_generic

    def pdf(self, x, *args, **kwds):
        """
        Probability density function at x of the given RV.

        Parameters
        ----------
        x : array_like
            quantiles
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at x

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        # dtyp = np.find_common_type([x.dtype, np.float64], [])
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc)/scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._support_mask(x, *args) & (scale > 0)
        cond = cond0 & cond1
        output = zeros(shape(cond), dtyp)
        putmask(output, (1-cond0)+np.isnan(x), self.badvalue)
        if np.any(cond):
            goodargs = argsreduce(cond, *((x,)+args+(scale,)))
            scale, goodargs = goodargs[-1], goodargs[:-1]
            place(output, cond, self._pdf(*goodargs) / scale)
        if output.ndim == 0:
            return output[()]
        return output

    def logpdf(self, x, *args, **kwds):
        """
        Log of the probability density function at x of the given RV.

        This uses a more numerically accurate calculation if available.

        Parameters
        ----------
        x : array_like
            quantiles
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        logpdf : array_like
            Log of the probability density function evaluated at x

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        # dtyp = np.find_common_type([x.dtype, np.float64], [])
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc)/scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._support_mask(x, *args) & (scale > 0)
        cond = cond0 & cond1
        output = empty(shape(cond), dtyp)
        output.fill(-np.inf)
        putmask(output, (1-cond0)+np.isnan(x), self.badvalue)
        if np.any(cond):
            goodargs = argsreduce(cond, *((x,)+args+(scale,)))
            scale, goodargs = goodargs[-1], goodargs[:-1]
            place(output, cond, self._logpdf(*goodargs) - log(scale))
        if output.ndim == 0:
            return output[()]
        return output

    def cdf(self, x, *args, **kwds):
        """
        Cumulative distribution function of the given RV.

        Parameters
        ----------
        x : array_like
            quantiles
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        cdf : ndarray
            Cumulative distribution function evaluated at `x`

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        _a, _b = self._get_support(*args)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        # dtyp = np.find_common_type([x.dtype, np.float64], [])
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc)/scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._open_support_mask(x, *args) & (scale > 0)
        cond2 = (x >= _b) & cond0
        cond = cond0 & cond1
        output = zeros(shape(cond), dtyp)
        place(output, (1-cond0)+np.isnan(x), self.badvalue)
        place(output, cond2, 1.0)
        if np.any(cond):  # call only if at least 1 entry
            goodargs = argsreduce(cond, *((x,)+args))
            place(output, cond, self._cdf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def logcdf(self, x, *args, **kwds):
        """
        Log of the cumulative distribution function at x of the given RV.

        Parameters
        ----------
        x : array_like
            quantiles
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        logcdf : array_like
            Log of the cumulative distribution function evaluated at x

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        _a, _b = self._get_support(*args)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        # dtyp = np.find_common_type([x.dtype, np.float64], [])
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc)/scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._open_support_mask(x, *args) & (scale > 0)
        cond2 = (x >= _b) & cond0
        cond = cond0 & cond1
        output = empty(shape(cond), dtyp)
        output.fill(-np.inf)
        place(output, (1-cond0)*(cond1 == cond1)+np.isnan(x), self.badvalue)
        place(output, cond2, 0.0)
        if np.any(cond):  # call only if at least 1 entry
            goodargs = argsreduce(cond, *((x,)+args))
            place(output, cond, self._logcdf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def sf(self, x, *args, **kwds):
        """
        Survival function (1 - `cdf`) at x of the given RV.

        Parameters
        ----------
        x : array_like
            quantiles
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        sf : array_like
            Survival function evaluated at x

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        _a, _b = self._get_support(*args)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        # dtyp = np.find_common_type([x.dtype, np.float64], [])
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc)/scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._open_support_mask(x, *args) & (scale > 0)
        cond2 = cond0 & (x <= _a)
        cond = cond0 & cond1
        output = zeros(shape(cond), dtyp)
        place(output, (1-cond0)+np.isnan(x), self.badvalue)
        place(output, cond2, 1.0)
        if np.any(cond):
            goodargs = argsreduce(cond, *((x,)+args))
            place(output, cond, self._sf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def logsf(self, x, *args, **kwds):
        """
        Log of the survival function of the given RV.

        Returns the log of the "survival function," defined as (1 - `cdf`),
        evaluated at `x`.

        Parameters
        ----------
        x : array_like
            quantiles
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        logsf : ndarray
            Log of the survival function evaluated at `x`.

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        _a, _b = self._get_support(*args)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        # dtyp = np.find_common_type([x.dtype, np.float64], [])
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc)/scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._open_support_mask(x, *args) & (scale > 0)
        cond2 = cond0 & (x <= _a)
        cond = cond0 & cond1
        output = empty(shape(cond), dtyp)
        output.fill(-np.inf)
        place(output, (1-cond0)+np.isnan(x), self.badvalue)
        place(output, cond2, 0.0)
        if np.any(cond):
            goodargs = argsreduce(cond, *((x,)+args))
            place(output, cond, self._logsf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def ppf(self, q, *args, **kwds):
        """
        Percent point function (inverse of `cdf`) at q of the given RV.

        Parameters
        ----------
        q : array_like
            lower tail probability
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        x : array_like
            quantile corresponding to the lower tail probability q.

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        _a, _b = self._get_support(*args)
        q, loc, scale = map(asarray, (q, loc, scale))
        args = tuple(map(asarray, args))
        cond0 = self._argcheck(*args) & (scale > 0) & (loc == loc)
        cond1 = (0 < q) & (q < 1)
        cond2 = cond0 & (q == 0)
        cond3 = cond0 & (q == 1)
        cond = cond0 & cond1
        output = np.full(shape(cond), self.badvalue)

        lower_bound = _a * scale + loc
        upper_bound = _b * scale + loc
        place(output, cond2, argsreduce(cond2, lower_bound)[0])
        place(output, cond3, argsreduce(cond3, upper_bound)[0])

        if np.any(cond):  # call only if at least 1 entry
            goodargs = argsreduce(cond, *((q,)+args+(scale, loc)))
            scale, loc, goodargs = goodargs[-2], goodargs[-1], goodargs[:-2]
            place(output, cond, self._ppf(*goodargs) * scale + loc)
        if output.ndim == 0:
            return output[()]
        return output

    def isf(self, q, *args, **kwds):
        """
        Inverse survival function (inverse of `sf`) at q of the given RV.

        Parameters
        ----------
        q : array_like
            upper tail probability
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        x : ndarray or scalar
            Quantile corresponding to the upper tail probability q.

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        _a, _b = self._get_support(*args)
        q, loc, scale = map(asarray, (q, loc, scale))
        args = tuple(map(asarray, args))
        cond0 = self._argcheck(*args) & (scale > 0) & (loc == loc)
        cond1 = (0 < q) & (q < 1)
        cond2 = cond0 & (q == 1)
        cond3 = cond0 & (q == 0)
        cond = cond0 & cond1
        output = np.full(shape(cond), self.badvalue)

        lower_bound = _a * scale + loc
        upper_bound = _b * scale + loc
        place(output, cond2, argsreduce(cond2, lower_bound)[0])
        place(output, cond3, argsreduce(cond3, upper_bound)[0])

        if np.any(cond):
            goodargs = argsreduce(cond, *((q,)+args+(scale, loc)))
            scale, loc, goodargs = goodargs[-2], goodargs[-1], goodargs[:-2]
            place(output, cond, self._isf(*goodargs) * scale + loc)
        if output.ndim == 0:
            return output[()]
        return output

    def _nnlf(self, x, *args):
        return -np.sum(self._logpdf(x, *args), axis=0)

    def _unpack_loc_scale(self, theta):
        try:
            loc = theta[-2]
            scale = theta[-1]
            args = tuple(theta[:-2])
        except IndexError:
            raise ValueError("Not enough input arguments.")
        return loc, scale, args

    def nnlf(self, theta, x):
        '''Return negative loglikelihood function.

        Notes
        -----
        This is ``-sum(log pdf(x, theta), axis=0)`` where `theta` are the
        parameters (including loc and scale).
        '''
        loc, scale, args = self._unpack_loc_scale(theta)
        if not self._argcheck(*args) or scale <= 0:
            return inf
        x = asarray((x-loc) / scale)
        n_log_scale = len(x) * log(scale)
        if np.any(~self._support_mask(x, *args)):
            return inf
        return self._nnlf(x, *args) + n_log_scale

    def _nnlf_and_penalty(self, x, args):
        cond0 = ~self._support_mask(x, *args)
        n_bad = np.count_nonzero(cond0, axis=0)
        if n_bad > 0:
            x = argsreduce(~cond0, x)[0]
        logpdf = self._logpdf(x, *args)
        finite_logpdf = np.isfinite(logpdf)
        n_bad += np.sum(~finite_logpdf, axis=0)
        if n_bad > 0:
            penalty = n_bad * log(_XMAX) * 100
            return -np.sum(logpdf[finite_logpdf], axis=0) + penalty
        return -np.sum(logpdf, axis=0)

    def _penalized_nnlf(self, theta, x):
        ''' Return penalized negative loglikelihood function,
        i.e., - sum (log pdf(x, theta), axis=0) + penalty
           where theta are the parameters (including loc and scale)
        '''
        loc, scale, args = self._unpack_loc_scale(theta)
        if not self._argcheck(*args) or scale <= 0:
            return inf
        x = asarray((x-loc) / scale)
        n_log_scale = len(x) * log(scale)
        return self._nnlf_and_penalty(x, args) + n_log_scale

    # return starting point for fit (shape arguments + loc + scale)
    def _fitstart(self, data, args=None):
        if args is None:
            args = (1.0,)*self.numargs
        loc, scale = self._fit_loc_scale_support(data, *args)
        return args + (loc, scale)

    # Return the (possibly reduced) function to optimize in order to find MLE
    #  estimates for the .fit method
    def _reduce_func(self, args, kwds):
        # First of all, convert fshapes params to fnum: eg for stats.beta,
        # shapes='a, b'. To fix `a`, can specify either `f1` or `fa`.
        # Convert the latter into the former.
        if self.shapes:
            shapes = self.shapes.replace(',', ' ').split()
            for j, s in enumerate(shapes):
                val = kwds.pop('f' + s, None) or kwds.pop('fix_' + s, None)
                if val is not None:
                    key = 'f%d' % j
                    if key in kwds:
                        raise ValueError("Duplicate entry for %s." % key)
                    else:
                        kwds[key] = val

        args = list(args)
        Nargs = len(args)
        fixedn = []
        names = ['f%d' % n for n in range(Nargs - 2)] + ['floc', 'fscale']
        x0 = []
        for n, key in enumerate(names):
            if key in kwds:
                fixedn.append(n)
                args[n] = kwds.pop(key)
            else:
                x0.append(args[n])

        if len(fixedn) == 0:
            func = self._penalized_nnlf
            restore = None
        else:
            if len(fixedn) == Nargs:
                raise ValueError(
                    "All parameters fixed. There is nothing to optimize.")

            def restore(args, theta):
                # Replace with theta for all numbers not in fixedn
                # This allows the non-fixed values to vary, but
                #  we still call self.nnlf with all parameters.
                i = 0
                for n in range(Nargs):
                    if n not in fixedn:
                        args[n] = theta[i]
                        i += 1
                return args

            def func(theta, x):
                newtheta = restore(args[:], theta)
                return self._penalized_nnlf(newtheta, x)

        return x0, func, restore, args

    def fit(self, data, *args, **kwds):
        """
        Return MLEs for shape (if applicable), location, and scale
        parameters from data.

        MLE stands for Maximum Likelihood Estimate.  Starting estimates for
        the fit are given by input arguments; for any arguments not provided
        with starting estimates, ``self._fitstart(data)`` is called to generate
        such.

        One can hold some parameters fixed to specific values by passing in
        keyword arguments ``f0``, ``f1``, ..., ``fn`` (for shape parameters)
        and ``floc`` and ``fscale`` (for location and scale parameters,
        respectively).

        Parameters
        ----------
        data : array_like
            Data to use in calculating the MLEs.
        args : floats, optional
            Starting value(s) for any shape-characterizing arguments (those not
            provided will be determined by a call to ``_fitstart(data)``).
            No default value.
        kwds : floats, optional
            Starting values for the location and scale parameters; no default.
            Special keyword arguments are recognized as holding certain
            parameters fixed:

            - f0...fn : hold respective shape parameters fixed.
              Alternatively, shape parameters to fix can be specified by name.
              For example, if ``self.shapes == "a, b"``, ``fa``and ``fix_a``
              are equivalent to ``f0``, and ``fb`` and ``fix_b`` are
              equivalent to ``f1``.

            - floc : hold location parameter fixed to specified value.

            - fscale : hold scale parameter fixed to specified value.

            - optimizer : The optimizer to use.  The optimizer must take ``func``,
              and starting position as the first two arguments,
              plus ``args`` (for extra arguments to pass to the
              function to be optimized) and ``disp=0`` to suppress
              output as keyword arguments.

        Returns
        -------
        mle_tuple : tuple of floats
            MLEs for any shape parameters (if applicable), followed by those
            for location and scale. For most random variables, shape statistics
            will be returned, but there are exceptions (e.g. ``norm``).

        Notes
        -----
        This fit is computed by maximizing a log-likelihood function, with
        penalty applied for samples outside of range of the distribution. The
        returned answer is not guaranteed to be the globally optimal MLE, it
        may only be locally optimal, or the optimization may fail altogether.

        Examples
        --------

        Generate some data to fit: draw random variates from the `beta`
        distribution

        >>> from scipy.stats import beta
        >>> a, b = 1., 2.
        >>> x = beta.rvs(a, b, size=1000)

        Now we can fit all four parameters (``a``, ``b``, ``loc`` and ``scale``):

        >>> a1, b1, loc1, scale1 = beta.fit(x)

        We can also use some prior knowledge about the dataset: let's keep
        ``loc`` and ``scale`` fixed:

        >>> a1, b1, loc1, scale1 = beta.fit(x, floc=0, fscale=1)
        >>> loc1, scale1
        (0, 1)

        We can also keep shape parameters fixed by using ``f``-keywords. To
        keep the zero-th shape parameter ``a`` equal 1, use ``f0=1`` or,
        equivalently, ``fa=1``:

        >>> a1, b1, loc1, scale1 = beta.fit(x, fa=1, floc=0, fscale=1)
        >>> a1
        1

        Not all distributions return estimates for the shape parameters.
        ``norm`` for example just returns estimates for location and scale:

        >>> from scipy.stats import norm
        >>> x = norm.rvs(a, b, size=1000, random_state=123)
        >>> loc1, scale1 = norm.fit(x)
        >>> loc1, scale1
        (0.92087172783841631, 2.0015750750324668)
        """
        Narg = len(args)
        if Narg > self.numargs:
            raise TypeError("Too many input arguments.")

        start = [None]*2
        if (Narg < self.numargs) or not ('loc' in kwds and
                                         'scale' in kwds):
            # get distribution specific starting locations
            start = self._fitstart(data)
            args += start[Narg:-2]
        loc = kwds.pop('loc', start[-2])
        scale = kwds.pop('scale', start[-1])
        args += (loc, scale)
        x0, func, restore, args = self._reduce_func(args, kwds)

        optimizer = kwds.pop('optimizer', optimize.fmin)
        # convert string to function in scipy.optimize
        if not callable(optimizer) and isinstance(optimizer, str):
            if not optimizer.startswith('fmin_'):
                optimizer = "fmin_"+optimizer
            if optimizer == 'fmin_':
                optimizer = 'fmin'
            try:
                optimizer = getattr(optimize, optimizer)
            except AttributeError:
                raise ValueError("%s is not a valid optimizer" % optimizer)

        # by now kwds must be empty, since everybody took what they needed
        if kwds:
            raise TypeError("Unknown arguments: %s." % kwds)

        vals = optimizer(func, x0, args=(ravel(data),), disp=0)
        if restore is not None:
            vals = restore(args, vals)
        vals = tuple(vals)
        return vals

    def _fit_loc_scale_support(self, data, *args):
        """
        Estimate loc and scale parameters from data accounting for support.

        Parameters
        ----------
        data : array_like
            Data to fit.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).

        Returns
        -------
        Lhat : float
            Estimated location parameter for the data.
        Shat : float
            Estimated scale parameter for the data.

        """
        data = np.asarray(data)

        # Estimate location and scale according to the method of moments.
        loc_hat, scale_hat = self.fit_loc_scale(data, *args)

        # Compute the support according to the shape parameters.
        self._argcheck(*args)
        _a, _b = self._get_support(*args)
        a, b = _a, _b
        support_width = b - a

        # If the support is empty then return the moment-based estimates.
        if support_width <= 0:
            return loc_hat, scale_hat

        # Compute the proposed support according to the loc and scale
        # estimates.
        a_hat = loc_hat + a * scale_hat
        b_hat = loc_hat + b * scale_hat

        # Use the moment-based estimates if they are compatible with the data.
        data_a = np.min(data)
        data_b = np.max(data)
        if a_hat < data_a and data_b < b_hat:
            return loc_hat, scale_hat

        # Otherwise find other estimates that are compatible with the data.
        data_width = data_b - data_a
        rel_margin = 0.1
        margin = data_width * rel_margin

        # For a finite interval, both the location and scale
        # should have interesting values.
        if support_width < np.inf:
            loc_hat = (data_a - a) - margin
            scale_hat = (data_width + 2 * margin) / support_width
            return loc_hat, scale_hat

        # For a one-sided interval, use only an interesting location parameter.
        if a > -np.inf:
            return (data_a - a) - margin, 1
        elif b < np.inf:
            return (data_b - b) + margin, 1
        else:
            raise RuntimeError

    def fit_loc_scale(self, data, *args):
        """
        Estimate loc and scale parameters from data using 1st and 2nd moments.

        Parameters
        ----------
        data : array_like
            Data to fit.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).

        Returns
        -------
        Lhat : float
            Estimated location parameter for the data.
        Shat : float
            Estimated scale parameter for the data.

        """
        mu, mu2 = self.stats(*args, **{'moments': 'mv'})
        tmp = asarray(data)
        muhat = tmp.mean()
        mu2hat = tmp.var()
        Shat = sqrt(mu2hat / mu2)
        Lhat = muhat - Shat*mu
        if not np.isfinite(Lhat):
            Lhat = 0
        if not (np.isfinite(Shat) and (0 < Shat)):
            Shat = 1
        return Lhat, Shat

    def _entropy(self, *args):
        def integ(x):
            val = self._pdf(x, *args)
            return entr(val)

        # upper limit is often inf, so suppress warnings when integrating
        _a, _b = self._get_support(*args)
        olderr = np.seterr(over='ignore')
        h = integrate.quad(integ, _a, _b)[0]
        np.seterr(**olderr)

        if not np.isnan(h):
            return h
        else:
            # try with different limits if integration problems
            low, upp = self.ppf([1e-10, 1. - 1e-10], *args)
            if np.isinf(_b):
                upper = upp
            else:
                upper = _b
            if np.isinf(_a):
                lower = low
            else:
                lower = _a
            return integrate.quad(integ, lower, upper)[0]

    def expect(self, func=None, args=(), loc=0, scale=1, lb=None, ub=None,
               conditional=False, **kwds):
        """Calculate expected value of a function with respect to the
        distribution by numerical integration.

        The expected value of a function ``f(x)`` with respect to a
        distribution ``dist`` is defined as::

                    ub
            E[f(x)] = Integral(f(x) * dist.pdf(x)),
                    lb

        where ``ub`` and ``lb`` are arguments and ``x`` has the ``dist.pdf(x)``
        distribution. If the bounds ``lb`` and ``ub`` correspond to the
        support of the distribution, e.g. ``[-inf, inf]`` in the default
        case, then the integral is the unrestricted expectation of ``f(x)``.
        Also, the function ``f(x)`` may be defined such that ``f(x)`` is ``0``
        outside a finite interval in which case the expectation is
        calculated within the finite range ``[lb, ub]``.

        Parameters
        ----------
        func : callable, optional
            Function for which integral is calculated. Takes only one argument.
            The default is the identity mapping f(x) = x.
        args : tuple, optional
            Shape parameters of the distribution.
        loc : float, optional
            Location parameter (default=0).
        scale : float, optional
            Scale parameter (default=1).
        lb, ub : scalar, optional
            Lower and upper bound for integration. Default is set to the
            support of the distribution.
        conditional : bool, optional
            If True, the integral is corrected by the conditional probability
            of the integration interval.  The return value is the expectation
            of the function, conditional on being in the given interval.
            Default is False.

        Additional keyword arguments are passed to the integration routine.

        Returns
        -------
        expect : float
            The calculated expected value.

        Notes
        -----
        The integration behavior of this function is inherited from
        `scipy.integrate.quad`. Neither this function nor
        `scipy.integrate.quad` can verify whether the integral exists or is
        finite. For example ``cauchy(0).mean()`` returns ``np.nan`` and
        ``cauchy(0).expect()`` returns ``0.0``.

        Examples
        --------

        To understand the effect of the bounds of integration consider
        >>> from scipy.stats import expon
        >>> expon(1).expect(lambda x: 1, lb=0.0, ub=2.0)
        0.6321205588285578

        This is close to

        >>> expon(1).cdf(2.0) - expon(1).cdf(0.0)
        0.6321205588285577

        If ``conditional=True``

        >>> expon(1).expect(lambda x: 1, lb=0.0, ub=2.0, conditional=True)
        1.0000000000000002

        The slight deviation from 1 is due to numerical integration.
        """
        lockwds = {'loc': loc,
                   'scale': scale}
        self._argcheck(*args)
        _a, _b = self._get_support(*args)
        if func is None:
            def fun(x, *args):
                return x * self.pdf(x, *args, **lockwds)
        else:
            def fun(x, *args):
                return func(x) * self.pdf(x, *args, **lockwds)
        if lb is None:
            lb = loc + _a * scale
        if ub is None:
            ub = loc + _b * scale
        if conditional:
            invfac = (self.sf(lb, *args, **lockwds)
                      - self.sf(ub, *args, **lockwds))
        else:
            invfac = 1.0
        kwds['args'] = args
        # Silence floating point warnings from integration.
        olderr = np.seterr(all='ignore')
        vals = integrate.quad(fun, lb, ub, **kwds)[0] / invfac
        np.seterr(**olderr)
        return vals


class rv_discrete(rv_generic):

    def __new__(cls, a=0, b=inf, name=None, badvalue=None,
                moment_tol=1e-8, values=None, inc=1, longname=None,
                shapes=None, extradoc=None, seed=None):

        if values is not None:
            # dispatch to a subclass
            return super(rv_discrete, cls).__new__(rv_sample)
        else:
            # business as usual
            return super(rv_discrete, cls).__new__(cls)

    def _nonzero(self, k, *args):
        return floor(k) == k

    def _pmf(self, k, *args):
        return self._cdf(k, *args) - self._cdf(k-1, *args)

    def _logpmf(self, k, *args):
        return log(self._pmf(k, *args))

    def _cdf_single(self, k, *args):
        _a, _b = self._get_support(*args)
        m = arange(int(_a), k+1)
        return np.sum(self._pmf(m, *args), axis=0)

    def _cdf(self, x, *args):
        k = floor(x)
        return self._cdfvec(k, *args)

    # generic _logcdf, _sf, _logsf, _ppf, _isf, _rvs defined in rv_generic

    def rvs(self, *args, **kwargs):
        """
        Random variates of given type.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).
        size : int or tuple of ints, optional
            Defining number of random variates (Default is 1).  Note that `size`
            has to be given as keyword, not as positional argument.
        random_state : None or int or ``np.random.RandomState`` instance, optional
            If int or RandomState, use it for drawing the random variates.
            If None, rely on ``self.random_state``.
            Default is None.

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of given `size`.

        """
        kwargs['discrete'] = True
        return super(rv_discrete, self).rvs(*args, **kwargs)

    def pmf(self, k, *args, **kwds):
        """
        Probability mass function at k of the given RV.

        Parameters
        ----------
        k : array_like
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        pmf : array_like
            Probability mass function evaluated at k

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        _a, _b = self._get_support(*args)
        k, loc = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        k = asarray((k-loc))
        cond0 = self._argcheck(*args)
        cond1 = (k >= _a) & (k <= _b) & self._nonzero(k, *args)
        cond = cond0 & cond1
        output = zeros(shape(cond), 'd')
        place(output, (1-cond0) + np.isnan(k), self.badvalue)
        if np.any(cond):
            goodargs = argsreduce(cond, *((k,)+args))
            place(output, cond, np.clip(self._pmf(*goodargs), 0, 1))
        if output.ndim == 0:
            return output[()]
        return output

    def logpmf(self, k, *args, **kwds):
        """
        Log of the probability mass function at k of the given RV.

        Parameters
        ----------
        k : array_like
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter. Default is 0.

        Returns
        -------
        logpmf : array_like
            Log of the probability mass function evaluated at k.

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        _a, _b = self._get_support(*args)
        k, loc = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        k = asarray((k-loc))
        cond0 = self._argcheck(*args)
        cond1 = (k >= _a) & (k <= _b) & self._nonzero(k, *args)
        cond = cond0 & cond1
        output = empty(shape(cond), 'd')
        output.fill(-np.inf)
        place(output, (1-cond0) + np.isnan(k), self.badvalue)
        if np.any(cond):
            goodargs = argsreduce(cond, *((k,)+args))
            place(output, cond, self._logpmf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def cdf(self, k, *args, **kwds):
        """
        Cumulative distribution function of the given RV.

        Parameters
        ----------
        k : array_like, int
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        cdf : ndarray
            Cumulative distribution function evaluated at `k`.

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        _a, _b = self._get_support(*args)
        k, loc = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        k = asarray((k-loc))
        cond0 = self._argcheck(*args)
        cond1 = (k >= _a) & (k < _b)
        cond2 = (k >= _b)
        cond = cond0 & cond1
        output = zeros(shape(cond), 'd')
        place(output, (1-cond0) + np.isnan(k), self.badvalue)
        place(output, cond2*(cond0 == cond0), 1.0)

        if np.any(cond):
            goodargs = argsreduce(cond, *((k,)+args))
            place(output, cond, np.clip(self._cdf(*goodargs), 0, 1))
        if output.ndim == 0:
            return output[()]
        return output

    def logcdf(self, k, *args, **kwds):
        """
        Log of the cumulative distribution function at k of the given RV.

        Parameters
        ----------
        k : array_like, int
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        logcdf : array_like
            Log of the cumulative distribution function evaluated at k.

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        _a, _b = self._get_support(*args)
        k, loc = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        k = asarray((k-loc))
        cond0 = self._argcheck(*args)
        cond1 = (k >= _a) & (k < _b)
        cond2 = (k >= _b)
        cond = cond0 & cond1
        output = empty(shape(cond), 'd')
        output.fill(-np.inf)
        place(output, (1-cond0) + np.isnan(k), self.badvalue)
        place(output, cond2*(cond0 == cond0), 0.0)

        if np.any(cond):
            goodargs = argsreduce(cond, *((k,)+args))
            place(output, cond, self._logcdf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def sf(self, k, *args, **kwds):
        """
        Survival function (1 - `cdf`) at k of the given RV.

        Parameters
        ----------
        k : array_like
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        sf : array_like
            Survival function evaluated at k.

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        _a, _b = self._get_support(*args)
        k, loc = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        k = asarray(k-loc)
        cond0 = self._argcheck(*args)
        cond1 = (k >= _a) & (k < _b)
        cond2 = (k < _a) & cond0
        cond = cond0 & cond1
        output = zeros(shape(cond), 'd')
        place(output, (1-cond0) + np.isnan(k), self.badvalue)
        place(output, cond2, 1.0)
        if np.any(cond):
            goodargs = argsreduce(cond, *((k,)+args))
            place(output, cond, np.clip(self._sf(*goodargs), 0, 1))
        if output.ndim == 0:
            return output[()]
        return output

    def logsf(self, k, *args, **kwds):
        """
        Log of the survival function of the given RV.

        Returns the log of the "survival function," defined as 1 - `cdf`,
        evaluated at `k`.

        Parameters
        ----------
        k : array_like
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        logsf : ndarray
            Log of the survival function evaluated at `k`.

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        _a, _b = self._get_support(*args)
        k, loc = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        k = asarray(k-loc)
        cond0 = self._argcheck(*args)
        cond1 = (k >= _a) & (k < _b)
        cond2 = (k < _a) & cond0
        cond = cond0 & cond1
        output = empty(shape(cond), 'd')
        output.fill(-np.inf)
        place(output, (1-cond0) + np.isnan(k), self.badvalue)
        place(output, cond2, 0.0)
        if np.any(cond):
            goodargs = argsreduce(cond, *((k,)+args))
            place(output, cond, self._logsf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def ppf(self, q, *args, **kwds):
        """
        Percent point function (inverse of `cdf`) at q of the given RV.

        Parameters
        ----------
        q : array_like
            Lower tail probability.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        k : array_like
            Quantile corresponding to the lower tail probability, q.

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        _a, _b = self._get_support(*args)
        q, loc = map(asarray, (q, loc))
        args = tuple(map(asarray, args))
        cond0 = self._argcheck(*args) & (loc == loc)
        cond1 = (q > 0) & (q < 1)
        cond2 = (q == 1) & cond0
        cond = cond0 & cond1
        output = np.full(shape(cond), self.badvalue, typecode='d')
        # output type 'd' to handle nin and inf
        place(output, (q == 0)*(cond == cond), _a-1)
        place(output, cond2, _b)
        if np.any(cond):
            goodargs = argsreduce(cond, *((q,)+args+(loc,)))
            loc, goodargs = goodargs[-1], goodargs[:-1]
            place(output, cond, self._ppf(*goodargs) + loc)

        if output.ndim == 0:
            return output[()]
        return output

    def isf(self, q, *args, **kwds):
        """
        Inverse survival function (inverse of `sf`) at q of the given RV.

        Parameters
        ----------
        q : array_like
            Upper tail probability.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        k : ndarray or scalar
            Quantile corresponding to the upper tail probability, q.

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        _a, _b = self._get_support(*args)
        q, loc = map(asarray, (q, loc))
        args = tuple(map(asarray, args))
        cond0 = self._argcheck(*args) & (loc == loc)
        cond1 = (q > 0) & (q < 1)
        cond2 = (q == 1) & cond0
        cond = cond0 & cond1

        # same problem as with ppf; copied from ppf and changed
        output = np.full(shape(cond), self.badvalue).astype('d')
        # output type 'd' to handle nin and inf
        place(output, (q == 0)*(cond == cond), _b)
        place(output, cond2, _a-1)

        # call place only if at least 1 valid argument
        if np.any(cond):
            goodargs = argsreduce(cond, *((q,)+args+(loc,)))
            loc, goodargs = goodargs[-1], goodargs[:-1]
            # PB same as ticket 766
            place(output, cond, self._isf(*goodargs) + loc)

        if output.ndim == 0:
            return output[()]
        return output



class rv_sample(rv_discrete):
    """A 'sample' discrete distribution defined by the support and values.

       The ctor ignores most of the arguments, only needs the `values` argument.
    """
    def __init__(self, a=0, b=inf, name=None, badvalue=None,
                 moment_tol=1e-8, values=None, inc=1, longname=None,
                 shapes=None, extradoc=None, seed=None):

        super(rv_discrete, self).__init__(seed)

        if values is None:
            raise ValueError("rv_sample.__init__(..., values=None,...)")

        # cf generic freeze
        self._ctor_param = dict(
            a=a, b=b, name=name, badvalue=badvalue,
            moment_tol=moment_tol, values=values, inc=inc,
            longname=longname, shapes=shapes, extradoc=extradoc, seed=seed)

        if badvalue is None:
            badvalue = nan
        self.badvalue = badvalue
        self.moment_tol = moment_tol
        self.inc = inc
        self.shapes = shapes
        self.vecentropy = self._entropy

        xk, pk = values

        if np.shape(xk) != np.shape(pk):
            raise ValueError("xk and pk must have the same shape.")
        if np.less(pk, 0.0).any():
            raise ValueError("All elements of pk must be non-negative.")
        if not np.allclose(np.sum(pk), 1):
            raise ValueError("The sum of provided pk is not 1.")

        indx = np.argsort(np.ravel(xk))
        self.xk = np.take(np.ravel(xk), indx, 0)
        self.pk = np.take(np.ravel(pk), indx, 0)
        self.a = self.xk[0]
        self.b = self.xk[-1]

        self.qvals = np.cumsum(self.pk, axis=0)

        self.shapes = ' '   # bypass inspection
        self._construct_argparser(meths_to_inspect=[self._pmf],
                                  locscale_in='loc=0',
                                  # scale=1 for discrete RVs
                                  locscale_out='loc, 1')

        self._construct_docstrings(name, longname, extradoc)

    def _get_support(self, *args):
        """Return the support of the (unscaled, unshifted) distribution.

        Parameters
        ----------
        arg1, arg2, ... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        Returns
        -------
        a, b : numeric (float, or int or +/-np.inf)
            end-points of the distribution's support.
        """
        return self.a, self.b

    def _pmf(self, x):
        return np.select([x == k for k in self.xk],
                         [np.broadcast_arrays(p, x)[0] for p in self.pk], 0)

    def _cdf(self, x):
        xx, xxk = np.broadcast_arrays(x[:, None], self.xk)
        indx = np.argmax(xxk > xx, axis=-1) - 1
        return self.qvals[indx]

    def _ppf(self, q):
        qq, sqq = np.broadcast_arrays(q[..., None], self.qvals)
        indx = argmax(sqq >= qq, axis=-1)
        return self.xk[indx]

    def _rvs(self):
        # Need to define it explicitly, otherwise .rvs() with size=None
        # fails due to explicit broadcasting in _ppf
        U = self._random_state.random_sample(self._size)
        if self._size is None:
            U = np.array(U, ndmin=1)
            Y = self._ppf(U)[0]
        else:
            Y = self._ppf(U)
        return Y

    def generic_moment(self, n):
        n = asarray(n)
        return np.sum(self.xk**n[np.newaxis, ...] * self.pk, axis=0)



## Normal distribution

# loc = mu, scale = std
# Keep these implementations out of the class definition so they can be reused
# by other distributions.
_norm_pdf_C = np.sqrt(2*np.pi)
_norm_pdf_logC = np.log(_norm_pdf_C)


def _norm_pdf(x):
    return np.exp(-x**2/2.0) / _norm_pdf_C


def _norm_logpdf(x):
    return -x**2 / 2.0 - _norm_pdf_logC


def _norm_cdf(x):
    return sc.ndtr(x)


def _norm_logcdf(x):
    return sc.log_ndtr(x)


def _norm_ppf(q):
    return sc.ndtri(q)


def _norm_sf(x):
    return _norm_cdf(-x)


def _norm_logsf(x):
    return _norm_logcdf(-x)


def _norm_isf(q):
    return -_norm_ppf(q)



class truncnorm_gen(rv_continuous):
    r"""A truncated normal continuous random variable.

    %(before_notes)s

    Notes
    -----
    The standard form of this distribution is a standard normal truncated to
    the range [a, b] --- notice that a and b are defined over the domain of the
    standard normal.  To convert clip values for a specific mean and standard
    deviation, use::

        a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std

    `truncnorm` takes :math:`a` and :math:`b` as shape parameters.

    %(after_notes)s

    %(example)s

    """
    def _argcheck(self, a, b):
        return a < b

    def _get_support(self, a, b):
        return a, b

    def _get_norms(self, a, b):
        _nb = _norm_cdf(b)
        _na = _norm_cdf(a)
        _sb = _norm_sf(b)
        _sa = _norm_sf(a)
        _delta = np.where(a > 0, _sa - _sb, _nb - _na)
        with np.errstate(divide='ignore'):
            return _na, _nb, _sa, _sb, _delta, np.log(_delta)

    def _pdf(self, x, a, b):
        ans = self._get_norms(a, b)
        _delta = ans[4]
        return _norm_pdf(x) / _delta

    def _logpdf(self, x, a, b):
        ans = self._get_norms(a, b)
        _logdelta = ans[5]
        return _norm_logpdf(x) - _logdelta

    def _cdf(self, x, a, b):
        ans = self._get_norms(a, b)
        _na, _delta = ans[0], ans[4]
        return (_norm_cdf(x) - _na) / _delta

    def _ppf(self, q, a, b):
        # XXX Use _lazywhere...
        ans = self._get_norms(a, b)
        _na, _nb, _sa, _sb = ans[:4]
        ppf = np.where(a > 0,
                       _norm_isf(q*_sb + _sa*(1.0-q)),
                       _norm_ppf(q*_nb + _na*(1.0-q)))
        return ppf

    def _stats(self, a, b):
        ans = self._get_norms(a, b)
        nA, nB = ans[:2]
        d = nB - nA
        pA, pB = _norm_pdf(a), _norm_pdf(b)
        mu = (pA - pB) / d   # correction sign
        mu2 = 1 + (a*pA - b*pB) / d - mu*mu
        return mu, mu2, None, None


truncnorm = truncnorm_gen(name='truncnorm')
