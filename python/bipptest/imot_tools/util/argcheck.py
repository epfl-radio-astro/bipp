# #############################################################################
# argcheck.py
# ===========
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

"""
Helper functions to ease argument checking.
"""

import collections.abc as abc
import functools
import inspect
import keyword
import math
import numbers

import numpy as np
import scipy.sparse as sparse


def check(*args):
    """
    Validate function parameters using boolean tests.

    It is common to check parameters for correctness before executing the function/class to which
    they are bound using boolean tests.  :py:func:`~imot_tools.util.argcheck.check` is a decorator
    that intercepts the output of boolean functions and raises :py:exc:`ValueError` when the result
    is :py:obj:`False`.

    Parameters
    ----------
    *args
        2 invocations supported:

        a) 2-argument mode:

            * `args[0]`: name of the decorated functon's parameter to test.
            * `args[1]`: boolean function to apply to the parameter value.

        b) 1-argument mode: (parameter-name -> boolean function) map.

    Returns
    -------
    :py:obj:`~typing.Callable`
        Function decorator.

    Raises
    ------
    :py:exc:`ValueError`
        If any of the boolean functions return :py:obj:`False`.

    Examples
    --------
    .. testsetup::

       from imot_tools.util.argcheck import check, require_all

       def is_5(obj):
           return obj == 5

       def is_int(obj):
           return isinstance(obj, int)

       def is_str(obj):
           return isinstance(obj, str)

    Suppose we have the following boolean functions to test an object for similarity to the number
    5:

    .. doctest::

       >>> def is_5(obj):
       ...     return obj == 5

       >>> def is_int(obj):
       ...     return isinstance(obj, int)

       >>> def is_str(obj):
       ...     return isinstance(obj, str)

    When used in conjunction with :py:func:`~imot_tools.util.argcheck.check`, type-checking function
    parameters becomes possible:

    .. doctest::

       >>> @check('x', is_5)  # 2-argument mode
       ... def f(x):
       ...     return x

       >>> f(5)
       5

       >>> f(4)
       Traceback (most recent call last):
           ...
       ValueError: Parameter[x] of f() does not satisfy is_5().

    .. doctest::

       >>> @check(dict(x=is_str, y=is_int))  # 1-argument mode
       ... def g(x, y):
       ...     return x, y

       >>> g('5', 3)
       ('5', 3)

       >>> g(5, 3)
       Traceback (most recent call last):
           ...
       ValueError: Parameter[x] of g() does not satisfy is_str().
    """
    if len(args) == 1:
        return _check(m=args[0])
    elif len(args) == 2:
        return _check(m={args[0]: args[1]})
    else:
        raise ValueError("Expected 1 or 2 arguments.")


def _check(m):
    if not isinstance(m, abc.Mapping):
        raise TypeError("Expected (str, boolean function) map")

    key_error = lambda k: f"Key[{k}] must be a valid string identifier."
    value_error = lambda k: f"Value[Key[{k}]] must be a boolean function."

    for k, v in m.items():
        if not isinstance(k, str):
            raise TypeError(key_error(k))
        if not (k.isidentifier() and (not keyword.iskeyword(k))):
            raise ValueError(key_error(k))

        if not inspect.isfunction(v):
            raise TypeError(value_error(k))

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_args = inspect.getcallargs(func, *args, **kwargs)

            for k, fn in m.items():
                if k not in func_args:
                    raise ValueError(
                        f"Parameter[{k}] not part of {func.__qualname__}() parameter list."
                    )

                if fn(func_args[k]) is False:
                    raise ValueError(
                        f"Parameter[{k}] of {func.__qualname__}()"
                        f" does not satisfy {fn.__name__}()."
                    )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def allow_None(func):
    """
    Relax boolean function for :py:obj:`None` input.

    A boolean function wrapped by :py:func:`~imot_tools.util.argcheck.allow_None` returns
    :py:obj:`True` if it's input is :py:obj:`None`.

    Parameters
    ----------
    func : :py:obj:`~typing.Callable`
        Boolean function.

    Returns
    -------
    :py:obj:`~typing.Callable`
        Boolean function.

    Examples
    --------
    .. testsetup::

       from imot_tools.util.argcheck import allow_None, check, is_integer

    .. doctest::

       >>> def is_5(x):
       ...     return x == 5

       >>> is_5(5), is_5(None)
       (True, False)

       >>> allow_None(is_5)(None)
       True

    When used in conjunction with :py:func:`~imot_tools.util.argcheck.check`, it is possible to
    type-check parameters having default arguments set to :py:obj:`None`:

    .. doctest::

       >>> @check('x', is_integer)
       ... def f(x: int = None):
       ...     return print(x)

       >>> f()  # ValueError because is_integer(None) is False.
       Traceback (most recent call last):
           ...
       ValueError: Parameter[x] of f() does not satisfy is_integer().

    .. doctest::

       >>> @check('x', allow_None(is_integer))  # redefined to allow None.
       ... def g(x: int = None):
       ...     return print(x)

       >>> g()  # Now it works.
       None
    """
    if not inspect.isfunction(func):
        raise TypeError("Parameter[func] must be a boolean function.")

    @functools.wraps(func)
    def wrapper(x):
        if x is None:
            return True

        return func(x)

    wrapper.__name__ = f"allow_None({func.__name__})"

    return wrapper


def accept_any(*funcs):
    """
    Lazy union of boolean functions.

    Parameters
    ----------
    *funcs : list(bool_func)
        Boolean functions.

    Returns
    -------
    :py:obj:`~typing.Callable`
        Boolean function.

    Examples
    --------
    .. testsetup::

       import numpy as np

       from imot_tools.util.argcheck import accept_any, check, has_shape

    .. doctest::

       >>> def is_int(x):
       ...     if isinstance(x, int):
       ...         return True
       ...     return False

       >>> def is_5(x):
       ...     if x == 5:
       ...         return True
       ...     return False

       >>> accept_any(is_int, is_5)(4)  # passes is_int(), is_5() un-tested
       True

       >>> accept_any(is_int, is_5)(np.r_[5][0])  # passes is_5()
       True

       >>> accept_any(is_int, is_5)('5')  # fails both
       False

    When used with :py:func:`~imot_tools.util.argcheck.check`, a parameter  can be verified to
    satisfy one of several choices.

    .. doctest::

       >>> @check('x', accept_any(has_shape([2, 2]), has_shape([3, 3])))
       ... def z_rot_trace(x: np.ndarray):
       ...     return np.trace(x)

       >>> z_rot_trace(x=np.diag(np.arange(1, 3)))
       3

       >>> z_rot_trace(x=np.diag(np.arange(1, 4)))
       6
    """
    if not all(inspect.isfunction(_) for _ in funcs):
        raise TypeError("Parameter[*funcs] must contain boolean functions.")

    def union(x):
        for fn in funcs:
            if fn(x) is True:
                return True

        return False

    union.__name__ = f"accept_any({[fn.__name__ for fn in funcs]})"

    return union


def require_all(*funcs):
    """
    Lazy intersection of boolean functions.

    Parameters
    ----------
    *funcs : list(bool_func)
        Boolean functions.

    Returns
    -------
    :py:obj:`~typing.Callable`
        Boolean function.

    Examples
    --------
    .. testsetup::

       from imot_tools.util.argcheck import require_all, check

    .. doctest::

       >>> def is_int(x):
       ...     if isinstance(x, int):
       ...         return True
       ...     return False

       >>> def is_5(x):
       ...     if x == 5:
       ...         return True
       ...     return False

       >>> require_all(is_int, is_5)('5')  # fails is_int()
       False

       >>> require_all(is_int, is_5)(4)  # passes is_int(), fails is_5()
       False

       >>> require_all(is_int, is_5)(5)  # both pass
       True

    When used with :py:func:`~imot_tools.util.argcheck.check`, a parameter can be verified to
    satisfy several functions simultaneously:

    .. doctest::

       >>> def le_5(x: int):
       ...     if x <= 5:
       ...         return True
       ...     return False

       >>> def gt_0(x: int):
       ...     if x > 0:
       ...         return True
       ...     return False

       >>> @check('x', require_all(gt_0, le_5))
       ... def f(x):
       ...     return x

       >>> f(3)
       3

       >>> f(-1)
       Traceback (most recent call last):
           ...
       ValueError: Parameter[x] of f() does not satisfy require_all(['gt_0', 'le_5'])().
    """
    if not all(inspect.isfunction(_) for _ in funcs):
        raise TypeError("Parameter[*funcs] must contain boolean functions.")

    def intersection(x):
        for fn in funcs:
            if fn(x) is False:
                return False

        return True

    intersection.__name__ = f"require_all({[fn.__name__ for fn in funcs]})"

    return intersection


def is_instance(*klass):
    """
    Validate instance types.

    Parameters
    ----------
    *klass : list(type)
        Accepted classes.

    Returns
    -------
    :py:obj:`~typing.Callable`
        Boolean function.

    Examples
    --------
    .. testsetup::

       import numpy as np

       from imot_tools.util.argcheck import is_instance, check

    .. doctest::

       >>> is_instance(str, int)('5')
       True

       >>> is_instance(np.ndarray)([])
       False

    When used with :py:func:`~imot_tools.util.argcheck.check`, function parameters can verified to
    be of a certain type:

    .. doctest::

       >>> @check('x', is_instance(str))
       ... def f(x):
       ...     return x

       >>> f('hello')
       'hello'

       >>> f(5)
       Traceback (most recent call last):
           ...
       ValueError: Parameter[x] of f() does not satisfy is_instance(['str'])().
    """
    if not all(inspect.isclass(_) for _ in klass):
        raise TypeError("Parameter[*klass] must contain types.")

    def _is_instance(x):
        if isinstance(x, klass):
            return True

        return False

    _is_instance.__name__ = f"is_instance({[cl.__name__ for cl in klass]})"

    return _is_instance


def is_scalar(x):
    """
    Return :py:obj:`True` if `x` is a scalar object.

    Examples
    --------
    .. testsetup::

       from imot_tools.util.argcheck import is_scalar

    .. doctest::

       >>> is_scalar(5)
       True

       >>> is_scalar([5])
       False
    """
    if not isinstance(x, abc.Container):
        return True

    return False


def is_array_like(x):
    """
    Return :py:obj:`True` if `x` is an array-like object.

    Examples
    --------
    .. testsetup::

       from imot_tools.util.argcheck import is_array_like

    .. doctest::

       >>> is_array_like(5)
       False

       >>> [is_array_like(_) for _ in (tuple(), np.array([]), range(5))]
       [True, True, True]

       >>> [is_array_like(_) for _ in (set(), dict())]
       [False, False]
    """
    if isinstance(x, (np.ndarray, abc.Sequence)):
        return True

    return False


def is_array_shape(x):
    """
    Return :py:obj:`True` if `x` is a valid array shape specifier.

    Examples
    --------
    .. testsetup::

       from imot_tools.util.argcheck import is_array_shape

    .. doctest::

       >>> is_array_shape((5, 4))
       True

       >>> is_array_shape((5, 0))
       False
    """
    if is_array_like(x):
        x = np.array(x, copy=False)

        if x.ndim == 1:
            if (len(x) > 0) and np.issubdtype(x.dtype, np.integer) and np.all(x > 0):
                return True

    return False


def has_shape(shape):
    """
    Validate array shapes.

    Parameters
    ----------
    shape : list(int)
        Desired array dimensions.

    Returns
    -------
    :py:obj:`~typing.Callable`
        Boolean function.

    Examples
    --------
    .. testsetup::

       from imot_tools.util.argcheck import has_shape, check

    .. doctest::

       >>> has_shape((1,))([5,])
       True

       >>> has_shape([5,])((1, 2))
       False
    """
    if not is_array_shape(shape):
        raise ValueError("Parameter[shape] must be a valid shape specifier.")

    shape = tuple(shape)

    def _has_shape(x):
        if is_array_like(x):
            x = np.array(x, copy=False)
        elif sparse.isspmatrix(x):
            pass
        else:
            return False

        if x.shape == shape:
            return True

        return False

    _has_shape.__name__ = f"has_shape({list(shape)})"

    return _has_shape


def has_ndim(ndim):
    """
    Validate array dimensions.

    Parameters
    ----------
    ndim : int
        Desired number of dimensions.

    Returns
    -------
    :py:obj:`~typing.Callable`
        Boolean function.

    Examples
    --------
    .. testsetup::

       from imot_tools.util.argcheck import has_ndim, check

    .. doctest::

       >>> has_ndim(1)([5,])
       True

       >>> has_ndim(2)((1,))
       False
    """
    if not ((is_integer(ndim)) and (ndim > 0)):
        raise ValueError("Parameter[ndim] must be positive.")

    def _has_ndim(x):
        if is_array_like(x):
            x = np.array(x, copy=False)
        else:
            return False

        if x.ndim == ndim:
            return True

        return False

    _has_ndim.__name__ = f"has_ndim({ndim})"

    return _has_ndim


def is_integer(x):
    """
    Return :py:obj:`True` if `x` is an integer.

    Examples
    --------
    .. testsetup::

       from imot_tools.util.argcheck import is_integer

    .. doctest::

       >>> is_integer(5)
       True

       >>> is_integer(5.0)
       False
    """
    return isinstance(x, numbers.Integral)


def has_integers(x):
    """
    Return :py:obj:`True` if `x` contains integers.

    Examples
    --------
    .. testsetup::

       import numpy as np
       from imot_tools.util.argcheck import has_integers

    .. doctest::

       >>> has_integers([5]), has_integers(np.r_[:5])
       (True, True)

       >>> has_integers([5.]), has_integers(np.ones((5, 3)))
       (False, False)
    """
    if is_array_like(x):
        x = np.array(x, copy=False)

        if np.issubdtype(x.dtype, np.integer):
            return True

    return False


def is_boolean(x):
    """
    Return :py:obj:`True` if `x` is a boolean.

    Examples
    --------
    .. testsetup::

       from imot_tools.util.argcheck import is_boolean

    .. doctest::

       >>> is_boolean(True), is_boolean(False)
       (True, True)

       >>> is_boolean(0), is_boolean(1)
       (False, False)
    """
    if isinstance(x, bool):
        return True

    return False


def has_booleans(x):
    """
    Return :py:obj:`True` if `x` contains booleans.

    Examples
    --------
    .. testsetup::

       import numpy as np

       from imot_tools.util.argcheck import has_booleans

    .. doctest::

       >>> has_booleans(np.ones((1, 2), dtype=bool)), has_booleans([True])
       (True, True)

       >>> has_booleans(np.ones((1, 2)))
       False
    """
    if is_array_like(x):
        x = np.array(x, copy=False)

        if np.issubdtype(x.dtype, np.bool_):
            return True

    return False


def is_even(x):
    """
    Return :py:obj:`True` if `x` is an even integer.

    Examples
    --------
    .. testsetup::

       from imot_tools.util.argcheck import is_even

    .. doctest::

       >>> is_even(2)
       True

       >>> is_even(3)
       False
    """
    if is_integer(x):
        if x % 2 == 0:
            return True

    return False


def has_evens(x):
    """
    Return :py:obj:`True` if `x` contains even integers.

    Examples
    --------
    .. testsetup::

       import numpy as np
       from imot_tools.util.argcheck import has_evens

    .. doctest::

       >>> has_evens(np.arange(5))
       False

       >>> has_evens(np.arange(0, 6, 2))
       True
    """
    if has_integers(x):
        x = np.array(x, copy=False)

        if np.all(x % 2 == 0):
            return True

    return False


def is_odd(x):
    """
    Return :py:obj:`True` if `x` is an odd integer.

    Examples
    --------
    .. testsetup::

       from imot_tools.util.argcheck import is_odd

    .. doctest::

       >>> is_odd(2)
       False

       >>> is_odd(3)
       True
    """
    if is_integer(x):
        if x % 2 == 1:
            return True

    return False


def has_odds(x):
    """
    Return :py:obj:`True` if `x` contains odd integers.

    Examples
    --------
    .. testsetup::

       import numpy as np

       from imot_tools.util.argcheck import has_odds

    .. doctest::

       >>> has_odds(np.arange(5))
       False

       >>> has_odds(np.arange(1, 7, 2))
       True
    """
    if has_integers(x):
        x = np.array(x, copy=False)

        if np.all(x % 2 == 1):
            return True

    return False


def is_pow2(x):
    """
    Return :py:obj:`True` if `x` is a power of 2.

    Examples
    --------
    .. testsetup::

       from imot_tools.util.argcheck import is_pow2

    .. doctest::

       >>> is_pow2(8)
       True

       >>> is_pow2(9)
       False
    """
    if is_integer(x):
        if x > 0:
            exp = math.log2(x)
            if math.isclose(exp, math.floor(exp)):
                return True

    return False


def has_pow2s(x):
    """
    Return :py:obj:`True` if `x` contains powers of 2.

    Examples
    --------
    .. testsetup::

       import numpy as np

       from imot_tools.util.argcheck import has_pow2s

    .. doctest::

       >>> has_pow2s([2, 4, 8])
       True

       >>> has_pow2s(np.arange(10))
       False
    """
    if has_integers(x):
        x = np.array(x, copy=False)

        if np.all(x > 0):
            exp = np.log2(x)
            if np.allclose(exp, np.floor(exp)):
                return True

    return False


def is_complex(x):
    """
    Return :py:obj:`True` if `x` is a complex number.

    Examples
    --------
    .. testsetup::

       from imot_tools.util.argcheck import is_complex

    .. doctest::

       >>> is_complex(5), is_complex(5.0)
       (False, False)

       >>> is_complex(5 + 5j), is_complex(1j * np.r_[0][0])
       (True, True)
    """
    if isinstance(x, numbers.Complex) and (not isinstance(x, numbers.Real)):
        return True

    return False


def has_complex(x):
    """
    Return :py:obj:`True` if `x` contains complex numbers.

    Examples
    --------
    .. testsetup::

       from imot_tools.util.argcheck import has_complex

    .. doctest::

       >>> has_complex([1j, 0])  # upcast to complex numbers.
       True

       >>> has_complex(1j * np.ones((5, 3)))
       True
    """
    if is_array_like(x):
        x = np.array(x, copy=False)

        if np.issubdtype(x.dtype, np.complexfloating):
            return True

    return False


def is_real(x):
    """
    Return :py:obj:`True` if `x` is a real number.

    Examples
    --------
    .. testsetup::

       from imot_tools.util.argcheck import is_real

    .. doctest::

       >>> is_real(5), is_real(5.0)
       (True, True)

       >>> is_real(1j)
       False
    """
    return isinstance(x, numbers.Real)


def has_reals(x):
    """
    Return :py:obj:`True` if `x` contains real numbers.

    Examples
    --------
    .. testsetup::

       from imot_tools.util.argcheck import has_reals

    .. doctest::

       >>> has_reals([5]), has_reals(np.arange(10))
       (True, True)

       >>> has_reals(1j * np.ones(5))
       False
    """
    if is_array_like(x):
        x = np.array(x, copy=False)

        if np.issubdtype(x.dtype, np.integer) or np.issubdtype(x.dtype, np.floating):
            return True

    return False
