"""
Rewrote the Signature module that fix default signature error for autodoc module.
"""

import inspect
import re
import types
import functools


def _sort_param(param_list, target_str):
    """Sort param_list as default order."""
    ls_certain = []
    for i in target_str.split(','):
        param_uncertain = i.split('=')[0].split(':')[0].strip().replace("*", "")
        if param_uncertain in param_list:
            ls_certain.append(param_uncertain)
    return ls_certain


def get_anotation(func):
    """Get anotation for string type."""
    anotation_dict = dict()
    source_code = inspect.getsource(func)
    func_code = func.__code__
    pos_count = func_code.co_argcount
    arg_names = func_code.co_varnames
    karg_pos = func_code.co_kwonlyargcount
    all_params_str = re.findall(r"def [\w_\d\-]+\(([\S\s]*?)(\):|\) ->.*?:)", source_code)
    all_params = all_params_str[0][0].replace("\n", "").replace("'", "\"")
    if "->" in all_params_str[0][1]:
        return_anotation = re.findall(r"->(.*?):\n", source_code)[0]
        anotation_dict["return"] = return_anotation
    kwargs_num = all_params.count("*") - all_params.count("**")
    all_param_names = list(arg_names[:pos_count + karg_pos + kwargs_num])

    # sub null spaces from matched all param str.
    re_space_sub = re.compile(r",\s{2,}")
    all_params = re_space_sub.sub(",", all_params)
    if ":" not in all_params:
        return None
    all_param_names = _sort_param(all_param_names, all_params)

    re_defaults_param = re.compile(r".*?)".join(["("+i for i in all_param_names]) + r".*)")
    anotation_params = re_defaults_param.findall(all_params)
    if anotation_params:
        if not isinstance(anotation_params[0], tuple):
            anotation_param = [anotation_params[0]]
        else:
            anotation_param = anotation_params[0]
        for i in anotation_param:
            if ":" not in i:
                continue
            key, value = i.split("=")[0].split(":")
            value = value.rstrip(', ')
            anotation_dict[key] = value

    return anotation_dict


def get_default_params(func):
    """ Get the default signatures from function. """
    source_code = inspect.getsource(func)
    func_code = func.__code__
    pos_count = func_code.co_argcount
    arg_names = func_code.co_varnames
    karg_pos = func_code.co_kwonlyargcount
    all_params_str = re.findall(r"def [\w_\d\-]+\(([\S\s]*?)(\):|\) ->.*?:)", source_code)
    all_params = all_params_str[0][0].replace("\n", "").replace("'", "\"")
    kwargs_num = all_params.count("*") - all_params.count("**")
    all_param_names = list(arg_names[:pos_count+karg_pos+kwargs_num])

    # sub null spaces from matched all param str.
    re_space_sub = re.compile(r",\s{2,}")
    all_params = re_space_sub.sub(",", all_params)

    all_param_names = _sort_param(all_param_names, all_params)

    re_defaults_param = re.compile(r"(.*?)".join(all_param_names) + r"(.*)")
    defaults_params = re_defaults_param.findall(all_params)
    if defaults_params:
        if isinstance(defaults_params[0], tuple):
            defaults_params = list([i[:-2] if i[-2:] == "**" else i for i in defaults_params[0]])
        defaults_params_list = []
        for i in defaults_params:
            if "=" in i and i:
                i = "".join(i.split('=')[-1]).strip(", ")
                if i[:6] == "lambda":
                    i = "<" + i + ">"
                defaults_params_list.append(i)
        defaults_params_tuple = tuple(defaults_params_list)
        return defaults_params_tuple
    return func.__defaults__


def _my_signature_from_function(cls, func):
    """Private helper: constructs Signature for the given python function."""

    is_duck_function = False
    if not inspect.isfunction(func):
        if inspect._signature_is_functionlike(func):  # pylint: disable=protected-access
            is_duck_function = True
        else:
            # If it's not a pure Python function, and not a duck type
            # of pure function:
            raise TypeError('{!r} is not a Python function'.format(func))

    Parameter = cls._parameter_cls   # pylint: disable=protected-access

    # Parameter information._partialmethod
    func_code = func.__code__
    pos_count = func_code.co_argcount
    arg_names = func_code.co_varnames
    positional = tuple(arg_names[:pos_count])
    keyword_only_count = func_code.co_kwonlyargcount
    keyword_only = arg_names[pos_count:(pos_count + keyword_only_count)]
    # annotations = func.__annotations__
    annotations = get_anotation(func)
    if not annotations:
        annotations = func.__annotations__
    defaults = get_default_params(func)
    if keyword_only_count == len(defaults):
        kwdefaults = dict()
        for num, arg_name in enumerate(keyword_only):
            kwdefaults[arg_name] = defaults[num]
    else:
        kwdefaults = func.__kwdefaults__
        if not isinstance(kwdefaults, type(None)):
            for key, value in kwdefaults.items():
                if isinstance(value, str):
                    kwdefaults[key] = '\'' + value + '\''
    pos_defaults = func.__defaults__

    if pos_defaults:
        pos_default_count = len(pos_defaults)
    else:
        pos_default_count = 0

    parameters = []

    # Non-keyword-only parameters w/o defaults.
    non_default_count = pos_count - pos_default_count
    for name in positional[:non_default_count]:
        annotation = annotations.get(name, inspect._empty)  # pylint: disable=protected-access
        parameters.append(Parameter(name, annotation=annotation,
                                    kind=inspect._POSITIONAL_OR_KEYWORD))  # pylint: disable=protected-access

    # ... w/ defaults.
    for offset, name in enumerate(positional[non_default_count:]):
        annotation = annotations.get(name, inspect._empty)  # pylint: disable=protected-access
        parameters.append(Parameter(name, annotation=annotation,
                                    kind=inspect._POSITIONAL_OR_KEYWORD,  # pylint: disable=protected-access
                                    default=defaults[offset]))

    # *args
    if func_code.co_flags & inspect.CO_VARARGS:
        name = arg_names[pos_count + keyword_only_count]
        annotation = annotations.get(name, inspect._empty)  # pylint: disable=protected-access
        parameters.append(Parameter(name, annotation=annotation,
                                    kind=inspect._VAR_POSITIONAL))  # pylint: disable=protected-access

    # Keyword-only parameters.
    for name in keyword_only:
        default = inspect._empty  # pylint: disable=protected-access
        if kwdefaults is not None:
            default = kwdefaults.get(name, inspect._empty)  # pylint: disable=protected-access

        annotation = annotations.get(name, inspect._empty)  # pylint: disable=protected-access
        parameters.append(Parameter(name, annotation=annotation,
                                    kind=inspect._KEYWORD_ONLY,  # pylint: disable=protected-access
                                    default=default))
    # **kwargs
    if func_code.co_flags & inspect.CO_VARKEYWORDS:
        index = pos_count + keyword_only_count
        if func_code.co_flags & inspect.CO_VARARGS:
            index += 1

        name = arg_names[index]
        annotation = annotations.get(name, inspect._empty)  # pylint: disable=protected-access
        parameters.append(Parameter(name, annotation=annotation,
                                    kind=inspect._VAR_KEYWORD))  # pylint: disable=protected-access

    # Is 'func' is a pure Python function - don't validate the
    # parameters list (for correct order and defaults), it should be OK.
    return cls(parameters,
               return_annotation=annotations.get('return', inspect._empty),  # pylint: disable=protected-access
               __validate_parameters__=is_duck_function)


def _my_signature_from_callable(obj, *,
                                follow_wrapper_chains=True,
                                skip_bound_arg=True,
                                sigcls):
    """Private helper function to get signature for arbitrary
    callable objects.
    """

    if not callable(obj):
        raise TypeError('{!r} is not a callable object'.format(obj))

    if isinstance(obj, types.MethodType):
        # In this case we skip the first parameter of the underlying
        # function (usually `self` or `cls`).
        sig = _my_signature_from_callable(
            obj.__func__,
            follow_wrapper_chains=follow_wrapper_chains,
            skip_bound_arg=skip_bound_arg,
            sigcls=sigcls)

        if skip_bound_arg:
            return inspect._signature_bound_method(sig)  # pylint: disable=protected-access
        return sig

    # Was this function wrapped by a decorator?
    if follow_wrapper_chains:
        obj = inspect.unwrap(obj, stop=(lambda f: hasattr(f, "__signature__")))
        if isinstance(obj, types.MethodType):
            # If the unwrapped object is a *method*, we might want to
            # skip its first parameter (self).
            # See test_signature_wrapped_bound_method for details.
            return _my_signature_from_callable(
                obj,
                follow_wrapper_chains=follow_wrapper_chains,
                skip_bound_arg=skip_bound_arg,
                sigcls=sigcls)

    try:
        sig = obj.__signature__
    except AttributeError:
        pass
    else:
        if sig is not None:
            if not isinstance(sig, MySignature):
                raise TypeError(
                    'unexpected object {!r} in __signature__ '
                    'attribute'.format(sig))
            return sig

    try:
        partialmethod = obj._partialmethod  # pylint: disable=protected-access
    except AttributeError:
        pass
    else:
        if isinstance(partialmethod, functools.partialmethod):
            # Unbound partialmethod (see functools.partialmethod)
            # This means, that we need to calculate the signature
            # as if it's a regular partial object, but taking into
            # account that the first positional argument
            # (usually `self`, or `cls`) will not be passed
            # automatically (as for boundmethods)

            wrapped_sig = _my_signature_from_callable(
                partialmethod.func,
                follow_wrapper_chains=follow_wrapper_chains,
                skip_bound_arg=skip_bound_arg,
                sigcls=sigcls)

            sig = inspect._signature_get_partial(wrapped_sig, partialmethod, (None,))  # pylint: disable=protected-access
            first_wrapped_param = tuple(wrapped_sig.parameters.values())[0]
            if first_wrapped_param.kind is Parameter.VAR_POSITIONAL:  # pylint: disable=no-else-return
                # First argument of the wrapped callable is `*args`, as in
                # `partialmethod(lambda *args)`.
                return sig
            else:
                sig_params = tuple(sig.parameters.values())
                assert (not sig_params or
                        first_wrapped_param is not sig_params[0])
                new_params = (first_wrapped_param,) + sig_params
                return sig.replace(parameters=new_params)

    if inspect.isfunction(obj) or inspect._signature_is_functionlike(obj):  # pylint: disable=protected-access
        # If it's a pure Python function, or an object that is duck type
        # of a Python function (Cython functions, for instance), then:
        return _my_signature_from_function(sigcls, obj)

    if inspect._signature_is_builtin(obj):  # pylint: disable=protected-access
        return inspect._signature_from_builtin(sigcls, obj,  # pylint: disable=protected-access
                                               skip_bound_arg=skip_bound_arg)

    if isinstance(obj, functools.partial):
        wrapped_sig = _my_signature_from_callable(
            obj.func,
            follow_wrapper_chains=follow_wrapper_chains,
            skip_bound_arg=skip_bound_arg,
            sigcls=sigcls)
        return inspect._signature_get_partial(wrapped_sig, obj)  # pylint: disable=protected-access

    sig = None
    if isinstance(obj, type):
        # obj is a class or a metaclass

        # First, let's see if it has an overloaded __call__ defined
        # in its metaclass
        call = inspect._signature_get_user_defined_method(type(obj), '__call__')  # pylint: disable=protected-access
        if call is not None:
            sig = _my_signature_from_callable(
                call,
                follow_wrapper_chains=follow_wrapper_chains,
                skip_bound_arg=skip_bound_arg,
                sigcls=sigcls)
        else:
            # Now we check if the 'obj' class has a '__new__' method
            new = inspect._signature_get_user_defined_method(obj, '__new__')  # pylint: disable=protected-access
            if new is not None:
                sig = _my_signature_from_callable(
                    new,
                    follow_wrapper_chains=follow_wrapper_chains,
                    skip_bound_arg=skip_bound_arg,
                    sigcls=sigcls)
            else:
                # Finally, we should have at least __init__ implemented
                init = inspect._signature_get_user_defined_method(obj, '__init__')  # pylint: disable=protected-access
                if init is not None:
                    sig = _my_signature_from_callable(
                        init,
                        follow_wrapper_chains=follow_wrapper_chains,
                        skip_bound_arg=skip_bound_arg,
                        sigcls=sigcls)

        if sig is None:
            # At this point we know, that `obj` is a class, with no user-
            # defined '__init__', '__new__', or class-level '__call__'

            for base in obj.__mro__[:-1]:
                # Since '__text_signature__' is implemented as a
                # descriptor that extracts text signature from the
                # class docstring, if 'obj' is derived from a builtin
                # class, its own '__text_signature__' may be 'None'.
                # Therefore, we go through the MRO (except the last
                # class in there, which is 'object') to find the first
                # class with non-empty text signature.
                try:
                    text_sig = base.__text_signature__
                except AttributeError:
                    pass
                else:
                    if text_sig:
                        # If 'obj' class has a __text_signature__ attribute:
                        # return a signature based on it
                        return inspect._signature_fromstr(sigcls, obj, text_sig)  # pylint: disable=protected-access

            # No '__text_signature__' was found for the 'obj' class.
            # Last option is to check if its '__init__' is
            # object.__init__ or type.__init__.
            if type not in obj.__mro__:
                # We have a class (not metaclass), but no user-defined
                # __init__ or __new__ for it
                if (obj.__init__ is object.__init__ and  # pylint: disable=no-else-return
                        obj.__new__ is object.__new__):
                    # Return a signature of 'object' builtin.
                    return sigcls.from_callable(object)
                else:
                    raise ValueError(
                        'no signature found for builtin type {!r}'.format(obj))

    elif not isinstance(obj, inspect._NonUserDefinedCallables):  # pylint: disable=protected-access
        # An object with __call__
        # We also check that the 'obj' is not an instance of
        # _WrapperDescriptor or _MethodWrapper to avoid
        # infinite recursion (and even potential segfault)
        call = inspect._signature_get_user_defined_method(type(obj), '__call__')  # pylint: disable=protected-access
        if call is not None:
            try:
                sig = _my_signature_from_callable(
                    call,
                    follow_wrapper_chains=follow_wrapper_chains,
                    skip_bound_arg=skip_bound_arg,
                    sigcls=sigcls)
            except ValueError as ex:
                msg = 'no signature found for {!r}'.format(obj)
                raise ValueError(msg) from ex

    if sig is not None:
        # For classes and objects we skip the first parameter of their
        # __call__, __new__, or __init__ methods
        if skip_bound_arg:
            return inspect._signature_bound_method(sig)  # pylint: disable=protected-access
        return sig

    if isinstance(obj, types.BuiltinFunctionType):
        # Raise a nicer error message for builtins
        msg = 'no signature found for builtin function {!r}'.format(obj)
        raise ValueError(msg)

    raise ValueError('callable {!r} is not supported by signature'.format(obj))


class MySignature(inspect.Signature):

    @classmethod
    def from_callable(cls, obj, *, follow_wrapped=True):
        """Constructs Signature for the given callable object."""
        return _my_signature_from_callable(obj, sigcls=cls,
                                           follow_wrapper_chains=follow_wrapped)


def signature(obj, *, follow_wrapped=True):
    """Get a signature object for the passed callable."""
    return MySignature.from_callable(obj, follow_wrapped=follow_wrapped)
