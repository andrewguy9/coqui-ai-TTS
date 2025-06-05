def juxt(*fns):
    """
    Takes a set of functions and returns a fn that is the juxtaposition
    of those fns.  The returned fn takes a variable number of args, and
    returns a vector containing the result of applying each fn to the
    args (left-to-right).
    juxt(a b c)(x) => [a(x), b(x), c(x)]
    """
    def combined(*args, **kwargs):
        return [fn(*args, **kwargs) for fn in fns]
    return combined

def identity(x):
    return x

def uniq(xs):
    seen = set()
    for x in xs:
        if x not in seen:
            seen.add(x)
            yield x
