from contextlib import contextmanager

g_override_dispatch = True

def set_global_override_dispatch(enabled: bool):
    global g_override_dispatch
    g_override_dispatch = enabled


def get_global_override_dispatch():
    global g_override_dispatch
    return g_override_dispatch


@contextmanager
def global_override_dispatch(enabled: bool):
    old_value = get_global_override_dispatch()
    set_global_override_dispatch(enabled)
    yield
    set_global_override_dispatch(old_value)