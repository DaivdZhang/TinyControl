class LinearTimeInvariant(object):
    def __init__(self, inputs=1, outputs=1, dt=None):
        self.inputs = inputs
        self.outputs = outputs
        self.dt = dt

    def isctime(self, strict=False):
        if self.dt is None:
            return False if strict else True
        return self.dt == 0

    def isdtime(self, strict):
        if self.dt is None:
            return False if strict else True
        return self.dt > 0

    def issiso(self):
        return self.outputs == 1 and self.inputs == 1


def isctime(sys_, strict=False):
    if sys_.dt is None:
        return False if strict else True
    return sys_.dt == 0


def isdtime(sys_, strict=False):
    if sys_.dt is None:
        return False if strict else True
    return sys_.dt > 0


def issiso(sys_):
    return sys_.outputs == 1 and sys_.inputs == 1
