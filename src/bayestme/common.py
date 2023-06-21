import enum
import numpy


ArrayType = numpy.array


class InferenceType(enum.Enum):
    SVI = "SVI"
    MCMC = "MCMC"

    def __str__(self):
        return self.value
