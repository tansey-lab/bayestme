import enum
import numpy
import numpy.random
from typing import Optional

ArrayType = numpy.array


class InferenceType(enum.Enum):
    SVI = "SVI"
    MCMC = "MCMC"

    def __str__(self):
        return self.value


def create_rng(seed: Optional[int]):
    if seed is not None:
        seed = numpy.random.SeedSequence(seed).generate_state(1)
        return numpy.random.default_rng(seed.item())
    else:
        return numpy.random.default_rng()
