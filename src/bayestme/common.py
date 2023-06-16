import enum


class InferenceType(enum.Enum):
    SVI = "SVI"
    MCMC = "MCMC"

    def __str__(self):
        return self.value
