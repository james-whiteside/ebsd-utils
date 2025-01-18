class FieldNullError(ValueError):
    pass


class FieldsInconsistentError(ValueError):
    pass


class AggregateNullError(ValueError):
    pass


class AggregateInconsistentError(ValueError):
    pass


class PhaseMissingError(LookupError):
    pass
