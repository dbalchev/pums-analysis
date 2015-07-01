class ModelField:
    def __init__(self, field_name):
        self._field_name = field_name

    def __call__(self, fields_dict):
        return self._create_from_str(fields_dict[self._field_name])

    def _create_from_str(self, string_repr):
        raise NotImplementedError()


class IntegerField(ModelField):
    def _create_from_str(self, string_repr):
        return int(string_repr)


class CodeBasedField(ModelField):
    def _create_from_str(self, string_repr):
        return self._codes[string_repr]
