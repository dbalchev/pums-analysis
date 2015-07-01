class _ModelBase:
    @classmethod
    def from_dict(cls, fields_dict):
        new_obj = cls()
        for name, field_type in cls._fields:
            setattr(new_obj, name, field_type(fields_dict))
        return new_obj


class _ModelField:
    def __init__(self, field_name):
        self._field_name = field_name

    def __call__(self, fields_dict):
        return self._create_from_str(fields_dict[self._field_name])

    def _create_from_str(self, string_repr):
        raise NotImplementedError()


class Integer(_ModelField):
    def _create_from_str(self, string_repr):
        return int(string_repr)


class CodeBasedField(_ModelField):
    def _create_from_str(self, string_repr):
        return self._codes[string_repr]


class _ModelMeta(type):
    def __new__(cls, name, bases, namespace, **kws):
        fields = [x for x in namespace.items() if isinstance(x[1], _ModelField)]
        namespace = {name:value for name,value in namespace.items() \
            if not isinstance(value, _ModelField)}
        namespace["__slots__"] = tuple(f[0] for f in fields)
        new_class = type.__new__(cls, name, bases, namespace)
        new_class._fields = fields
        return new_class


class Model(_ModelBase, metaclass=_ModelMeta):
    pass
