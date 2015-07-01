from base.fields import ModelField


class _ModelBase:
    @classmethod
    def from_dict(cls, fields_dict):
        new_obj = cls()
        for name, field_type in cls._fields:
            setattr(new_obj, name, field_type(fields_dict))
        return new_obj

    def __str__(self):
        fields = ", ".join('"{}": "{}"'.format(name, getattr(self, name)) \
            for name, _ in type(self)._fields)
        return "{" + fields + "}"


class _ModelMeta(type):
    def __new__(cls, name, bases, namespace, **kws):
        fields = [x for x in namespace.items() if isinstance(x[1], ModelField)]
        namespace = {name:value for name,value in namespace.items() \
            if not isinstance(value, ModelField)}
        namespace["__slots__"] = tuple(f[0] for f in fields)
        new_class = type.__new__(cls, name, bases, namespace)
        new_class._fields = fields
        return new_class


class Model(_ModelBase, metaclass=_ModelMeta):
    pass
