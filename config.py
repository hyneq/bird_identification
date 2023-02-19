from typing import Optional, Callable
from dataclasses import fields, is_dataclass

def merge_conf(cfg_cls: type):

    if not is_dataclass(cfg_cls):
        raise TypeError("Config class must be a dataclass")
    
    cfg_fields = fields(cfg_cls)

    def merge_conf_with_cls(func: Callable) -> Callable:

        def inner(*args, cfg: Optional[cfg_cls]=None, **kwargs):
            if cfg:
                if not isinstance(cfg, cfg_cls):
                    raise TypeError("Config object must be a {} instance".format(cfg_cls.__name__))

                for field in cfg_fields:
                    if kwargs.get(field.name, None) is None:
                        kwargs[field.name] = getattr(cfg, field.name)
            
            return func(*args, **kwargs)

        return inner
    
    return merge_conf_with_cls