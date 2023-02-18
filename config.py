from typing import Optional
from dataclasses import is_dataclass

def merge_conf(func: function) -> function:

    def inner(*args, cfg: Optional[type]=None, **kwargs):
        if is_dataclass(cfg):
            for attr, val in cfg.__dict__.items():
                if kwargs.get(attr, None) is None:
                    kwargs[attr] = val
        
        func(*args, **kwargs)


    return inner