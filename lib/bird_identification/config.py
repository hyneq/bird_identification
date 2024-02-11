from typing import Optional, Callable
from dataclasses import fields, is_dataclass

from dacite import from_dict

from .defaults.config import DEFAULT_CONFIG_FILE_LOADER

def merge_conf(cfg_cls: type):

    if not is_dataclass(cfg_cls):
        raise TypeError("Config class must be a dataclass")
    
    cfg_fields = fields(cfg_cls)

    def merge_conf_with_cls(func: Callable) -> Callable:

        def inner(*args, cfg: Optional[cfg_cls]=None, **kwargs):
            if cfg:
                for field in cfg_fields:
                    if kwargs.get(field.name, None) is None:
                        kwargs[field.name] = getattr(cfg, field.name)
            
            return func(*args, **kwargs)

        return inner
    
    return merge_conf_with_cls

def conf_from_file(cfg_cls: type, loader=None):

    def conf_from_file_with_cls(func: Callable) -> Callable:

        def inner(*args, cfg_path: Optional[str]=None, cfg_loader=loader, **kwargs):
            if cfg_path:
                kwargs['cfg'] = load_conf_from_file(path=cfg_path, cfg_cls=cfg_cls, loader=cfg_loader)
            
            return func(*args, **kwargs)
        
        return inner
    
    return conf_from_file_with_cls

def load_conf_from_file(path: str, cfg_cls: type, loader=None):
    
    if not loader:
        loader = DEFAULT_CONFIG_FILE_LOADER

    with open(path) as f:
        cfg_dict = loader(f)
    
    return from_dict(data_class=cfg_cls, data=cfg_dict)