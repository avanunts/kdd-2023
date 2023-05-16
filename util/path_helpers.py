import builtins


def add_prefix(prefix, option):
    if option[0] == '=':
        return prefix + option
    return '{}.{}'.format(prefix, option)


def flat_dict(d):
    options = []
    match type(d):
        case builtins.dict:
            for k, v in d.items():
                inner_options = flat_dict(v)
                options += map(lambda x: add_prefix(k, x), inner_options)
            return options
        case builtins.str:
            return ['={}'.format(d)]
        case builtins.int:
            return ['={}'.format(d)]
        case builtins.float:
            return ['={}'.format(d)]
        case other:
            raise NotImplementedError('type {} is not supported'.format(other))


def config_path(config):
    return '_'.join(flat_dict(config))
