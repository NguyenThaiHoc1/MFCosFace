def generate_layer_name(name, branch_idx=None, prefix=None):
    if prefix is None:
        return None
    if branch_idx is None:
        return '_'.join((prefix, name))
    return '_'.join((prefix, 'Branch', str(branch_idx), name))


def get_layer_byname(model, name):
    split_name = name.split('/')
    if '' in split_name:
        return model
    layer_name = split_name[0]
    part_name = '/'.join(name.split('/')[1:])
    return get_layer_byname(model.get_layer(layer_name), part_name)
