"""Registered operator inputs for the shared declarative planning lifecycle."""


def operator_inputs(request):
    config = request.workload.config.to_dict()
    name = config.get('provider')
    if name not in ('splitk_sum', 'moe_layer', 'gemm_rs'):
        raise ValueError(f'unknown graph provider {name!r}')
    allowed = {'provider', 'small'} if name == 'moe_layer' else {'provider'}
    if set(config) - allowed:
        raise ValueError('unsupported operator provider configuration')
    if request.runtime:
        raise ValueError('operator runtime is provider-derived; overrides are forbidden')
    bindings = request.bindings.to_dict()
    allowed_bindings = {'splitk_sum': {'n'}, 'moe_layer': {'B', 'seed'},
                        'gemm_rs': {'M', 'K', 'N'}}[name]
    if set(bindings) - allowed_bindings:
        raise ValueError('unsupported or provider-derived binding')
    size_key = 'n' if name == 'splitk_sum' else 'B'
    if size_key in bindings and bindings[size_key] < 1:
        raise ValueError('operator size must be positive')
    if name == 'splitk_sum':
        from etx.operators import splitk_sum as provider
    elif name == 'gemm_rs':
        from etx.operators import gemm_rs as provider
    else:
        from etx.operators.moe_layer import provider as create
        small = config.get('small', False)
        if type(small) is not bool:
            raise ValueError('small requires a boolean')
        provider = create(small=small)
    bindings = provider.bindings(**bindings)
    if name == 'gemm_rs':
        for key, alignment in (('M', 256), ('K', 32), ('N', 128)):
            if bindings[key] < alignment or bindings[key] % alignment:
                raise ValueError(f'GEMM {key} must be a positive multiple of {alignment}')
    return provider.build(), bindings, provider.runtime(bindings)
