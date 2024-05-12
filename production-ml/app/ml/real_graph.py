import dgl

def build_factory_graph(data):

    g = dgl.heterograph({
        ('stage', 'stage_to_input', 'input'): ([], []),
        ('input', 'input_to_output', 'output'): ([], []),
        ('output', 'output_to_stage', 'stage'): ([], []),
        ('output', 'output_to_input', 'input'): ([], []),
    })

    g.add_nodes(data['num_stages'], ntype='input')
    
    return g