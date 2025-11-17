def pretty_print_fmn(fmn, title=None):
    """Pretty print a FormaMentisNetwork object.
    Works with both multiplex and single-layer FMNs.
    
    Parameters:
        fmn (FormaMentisNetwork): The FormaMentisNetwork object to print.
        title (str, optional): Title to use for the FMN.
    """
    # Check if fmn is multiplex or single-layer
    # Multiplex case. fmn.edges is a dict {layertype: [(u,v),...],...}
    # if multiplex or a list [(u,v),...] if net is single-layer.
    if isinstance(fmn.edges, dict):
        is_multiplex = True
    elif isinstance(fmn.edges, list):
        is_multiplex = False
    else:
        raise ValueError("fmn.edges must be either a dict (multiplex) or a list (single-layer).")
    # Print the edges
    print(f"FMN: {fmn.title if title is None else title}")
    if is_multiplex:
        for layer, layer_edges in fmn.edges.items():
            print(f"{layer}: {layer_edges}")
        print("")
    else:
        print(f"{fmn.edges}\n")