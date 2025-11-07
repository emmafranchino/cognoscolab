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


def verify_and_report_nodes(fmns, required, case_insensitive=True, strict=True):
    """
    Minimal unified verification:
      - Checks required nodes per FMN
      - Prints missing nodes per network
      - Prints global summary + per-node coverage
      - No DataFrame creation

    Parameters
    ----------
    fmns : dict[str, FormamentisNetwork] or list[FormamentisNetwork]
    required : iterable[str]
    case_insensitive : bool
    strict : bool
        True -> network must contain ALL required nodes
        False -> network must contain ANY required node

    Returns
    -------
    summary : dict
    """
    if not required:
        raise ValueError("required must be a non-empty iterable")
    required = list(required)

    # Print header
    print(f"Presence of nodes {required} in {'all' if strict else 'any'} FMNs:")

    from collections import Counter
    present_counter = Counter()

    def check_fmnt(fmnt):
        nodes = getattr(fmnt, "vertices", [])
        if case_insensitive:
            lower_map = {n.lower(): n for n in nodes}
            present = [lower_map[r.lower()] for r in required if r.lower() in lower_map]
            missing = [r for r in required if r.lower() not in lower_map]
        else:
            node_set = set(nodes)
            present = [r for r in required if r in node_set]
            missing = [r for r in required if r not in node_set]
        if strict:
            result = len(missing) == 0
            mode = "all"
        else:
            result = len(present) > 0
            mode = "any"
        return {
            "result": result,
            "mode": mode,
            "present": present,
            "missing": missing,
            "present_count": len(present),
            "missing_count": len(missing),
            "required_total": len(required),
            "network_total": len(nodes)
        }

    results = {}
    missing_fmnts = {}

    if not isinstance(fmns, dict):
        fmns = {f"fmnt_{i+1}": fmnt for i, fmnt in enumerate(fmns)}

    for title, fmnt in fmns.items():
        res = check_fmnt(fmnt)
        results[title] = res
        # Update per-node presence
        present_req_forms = {r.lower() for r in required} if case_insensitive else set(required)
        for r in required:
            if (r.lower() in {p.lower() for p in res["present"]}) if case_insensitive else (r in res["present"]):
                present_counter[r] += 1
        if not res["result"]:
            missing_fmnts[title] = res["missing"]
            print(f"FMN '{title}' missing: {res['missing']}")

    total = len(fmns)
    satisfying = sum(1 for r in results.values() if r["result"])
    percentage = (satisfying / total * 100) if total else 0.0

    print(f"\nNetworks satisfying ({'ALL' if strict else 'ANY'} of {required}): {satisfying}/{total} ({percentage:.2f}%)")
    for node in required:
        count = present_counter[node]
        print(f"Presence '{node}': {count}/{total} ({(count/total*100 if total else 0):.2f}%)\n")

    summary = {
        "required": required,
        "mode": "all" if strict else "any",
        "total_fmns": total,
        "fmns_with_requirement": satisfying,
        "percentage": percentage,
        "per_node_presence": {
            node: {
                "count": present_counter[node],
                "percentage": (present_counter[node] / total * 100) if total else 0.0
            } for node in required
        },
        "missing_fmnts": missing_fmnts,
        "results": results
    }
    return summary
