import numba


@numba.njit(inline="always")
def _cs(a, b):
    """
    Performs a compare-swap on two values.

    Args:
        a: First value.
        b: Second value.

    Returns:
        Tuple containing the smaller value followed by the larger value.
    """
    if a > b:
        return b, a
    else:
        return a, b


@numba.njit(inline="always")
def sort6_network(v0, v1, v2, v3, v4, v5):
    """
    Sorts 6 elements using a pre-defined sorting network.

    This function implements a sorting network from https://bertdobbelaere.github.io/sorting_networks.html
    to efficiently sort a fixed number of elements.

    Args:
        v0-v5: The 6 values to be sorted.

    Returns:
        Tuple containing the 6 input values in sorted order (ascending).
    """
    v0, v5 = _cs(v0, v5)
    v1, v3 = _cs(v1, v3)
    v2, v4 = _cs(v2, v4)
    
    v1, v2 = _cs(v1, v2)
    v3, v4 = _cs(v3, v4)
    
    v0, v3 = _cs(v0, v3)
    v2, v5 = _cs(v2, v5)
    
    v0, v1 = _cs(v0, v1)
    v2, v3 = _cs(v2, v3)
    v4, v5 = _cs(v4, v5)
    
    v1, v2 = _cs(v1, v2)
    v3, v4 = _cs(v3, v4)
    
    return v0, v1, v2, v3, v4, v5


@numba.njit(inline="always")
def sort7_network(v0, v1, v2, v3, v4, v5, v6):
    """
    Sorts 7 elements using a pre-defined sorting network.

    This function implements a sorting network from https://bertdobbelaere.github.io/sorting_networks.html
    to efficiently sort a fixed number of elements.

    Args:
        v0-v6: The 7 values to be sorted.

    Returns:
        Tuple containing the 7 input values in sorted order (ascending).
    """
    v0, v6 = _cs(v0, v6)
    v2, v3 = _cs(v2, v3)
    v4, v5 = _cs(v4, v5)
    
    v0, v2 = _cs(v0, v2)
    v1, v4 = _cs(v1, v4)
    v3, v6 = _cs(v3, v6)
    
    v0, v1 = _cs(v0, v1)
    v2, v5 = _cs(v2, v5)
    v3, v4 = _cs(v3, v4)

    v1, v2 = _cs(v1, v2)
    v4, v6 = _cs(v4, v6)

    v2, v3 = _cs(v2, v3)
    v4, v5 = _cs(v4, v5)
    
    v1, v2 = _cs(v1, v2)
    v3, v4 = _cs(v3, v4)
    v5, v6 = _cs(v5, v6)
    
    return v0, v1, v2, v3, v4, v5, v6

@numba.njit(inline="always")
def sort26_network(
    v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13,
    v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25
):
    """
    Sorts 26 elements using a pre-defined sorting network.

    This function implements a sorting network from https://bertdobbelaere.github.io/sorting_networks.html
    to efficiently sort a fixed number of elements.

    Args:
        v0-v25: The 26 values to be sorted.

    Returns:
        Tuple containing the 26 input values in sorted order (ascending).
    """
    v0, v1 = _cs(v0, v1); v2, v3 = _cs(v2, v3); v4, v5 = _cs(v4, v5); v6, v7 = _cs(v6, v7); v8, v9 = _cs(v8, v9); v10, v11 = _cs(v10, v11); v12, v13 = _cs(v12, v13); v14, v15 = _cs(v14, v15); v16, v17 = _cs(v16, v17); v18, v19 = _cs(v18, v19); v20, v21 = _cs(v20, v21); v22, v23 = _cs(v22, v23); v24, v25 = _cs(v24, v25)
    v0, v2 = _cs(v0, v2); v1, v3 = _cs(v1, v3); v4, v6 = _cs(v4, v6); v5, v7 = _cs(v5, v7); v8, v10 = _cs(v8, v10); v9, v11 = _cs(v9, v11); v14, v16 = _cs(v14, v16); v15, v17 = _cs(v15, v17); v18, v20 = _cs(v18, v20); v19, v21 = _cs(v19, v21); v22, v24 = _cs(v22, v24); v23, v25 = _cs(v23, v25)
    v0, v4 = _cs(v0, v4); v1, v6 = _cs(v1, v6); v2, v5 = _cs(v2, v5); v3, v7 = _cs(v3, v7); v8, v14 = _cs(v8, v14); v9, v16 = _cs(v9, v16); v10, v15 = _cs(v10, v15); v11, v17 = _cs(v11, v17); v18, v22 = _cs(v18, v22); v19, v24 = _cs(v19, v24); v20, v23 = _cs(v20, v23); v21, v25 = _cs(v21, v25)
    v0, v18 = _cs(v0, v18); v1, v19 = _cs(v1, v19); v2, v20 = _cs(v2, v20); v3, v21 = _cs(v3, v21); v4, v22 = _cs(v4, v22); v5, v23 = _cs(v5, v23); v6, v24 = _cs(v6, v24); v7, v25 = _cs(v7, v25); v9, v12 = _cs(v9, v12); v13, v16 = _cs(v13, v16)
    v3, v11 = _cs(v3, v11); v8, v9 = _cs(v8, v9); v10, v13 = _cs(v10, v13); v12, v15 = _cs(v12, v15); v14, v22 = _cs(v14, v22); v16, v17 = _cs(v16, v17)
    v0, v8 = _cs(v0, v8); v1, v9 = _cs(v1, v9); v2, v14 = _cs(v2, v14); v6, v12 = _cs(v6, v12); v7, v15 = _cs(v7, v15); v10, v18 = _cs(v10, v18); v11, v23 = _cs(v11, v23); v13, v19 = _cs(v13, v19); v16, v24 = _cs(v16, v24); v17, v25 = _cs(v17, v25)
    v1, v2 = _cs(v1, v2); v3, v18 = _cs(v3, v18); v4, v8 = _cs(v4, v8); v7, v22 = _cs(v7, v22); v17, v21 = _cs(v17, v21); v23, v24 = _cs(v23, v24)
    v3, v14 = _cs(v3, v14); v4, v10 = _cs(v4, v10); v5, v18 = _cs(v5, v18); v7, v20 = _cs(v7, v20); v8, v13 = _cs(v8, v13); v11, v22 = _cs(v11, v22); v12, v17 = _cs(v12, v17); v15, v21 = _cs(v15, v21)
    v1, v4 = _cs(v1, v4); v5, v6 = _cs(v5, v6); v7, v9 = _cs(v7, v9); v8, v10 = _cs(v8, v10); v15, v17 = _cs(v15, v17); v16, v18 = _cs(v16, v18); v19, v20 = _cs(v19, v20); v21, v24 = _cs(v21, v24)
    v2, v5 = _cs(v2, v5); v3, v10 = _cs(v3, v10); v6, v14 = _cs(v6, v14); v9, v13 = _cs(v9, v13); v11, v19 = _cs(v11, v19); v12, v16 = _cs(v12, v16); v15, v22 = _cs(v15, v22); v20, v23 = _cs(v20, v23)
    v2, v8 = _cs(v2, v8); v5, v7 = _cs(v5, v7); v6, v9 = _cs(v6, v9); v11, v12 = _cs(v11, v12); v13, v14 = _cs(v13, v14); v16, v19 = _cs(v16, v19); v17, v23 = _cs(v17, v23); v18, v20 = _cs(v18, v20)
    v2, v4 = _cs(v2, v4); v3, v5 = _cs(v3, v5); v6, v11 = _cs(v6, v11); v7, v10 = _cs(v7, v10); v9, v16 = _cs(v9, v16); v12, v13 = _cs(v12, v13); v14, v19 = _cs(v14, v19); v15, v18 = _cs(v15, v18); v20, v22 = _cs(v20, v22); v21, v23 = _cs(v21, v23)
    v3, v4 = _cs(v3, v4); v5, v8 = _cs(v5, v8); v6, v7 = _cs(v6, v7); v9, v11 = _cs(v9, v11); v10, v12 = _cs(v10, v12); v13, v15 = _cs(v13, v15); v14, v16 = _cs(v14, v16); v17, v20 = _cs(v17, v20); v18, v19 = _cs(v18, v19); v21, v22 = _cs(v21, v22)
    v5, v6 = _cs(v5, v6); v7, v8 = _cs(v7, v8); v9, v10 = _cs(v9, v10); v11, v12 = _cs(v11, v12); v13, v14 = _cs(v13, v14); v15, v16 = _cs(v15, v16); v17, v18 = _cs(v17, v18); v19, v20 = _cs(v19, v20)
    v4, v5 = _cs(v4, v5); v6, v7 = _cs(v6, v7); v8, v9 = _cs(v8, v9); v10, v11 = _cs(v10, v11); v12, v13 = _cs(v12, v13); v14, v15 = _cs(v14, v15); v16, v17 = _cs(v16, v17); v18, v19 = _cs(v18, v19); v20, v21 = _cs(v20, v21)
    
    return v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25




@numba.njit(inline="always")
def sort27_network(
    v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13,
    v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26
):
    """
    Sorts 27 elements using a pre-defined sorting network.

    This function implements a sorting network from https://bertdobbelaere.github.io/sorting_networks.html
    to efficiently sort a fixed number of elements.

    Args:
        v0-v26: The 27 values to be sorted.

    Returns:
        Tuple containing the 27 input values in sorted order (ascending).
    """
    v0, v19 = _cs(v0, v19); v1, v26 = _cs(v1, v26); v2, v25 = _cs(v2, v25); v3, v24 = _cs(v3, v24); v4, v23 = _cs(v4, v23); v5, v22 = _cs(v5, v22); v6, v21 = _cs(v6, v21); v7, v20 = _cs(v7, v20); v9, v10 = _cs(v9, v10); v11, v17 = _cs(v11, v17); v12, v15 = _cs(v12, v15); v13, v14 = _cs(v13, v14); v16, v18 = _cs(v16, v18)
    v0, v1 = _cs(v0, v1); v2, v3 = _cs(v2, v3); v4, v5 = _cs(v4, v5); v6, v7 = _cs(v6, v7); v8, v9 = _cs(v8, v9); v11, v16 = _cs(v11, v16); v12, v14 = _cs(v12, v14); v13, v15 = _cs(v13, v15); v17, v18 = _cs(v17, v18); v19, v26 = _cs(v19, v26); v20, v21 = _cs(v20, v21); v22, v23 = _cs(v22, v23); v24, v25 = _cs(v24, v25)
    v0, v2 = _cs(v0, v2); v1, v3 = _cs(v1, v3); v4, v6 = _cs(v4, v6); v5, v7 = _cs(v5, v7); v8, v18 = _cs(v8, v18); v9, v14 = _cs(v9, v14); v10, v12 = _cs(v10, v12); v13, v17 = _cs(v13, v17); v15, v16 = _cs(v15, v16); v19, v24 = _cs(v19, v24); v20, v22 = _cs(v20, v22); v21, v23 = _cs(v21, v23); v25, v26 = _cs(v25, v26)
    v0, v4 = _cs(v0, v4); v1, v5 = _cs(v1, v5); v2, v20 = _cs(v2, v20); v3, v21 = _cs(v3, v21); v6, v19 = _cs(v6, v19); v7, v25 = _cs(v7, v25); v8, v13 = _cs(v8, v13); v9, v17 = _cs(v9, v17); v10, v11 = _cs(v10, v11); v12, v15 = _cs(v12, v15); v14, v18 = _cs(v14, v18); v22, v24 = _cs(v22, v24); v23, v26 = _cs(v23, v26)
    v1, v2 = _cs(v1, v2); v3, v19 = _cs(v3, v19); v4, v6 = _cs(v4, v6); v5, v22 = _cs(v5, v22); v7, v20 = _cs(v7, v20); v8, v10 = _cs(v8, v10); v9, v12 = _cs(v9, v12); v11, v13 = _cs(v11, v13); v14, v16 = _cs(v14, v16); v15, v17 = _cs(v15, v17); v21, v23 = _cs(v21, v23); v24, v25 = _cs(v24, v25)
    v0, v8 = _cs(v0, v8); v1, v4 = _cs(v1, v4); v2, v6 = _cs(v2, v6); v3, v10 = _cs(v3, v10); v5, v7 = _cs(v5, v7); v9, v11 = _cs(v9, v11); v12, v13 = _cs(v12, v13); v14, v15 = _cs(v14, v15); v16, v17 = _cs(v16, v17); v18, v19 = _cs(v18, v19); v20, v22 = _cs(v20, v22); v21, v24 = _cs(v21, v24); v23, v25 = _cs(v23, v25)
    v1, v9 = _cs(v1, v9); v2, v13 = _cs(v2, v13); v4, v8 = _cs(v4, v8); v5, v12 = _cs(v5, v12); v6, v10 = _cs(v6, v10); v7, v20 = _cs(v7, v20); v14, v24 = _cs(v14, v24); v15, v22 = _cs(v15, v22); v17, v25 = _cs(v17, v25); v18, v21 = _cs(v18, v21); v23, v26 = _cs(v23, v26)
    v3, v4 = _cs(v3, v4); v6, v14 = _cs(v6, v14); v7, v11 = _cs(v7, v11); v8, v15 = _cs(v8, v15); v9, v18 = _cs(v9, v18); v10, v17 = _cs(v10, v17); v12, v23 = _cs(v12, v23); v13, v21 = _cs(v13, v21); v16, v20 = _cs(v16, v20); v19, v26 = _cs(v19, v26)
    v1, v3 = _cs(v1, v3); v2, v4 = _cs(v2, v4); v5, v6 = _cs(v5, v6); v7, v8 = _cs(v7, v8); v10, v13 = _cs(v10, v13); v11, v15 = _cs(v11, v15); v12, v16 = _cs(v12, v16); v14, v18 = _cs(v14, v18); v19, v24 = _cs(v19, v24); v20, v23 = _cs(v20, v23); v21, v22 = _cs(v21, v22); v25, v26 = _cs(v25, v26)
    v2, v7 = _cs(v2, v7); v4, v8 = _cs(v4, v8); v6, v9 = _cs(v6, v9); v10, v11 = _cs(v10, v11); v12, v14 = _cs(v12, v14); v13, v15 = _cs(v13, v15); v16, v18 = _cs(v16, v18); v17, v21 = _cs(v17, v21); v19, v20 = _cs(v19, v20); v23, v24 = _cs(v23, v24)
    v2, v3 = _cs(v2, v3); v4, v7 = _cs(v4, v7); v5, v6 = _cs(v5, v6); v8, v10 = _cs(v8, v10); v9, v12 = _cs(v9, v12); v11, v16 = _cs(v11, v16); v13, v14 = _cs(v13, v14); v15, v17 = _cs(v15, v17); v18, v19 = _cs(v18, v19); v20, v23 = _cs(v20, v23); v21, v22 = _cs(v21, v22); v24, v25 = _cs(v24, v25)
    v4, v5 = _cs(v4, v5); v6, v7 = _cs(v6, v7); v8, v9 = _cs(v8, v9); v10, v12 = _cs(v10, v12); v11, v13 = _cs(v11, v13); v14, v16 = _cs(v14, v16); v15, v18 = _cs(v15, v18); v17, v19 = _cs(v17, v19); v20, v21 = _cs(v20, v21); v22, v23 = _cs(v22, v23)
    v3, v4 = _cs(v3, v4); v5, v6 = _cs(v5, v6); v7, v8 = _cs(v7, v8); v9, v10 = _cs(v9, v10); v11, v12 = _cs(v11, v12); v13, v14 = _cs(v13, v14); v15, v16 = _cs(v15, v16); v17, v18 = _cs(v17, v18); v19, v20 = _cs(v19, v20); v21, v22 = _cs(v21, v22); v23, v24 = _cs(v23, v24)
    
    return v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26