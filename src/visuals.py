import matplotlib.pyplot as plt

# --- Utility functions ---
def max_depth(node):
    """Returns the maximum depth of the tree

    Args:
        node (dict): The root node of the tree

    Returns:
        int: The maximum depth of the tree
    """

    if node is None or node.get("leaf", False):
        return 1
    
    return 1 + max(max_depth(node.get("left")), max_depth(node.get("right")))

def count_leaves(node):
    """Count the number of leaf nodes in the tree

    Args:
        node (dict): The root node of the tree

    Returns:
        int: The number of leaf nodes in the tree
    """

    if node is None:
        return 0
    if node.get("leaf", False):
        return 1
    
    return count_leaves(node.get("left")) + count_leaves(node.get("right"))

# --- Assign x positions to leaves first ---
def assign_x_positions(node, x_start=0, leaf_positions=None):
    """Traverse the tree and assign x-coordinates to all nodes.
    Leaf nodes get consecutive integers; internal nodes are centered above their children.

    Args:
        node (dict): The node of the tree
        x_start (int): The starting x-coordinate for leaves
        leaf_positions (dict): Dictionary to store x-positions of nodes

    Returns:
        (int, dict): The next x-coordinate and the updated leaf_positions
    """
    if leaf_positions is None:
        leaf_positions = {}

    if node.get("leaf", False):
        leaf_positions[id(node)] = x_start
        return x_start + 1, leaf_positions

    x_left, leaf_positions = assign_x_positions(node.get("left"), x_start, leaf_positions)
    x_right, leaf_positions = assign_x_positions(node.get("right"), x_left, leaf_positions)

    # Center internal node above its children
    leaf_positions[id(node)] = (leaf_positions[id(node["left"])] + leaf_positions[id(node["right"])]) / 2
    return x_right, leaf_positions

# --- Recursive plotting ---
def plot_tree(node, x=None, y=1.0, dy=None, ax=None, depth=0, max_vis_depth=None, stop_depth=4, leaf_positions=None):
    """Recursively plot the tree using matplotlib.
    Each node is drawn as a rectangle, with lines connecting to children.

    Args:
        node (dict): The current node of the tree
        x (float): The x-coordinate of the current node
        y (float): The y-coordinate of the current node
        dy (float): The vertical spacing between levels
        ax (matplotlib.axes.Axes): The axes to plot on
        depth (int): The current depth in the tree
        max_vis_depth (int): The maximum depth for visualization
        stop_depth (int): The depth at which to stop plotting further
        leaf_positions (dict): Precomputed x-positions for nodes

    Returns:
        matplotlib.axes.Axes: The axes with the plotted tree
    """
    if ax is None:
        # First call: Initialize figure and compute tree metrics
        max_vis_depth = max_depth(node)
        _, leaf_positions = assign_x_positions(node)

        fig_width = 80
        fig_height = 30
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_axis_off()

        # Vertical spacing between levels
        dy = 0.9 / max_vis_depth

    if x is None:
        x = leaf_positions[id(node)]

    # --- Determine label and color ---
    if node.get("leaf", False):
        label = f"Leaf\nRoom {int(node['prediction'])}"
        color = "lightgreen"
    else:
        label = f"f{node['attribute']} <= {node['value']:.2f}"
        color = "burlywood"

    # --- Draw rectangle for this node ---
    font_size = 25
    box_width = 18  # scaled with font size
    box_height = 0.01   # scaled with font size
    ax.add_patch(plt.Rectangle(
        (x - box_width / 2, y - box_height / 2),
        box_width, box_height,
        edgecolor='black', facecolor=color
    ))
    ax.text(x, y, label, ha='center', va='center', fontsize=font_size)

    # Stop recursion if leaf
    if node.get("leaf", False):
        return ax
    
    if stop_depth is not None and depth >= stop_depth:
        return ax
    
    # --- Plot children ---
    child_y = y - dy
    for child in ["left", "right"]:
        if node.get(child):
            child_x = leaf_positions[id(node[child])]
            ax.plot([x, child_x], [y - box_height / 2, child_y + box_height / 2], 'k-', lw=0.6)
            plot_tree(node[child], child_x, child_y, dy, ax, depth + 1, max_vis_depth, stop_depth, leaf_positions)

    return ax
