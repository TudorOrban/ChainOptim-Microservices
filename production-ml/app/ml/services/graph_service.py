
from collections import deque
from typing import List
from app.types.factory_graph import FactoryGraph

"""
Graph utils used to (classically) determine the optimal distribution of resources.
"""
def determine_dependency_subtree(factory_graph: FactoryGraph, stage_id: int) -> FactoryGraph:
    relevant_nodes = set()
    relevant_edges = set()

    def traverse_graph(current_stage_id: int):
        if current_stage_id in relevant_nodes:
            return
        relevant_nodes.add(current_stage_id)

        # Check all edges to see if the current stage is a destination in any edge
        for stage, edges in factory_graph.adj_list.items():
            for edge in edges:
                if edge.outgoing_factory_stage_id == current_stage_id:
                    relevant_edges.add(edge)
                    # Recurse through the graph from the source stage of this edge
                    traverse_graph(edge.incoming_factory_stage_id)

    traverse_graph(stage_id)

    filtered_nodes = {node_id: factory_graph.nodes[node_id] for node_id in relevant_nodes}
    filtered_adj_list = {node_id: [edge for edge in factory_graph.adj_list.get(node_id, []) if edge in relevant_edges] 
                         for node_id in relevant_nodes}
    
    return FactoryGraph(nodes=filtered_nodes, adj_list=filtered_adj_list)


def topological_sort(factory_graph: FactoryGraph) -> List[int]:
    # Step 1: Compute in-degrees
    in_degree = {node_id: 0 for node_id in factory_graph.nodes}
    for node in factory_graph.nodes:
        for edge in factory_graph.adj_list.get(node, []):
            in_degree[edge.outgoing_factory_stage_id] += 1

    # Step 2: Initialize queue
    zero_in_degree_queue = deque([node for node in factory_graph.nodes if in_degree[node] == 0])

    # Step 3: Topological sort
    topological_order = []

    while zero_in_degree_queue:
        node = zero_in_degree_queue.popleft()
        topological_order.append(node)

        # Update in-degrees
        for edge in factory_graph.adj_list.get(node, []):
            neighbor = edge.outgoing_factory_stage_id
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                zero_in_degree_queue.append(neighbor)

    # Make sure there is no cycle
    if len(topological_order) != len(factory_graph.nodes):
        raise ValueError("Cycle detected in the graph")
    
    return topological_order
