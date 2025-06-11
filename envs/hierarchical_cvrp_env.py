import torch

class HierarchicalCVRPEnv:
    """
    A fully-batched wrapper for a CVRPEnv that presents a view of a subgraph to the agent.
    It translates actions on the subgraph to actions on the base environment using tensor operations.
    """
    def __init__(self, base_env, node_indices):
        self.base_env = base_env
        self.device = self.base_env.device
        
        # node_indices are the ORIGINAL indices of the nodes in this subgraph
        # Shape: (batch_size, num_nodes_sub)
        self.node_indices = node_indices.to(self.device)
        self.batch_size, self.num_nodes_sub = self.node_indices.size()
        self.num_nodes_base = self.base_env.num_nodes

        # --- FIX: Replace dictionary maps with tensor-based mapping ---
        # `base_to_sub_map`: A tensor where the value at (b, base_idx) is the corresponding sub_idx, or -1 if not in the subgraph.
        self.base_to_sub_map = torch.full((self.batch_size, self.num_nodes_base), -1, dtype=torch.long, device=self.device)
        sub_indices_range = torch.arange(self.num_nodes_sub, device=self.device).expand(self.batch_size, -1)
        self.base_to_sub_map.scatter_(1, self.node_indices, sub_indices_range)

    def get_state(self):
        """
        Returns the state of the environment, but filtered for the current subgraph.
        This is now a fully-batched operation.
        """
        base_state = self.base_env.get_state()
        
        # --- Filter static features for the subgraph nodes ---
        # The depot (node 0) is always implicitly the first node in any subgraph representation.
        # However, the node_indices already correctly contain the depot from the GPool fix.
        sub_static_coords = torch.gather(base_state.static.coords, 1, self.node_indices.unsqueeze(-1).expand(-1, -1, 2))
        sub_static_demands = torch.gather(base_state.static.demands, 1, self.node_indices)
        
        # --- Map dynamic state to the subgraph ---
        current_node_base = base_state.dynamic.current_node
        # Use the map to find the subgraph index of the current node
        current_node_sub = torch.gather(self.base_to_sub_map, 1, current_node_base).long()

        # The visited mask for the subgraph
        sub_visited_mask = torch.gather(base_state.dynamic.visited_mask, 1, self.node_indices)

        # Create a new state tuple for the subgraph
        sub_dynamic = base_state.dynamic._replace(
            visited_mask=sub_visited_mask,
            current_node=current_node_sub
        )
        sub_static = base_state.static._replace(coords=sub_static_coords, demands=sub_static_demands)
        
        return base_state._replace(static=sub_static, dynamic=sub_dynamic)

    def step(self, sub_action):
        """
        Takes a batched action corresponding to indices in the subgraph and maps it to the base environment.
        sub_action shape: (batch_size)
        """
        # --- Map the subgraph action to the base action using node_indices as the map ---
        # sub_action contains the index within the subgraph (e.g., 5 out of 100)
        # We gather from node_indices to get the original index (e.g., 87 out of 1000)
        base_action = torch.gather(self.node_indices, 1, sub_action.unsqueeze(1))
        
        # Step the base environment with the translated action
        _, reward, done = self.base_env.step(base_action.squeeze(1))
        
        # Return the new subgraph state
        return self.get_state(), reward, done