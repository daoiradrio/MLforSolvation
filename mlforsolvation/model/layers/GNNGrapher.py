from torch_geometric.transforms import RadiusGraph
from torch.nn import PairwiseDistance



class GNN_Grapher:

    def __init__(self, radius, max_num_neighbors) -> None:
        self._distancer = PairwiseDistance()
        self._gnn_grapher = RadiusGraph(r=radius, loop=False, max_num_neighbors=max_num_neighbors)


    def build_gnn_graph(self, data):

        # Get Radius Graph
        graph = self._gnn_grapher(data)

        # Extract edge index
        edge_index = graph.edge_index

        # Extract edge features
        distances = self._distancer(data.pos[edge_index[0]], data.pos[edge_index[1]])

        # For GBNeck model distances are features
        edge_attributes = distances.unsqueeze(1)

        return edge_index, edge_attributes