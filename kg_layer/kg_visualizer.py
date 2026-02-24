"""
Knowledge Graph Visualizer for Hetionet

Provides visualization tools for exploring the knowledge graph structure,
especially focusing on compounds and their relationships to diseases.
"""

import os
import logging
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from kg_layer.kg_loader import load_hetionet_edges, extract_task_edges
from kg_layer.advanced_embeddings import AdvancedKGEmbedder

logger = logging.getLogger(__name__)


class KGVisualizer:
    """
    Visualizer for Hetionet knowledge graph with focus on compound-disease relationships.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.G = None
        self.embeddings = None
        self.entity_to_id = {}
        self.id_to_entity = {}

    def load_kg(self, relation_type: str = "CtD", max_entities: Optional[int] = None):
        """
        Load the knowledge graph for visualization.

        Args:
            relation_type: The relation type to focus on (e.g., "CtD", "DaG")
            max_entities: Maximum number of entities to include (for performance)
        """
        logger.info(f"Loading knowledge graph for relation: {relation_type}")
        
        # Load Hetionet edges
        df_edges = load_hetionet_edges(data_dir=self.data_dir)
        
        # Extract task-specific edges
        task_edges, self.entity_to_id, self.id_to_entity = extract_task_edges(
            df_edges, 
            relation_type=relation_type, 
            max_entities=max_entities
        )
        
        # Create NetworkX graph
        self.G = nx.Graph()
        
        # Add nodes with attributes
        for entity_id, entity_idx in self.entity_to_id.items():
            entity_type = entity_id.split("::")[0] if "::" in entity_id else "Unknown"
            self.G.add_node(
                entity_idx, 
                entity_id=entity_id, 
                entity_type=entity_type,
                label=entity_id.split("::")[-1]  # Just the ID part
            )
        
        # Add edges
        for _, row in task_edges.iterrows():
            self.G.add_edge(
                row["source_id"], 
                row["target_id"], 
                relation=row["metaedge"],
                weight=1.0
            )
        
        logger.info(f"Created graph with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")

    def load_embeddings(self, embedding_method: str = "ComplEx", embedding_dim: int = 64):
        """
        Load pre-trained embeddings for visualization.

        Args:
            embedding_method: The embedding method used (e.g., "ComplEx", "RotatE")
            embedding_dim: The embedding dimension
        """
        logger.info(f"Loading {embedding_method} embeddings (dim={embedding_dim})")
        
        embedder = AdvancedKGEmbedder(
            embedding_dim=embedding_dim,
            method=embedding_method,
            work_dir=self.data_dir
        )
        
        if embedder.load_embeddings():
            self.embeddings = embedder.entity_embeddings
            self.entity_to_id = embedder.entity_to_id
            self.id_to_entity = embedder.id_to_entity
            logger.info(f"Loaded embeddings for {len(self.entity_to_id)} entities")
        else:
            logger.warning("Could not load embeddings. Visualization will not include embedding-based layouts.")

    def visualize_graph_structure(self, layout: str = "spring", figsize: Tuple[int, int] = (12, 8)):
        """
        Visualize the overall structure of the knowledge graph.

        Args:
            layout: Layout algorithm ("spring", "circular", "random", "kamada_kawai", "spectral")
            figsize: Figure size as (width, height)
        """
        if self.G is None:
            raise ValueError("Knowledge graph not loaded. Call load_kg() first.")

        plt.figure(figsize=figsize)
        
        # Choose layout
        if layout == "spring":
            pos = nx.spring_layout(self.G, k=1, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(self.G)
        elif layout == "random":
            pos = nx.random_layout(self.G)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(self.G)
        elif layout == "spectral":
            pos = nx.spectral_layout(self.G)
        else:
            pos = nx.spring_layout(self.G)
        
        # Get node colors based on entity type
        node_colors = []
        for node in self.G.nodes():
            entity_type = self.G.nodes[node].get('entity_type', 'Unknown')
            if entity_type == 'Compound':
                node_colors.append('red')
            elif entity_type == 'Disease':
                node_colors.append('blue')
            elif entity_type == 'Gene':
                node_colors.append('green')
            elif entity_type == 'Anatomy':
                node_colors.append('orange')
            else:
                node_colors.append('gray')
        
        # Draw the graph
        nx.draw(
            self.G, 
            pos=pos,
            node_color=node_colors,
            node_size=50,
            edge_color='lightgray',
            alpha=0.7,
            with_labels=False
        )
        
        # Create legend
        legend_elements = [
            Patch(facecolor='red', label='Compounds'),
            Patch(facecolor='blue', label='Diseases'),
            Patch(facecolor='green', label='Genes'),
            Patch(facecolor='orange', label='Anatomy'),
            Patch(facecolor='gray', label='Other')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.title(f"Knowledge Graph Structure ({layout} layout)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def visualize_compound_network(self, compound_id: str, radius: int = 2, figsize: Tuple[int, int] = (12, 10)):
        """
        Visualize the neighborhood of a specific compound in the knowledge graph.

        Args:
            compound_id: The compound ID to focus on (e.g., "Compound::DB00001")
            radius: Radius of neighbors to include in the subgraph
            figsize: Figure size as (width, height)
        """
        if self.G is None:
            raise ValueError("Knowledge graph not loaded. Call load_kg() first.")
        
        if compound_id not in self.entity_to_id:
            available_compounds = [eid for eid in self.entity_to_id.keys() if eid.startswith("Compound::")]
            raise ValueError(f"Compound {compound_id} not found in graph. Available compounds: {available_compounds[:10]}...")
        
        center_node = self.entity_to_id[compound_id]
        
        # Get subgraph around the compound
        subgraph_nodes = nx.ego_graph(self.G, center_node, radius=radius).nodes()
        subgraph = self.G.subgraph(subgraph_nodes)
        
        plt.figure(figsize=figsize)
        
        # Position nodes using spring layout
        pos = nx.spring_layout(subgraph, k=1.5, iterations=100)
        
        # Color nodes by type
        node_colors = []
        node_sizes = []
        for node in subgraph.nodes():
            entity_type = subgraph.nodes[node].get('entity_type', 'Unknown')
            if entity_type == 'Compound':
                node_colors.append('red')
                node_sizes.append(300)
            elif entity_type == 'Disease':
                node_colors.append('blue')
                node_sizes.append(200)
            elif entity_type == 'Gene':
                node_colors.append('green')
                node_sizes.append(150)
            elif entity_type == 'Anatomy':
                node_colors.append('orange')
                node_sizes.append(150)
            else:
                node_colors.append('gray')
                node_sizes.append(100)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            subgraph, 
            pos=pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.8
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            subgraph, 
            pos=pos,
            edge_color='lightgray',
            alpha=0.5,
            width=0.5
        )
        
        # Draw labels for the center compound and diseases
        labels = {}
        for node in subgraph.nodes():
            entity_id = subgraph.nodes[node]['entity_id']
            entity_type = subgraph.nodes[node]['entity_type']
            if entity_id == compound_id or entity_type == 'Disease':
                labels[node] = entity_id.split("::")[-1][:10]  # Shortened label
        
        nx.draw_networkx_labels(
            subgraph, 
            pos=pos,
            labels=labels,
            font_size=8,
            font_weight='bold'
        )
        
        # Create legend
        legend_elements = [
            Patch(facecolor='red', label='Center Compound'),
            Patch(facecolor='blue', label='Diseases'),
            Patch(facecolor='green', label='Genes'),
            Patch(facecolor='orange', label='Anatomy'),
            Patch(facecolor='gray', label='Other')
        ]
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.title(f"Network around {compound_id} (radius={radius})")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def visualize_embeddings_2d(self, method: str = "tsne", figsize: Tuple[int, int] = (12, 10)):
        """
        Visualize embeddings in 2D using t-SNE or PCA.

        Args:
            method: Dimensionality reduction method ("tsne" or "pca")
            figsize: Figure size as (width, height)
        """
        if self.embeddings is None:
            raise ValueError("Embeddings not loaded. Call load_embeddings() first.")
        
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        
        # Reduce to 2D
        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.embeddings)-1))
        elif method == "pca":
            reducer = PCA(n_components=2, random_state=42)
        else:
            raise ValueError("Method must be 'tsne' or 'pca'")
        
        embeddings_2d = reducer.fit_transform(self.embeddings)
        
        # Create DataFrame for plotting
        df_plot = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'entity_id': list(self.id_to_entity.values()),
            'entity_type': [eid.split("::")[0] for eid in self.id_to_entity.values()]
        })
        
        # Add compound/disease specific info
        df_plot['is_compound'] = df_plot['entity_id'].str.startswith('Compound::')
        df_plot['is_disease'] = df_plot['entity_id'].str.startswith('Disease::')
        
        plt.figure(figsize=figsize)
        
        # Plot compounds and diseases separately for better visualization
        compounds = df_plot[df_plot['is_compound']]
        diseases = df_plot[df_plot['is_disease']]
        others = df_plot[~(df_plot['is_compound'] | df_plot['is_disease'])]
        
        if not compounds.empty:
            plt.scatter(compounds['x'], compounds['y'], c='red', label='Compounds', alpha=0.6, s=20)
        if not diseases.empty:
            plt.scatter(diseases['x'], diseases['y'], c='blue', label='Diseases', alpha=0.6, s=20)
        if not others.empty:
            plt.scatter(others['x'], others['y'], c='gray', label='Others', alpha=0.6, s=10)
        
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.title(f'Embeddings Visualization ({method.upper()})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def visualize_compound_disease_relationships(self, top_k: int = 20, figsize: Tuple[int, int] = (14, 8)):
        """
        Visualize relationships between compounds and diseases.

        Args:
            top_k: Number of top compounds/diseases to visualize
            figsize: Figure size as (width, height)
        """
        if self.G is None:
            raise ValueError("Knowledge graph not loaded. Call load_kg() first.")
        
        # Get compound-disease relationships
        compound_disease_edges = []
        for u, v, data in self.G.edges(data=True):
            source_entity = self.G.nodes[u]['entity_id']
            target_entity = self.G.nodes[v]['entity_id']
            
            # Check if it's a compound-disease relationship
            if (source_entity.startswith('Compound::') and target_entity.startswith('Disease::')) or \
               (source_entity.startswith('Disease::') and target_entity.startswith('Compound::')):
                
                # Determine which is compound and which is disease
                if source_entity.startswith('Compound::'):
                    compound = source_entity
                    disease = target_entity
                else:
                    compound = target_entity
                    disease = source_entity
                
                compound_disease_edges.append({
                    'compound': compound,
                    'disease': disease,
                    'relation': data.get('relation', 'unknown')
                })
        
        if not compound_disease_edges:
            logger.warning("No compound-disease relationships found in the current graph.")
            return
        
        df_cd = pd.DataFrame(compound_disease_edges)
        
        # Count relationships per compound and disease
        compound_counts = df_cd['compound'].value_counts().head(top_k)
        disease_counts = df_cd['disease'].value_counts().head(top_k)
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot compound counts
        axes[0].barh(range(len(compound_counts)), compound_counts.values)
        axes[0].set_yticks(range(len(compound_counts)))
        axes[0].set_yticklabels([cid.split("::")[-1] for cid in compound_counts.index], fontsize=8)
        axes[0].set_xlabel('Number of Disease Relationships')
        axes[0].set_title(f'Top {top_k} Compounds by Disease Connections')
        axes[0].invert_yaxis()
        
        # Plot disease counts
        axes[1].barh(range(len(disease_counts)), disease_counts.values)
        axes[1].set_yticks(range(len(disease_counts)))
        axes[1].set_yticklabels([did.split("::")[-1] for did in disease_counts.index], fontsize=8)
        axes[1].set_xlabel('Number of Compound Relationships')
        axes[1].set_title(f'Top {top_k} Diseases by Compound Connections')
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        plt.show()

    def interactive_compound_explorer(self):
        """
        Create an interactive visualization using Plotly for exploring compound relationships.
        """
        if self.G is None:
            raise ValueError("Knowledge graph not loaded. Call load_kg() first.")
        
        # Prepare node data
        node_data = []
        for node in self.G.nodes():
            entity_id = self.G.nodes[node]['entity_id']
            entity_type = self.G.nodes[node]['entity_type']
            node_data.append({
                'node_id': node,
                'entity_id': entity_id,
                'entity_type': entity_type,
                'label': entity_id.split("::")[-1]
            })
        
        df_nodes = pd.DataFrame(node_data)
        
        # Prepare edge data
        edge_data = []
        for u, v, data in self.G.edges(data=True):
            edge_data.append({
                'source': u,
                'target': v,
                'relation': data.get('relation', 'unknown')
            })
        
        df_edges = pd.DataFrame(edge_data)
        
        # Create a force-directed graph
        fig = go.Figure(data=go.Scatter(
            x=[],
            y=[],
            mode='markers',
            marker=dict(size=10),
            text=[],
            hovertemplate='<b>%{text}</b><br>Type: %{customdata}<extra></extra>',
            customdata=[]
        ))
        
        # Use networkx layout for positions
        pos = nx.spring_layout(self.G, k=1, iterations=50)
        
        # Extract positions
        x_nodes = [pos[node][0] for node in self.G.nodes()]
        y_nodes = [pos[node][1] for node in self.G.nodes()]
        
        # Color nodes by type
        colors = []
        for node in self.G.nodes():
            entity_type = self.G.nodes[node]['entity_type']
            if entity_type == 'Compound':
                colors.append('red')
            elif entity_type == 'Disease':
                colors.append('blue')
            elif entity_type == 'Gene':
                colors.append('green')
            elif entity_type == 'Anatomy':
                colors.append('orange')
            else:
                colors.append('gray')
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=x_nodes,
            y=y_nodes,
            mode='markers+text',
            marker=dict(
                size=12,
                color=colors,
                line=dict(width=1, color='black')
            ),
            text=df_nodes['label'],
            textposition="middle center",
            hovertemplate='<b>%{text}</b><br>Type: %{customdata}<extra></extra>',
            customdata=df_nodes['entity_type'],
            name='Entities'
        ))
        
        # Add edges
        edge_x = []
        edge_y = []
        for edge in self.G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color='rgba(100,100,100,0.5)'),
            hoverinfo='skip',
            mode='lines',
            name='Relationships'
        ))
        
        fig.update_layout(
            title="Interactive Knowledge Graph Explorer",
            showlegend=True,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        fig.show()

    def get_compound_info(self, compound_id: str) -> Dict:
        """
        Get detailed information about a compound and its relationships.

        Args:
            compound_id: The compound ID to query

        Returns:
            Dictionary with compound information
        """
        if self.G is None:
            raise ValueError("Knowledge graph not loaded. Call load_kg() first.")
        
        if compound_id not in self.entity_to_id:
            return {"error": f"Compound {compound_id} not found in graph"}
        
        node_id = self.entity_to_id[compound_id]
        
        # Get neighbors
        neighbors = list(self.G.neighbors(node_id))
        
        info = {
            "compound_id": compound_id,
            "node_id": node_id,
            "degree": self.G.degree(node_id),
            "neighborhood": []
        }
        
        for neighbor_id in neighbors:
            neighbor_entity = self.G.nodes[neighbor_id]['entity_id']
            edge_data = self.G[node_id][neighbor_id]
            
            info["neighborhood"].append({
                "entity": neighbor_entity,
                "entity_type": self.G.nodes[neighbor_id]['entity_type'],
                "relation": edge_data.get('relation', 'unknown')
            })
        
        # Separate by type
        diseases = [n for n in info["neighborhood"] if n["entity_type"] == "Disease"]
        genes = [n for n in info["neighborhood"] if n["entity_type"] == "Gene"]
        others = [n for n in info["neighborhood"] if n["entity_type"] not in ["Disease", "Gene"]]
        
        info["related_diseases"] = diseases
        info["related_genes"] = genes
        info["other_relationships"] = others
        
        return info


def create_compound_disease_dashboard(data_dir: str = "data", relation_type: str = "CtD"):
    """
    Create a comprehensive dashboard for exploring compound-disease relationships.
    
    Args:
        data_dir: Directory containing the data
        relation_type: The relation type to focus on
    """
    visualizer = KGVisualizer(data_dir=data_dir)
    visualizer.load_kg(relation_type=relation_type)
    
    print("Knowledge Graph Visualization Dashboard")
    print("=" * 50)
    print(f"Loaded graph with {visualizer.G.number_of_nodes()} nodes and {visualizer.G.number_of_edges()} edges")
    
    # Show statistics
    node_types = [visualizer.G.nodes[n]['entity_type'] for n in visualizer.G.nodes()]
    type_counts = pd.Series(node_types).value_counts()
    print("\nEntity type distribution:")
    for entity_type, count in type_counts.items():
        print(f"  {entity_type}: {count}")
    
    # Show some example compounds
    compounds = [n for n in visualizer.G.nodes() 
                 if visualizer.G.nodes[n]['entity_type'] == 'Compound']
    print(f"\nSample compounds (first 5):")
    for i, comp_node in enumerate(compounds[:5]):
        comp_id = visualizer.G.nodes[comp_node]['entity_id']
        print(f"  {i+1}. {comp_id}")
    
    return visualizer