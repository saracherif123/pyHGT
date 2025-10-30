#!/usr/bin/env python3
"""
Create a TINY synthetic OAG dataset for quick testing with GCN.
Very small subset: 50 papers, 20 authors, 5 venues, 10 fields
"""

import os
import dill
import numpy as np
import pandas as pd
from pyHGT.data import Graph

def create_tiny_oag_dataset(output_dir='./dataset', n_papers=50, n_authors=20, n_venues=5, n_fields=10):
    """
    Create a TINY synthetic dataset that matches OAG format exactly
    """
    print(f"🚀 Creating TINY synthetic OAG dataset:")
    print(f"  Papers: {n_papers}")
    print(f"  Authors: {n_authors}")
    print(f"  Venues: {n_venues}")
    print(f"  Fields: {n_fields}")
    
    np.random.seed(42)
    
    # Create graph object
    graph = Graph()
    
    # Create paper features (128-dimensional embeddings)
    paper_features = np.random.randn(n_papers, 128)
    paper_df = pd.DataFrame({
        'emb': [paper_features[i].tolist() for i in range(n_papers)],
        'year': np.random.randint(2010, 2020, n_papers),
        'citation': np.random.randint(0, 100, n_papers),  # Citation count
        'title': ['Paper_' + str(i) for i in range(n_papers)]  # Title field for feature extraction
    })
    graph.node_feature['paper'] = paper_df
    
    # Create author features
    author_features = np.random.randn(n_authors, 128)
    author_df = pd.DataFrame({
        'emb': [author_features[i].tolist() for i in range(n_authors)],
        'citation': np.random.randint(0, 50, n_authors)  # Citation count
    })
    graph.node_feature['author'] = author_df
    
    # Create venue features
    venue_features = np.random.randn(n_venues, 128)
    venue_df = pd.DataFrame({
        'emb': [venue_features[i].tolist() for i in range(n_venues)], 
        'citation': np.random.randint(0, 200, n_venues)  # Citation count
    })
    graph.node_feature['venue'] = venue_df
    
    # Create field features
    field_features = np.random.randn(n_fields, 128)
    field_df = pd.DataFrame({
        'emb': [field_features[i].tolist() for i in range(n_fields)],
        'citation': np.random.randint(0, 150, n_fields)  # Citation count
    })
    graph.node_feature['field'] = field_df
    
    # Initialize edge lists (OAG format)
    graph.edge_list = {
        'paper': {
            'author': {
                'PA': {},
                'rev_PA': {}
            },
            'venue': {
                'PV': {},
                'rev_PV': {}
            },
            'field': {
                'PF_in_L2': {},  # Paper-Field edges (L2 level)
                'rev_PF_in_L2': {}
            }
        },
        'author': {
            'paper': {
                'PA': {},
                'rev_PA': {}
            }
        },
        'venue': {
            'paper': {
                'PV': {},
                'rev_PV': {}
            }
        },
        'field': {
            'paper': {
                'PF_in_L2': {},
                'rev_PF_in_L2': {}
            }
        }
    }
    
    # Create edges
    print("Creating edges...")
    
    # Paper-Author edges
    for paper_id in range(n_papers):
        n_authors_per_paper = np.random.randint(1, 4)  # 1-3 authors per paper
        author_ids = np.random.choice(n_authors, n_authors_per_paper, replace=False)
        
        graph.edge_list['paper']['author']['PA'][paper_id] = {}
        for author_id in author_ids:
            year = graph.node_feature['paper'].iloc[paper_id]['year']
            graph.edge_list['paper']['author']['PA'][paper_id][author_id] = year
            # Reverse edge
            if author_id not in graph.edge_list['author']['paper']['PA']:
                graph.edge_list['author']['paper']['PA'][author_id] = {}
            graph.edge_list['author']['paper']['PA'][author_id][paper_id] = year
    
    # Paper-Venue edges
    for paper_id in range(n_papers):
        venue_id = np.random.randint(0, n_venues)
        year = graph.node_feature['paper'].iloc[paper_id]['year']
        
        graph.edge_list['paper']['venue']['PV'][paper_id] = {venue_id: year}
        # Reverse edge
        if venue_id not in graph.edge_list['venue']['paper']['PV']:
            graph.edge_list['venue']['paper']['PV'][venue_id] = {}
        graph.edge_list['venue']['paper']['PV'][venue_id][paper_id] = year
    
    # Paper-Field edges (L2 level) - This is the key for classification task
    for paper_id in range(n_papers):
        n_fields_per_paper = np.random.randint(1, 3)  # 1-2 fields per paper
        field_ids = np.random.choice(n_fields, n_fields_per_paper, replace=False)
        
        graph.edge_list['paper']['field']['PF_in_L2'][paper_id] = {}
        for field_id in field_ids:
            year = graph.node_feature['paper'].iloc[paper_id]['year']
            graph.edge_list['paper']['field']['PF_in_L2'][paper_id][field_id] = year
            # Reverse edge
            if field_id not in graph.edge_list['field']['paper']['PF_in_L2']:
                graph.edge_list['field']['paper']['PF_in_L2'][field_id] = {}
            graph.edge_list['field']['paper']['PF_in_L2'][field_id][paper_id] = year
    
    # Also need rev_PF_in_L2 for training script
    for paper_id in graph.edge_list['paper']['field']['PF_in_L2']:
        graph.edge_list['paper']['field']['rev_PF_in_L2'][paper_id] = {}
        for field_id in graph.edge_list['paper']['field']['PF_in_L2'][paper_id]:
            year = graph.edge_list['paper']['field']['PF_in_L2'][paper_id][field_id]
            graph.edge_list['paper']['field']['rev_PF_in_L2'][paper_id][field_id] = year
    
    # Create time information
    graph.times = {year: True for year in range(2010, 2020)}
    
    print(f"\n✅ Dataset created successfully!")
    print(f"  Total papers: {n_papers}")
    print(f"  Total fields (L2): {n_fields}")
    print(f"  Papers with fields: {len(graph.edge_list['paper']['field']['PF_in_L2'])}")
    
    # Save the dataset
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'graph_CS.pk')
    print(f"💾 Saving to {output_file}...")
    dill.dump(graph, open(output_file, 'wb'))
    
    print("✅ TINY OAG dataset created successfully!")
    return graph

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create TINY synthetic OAG dataset')
    parser.add_argument('--output_dir', type=str, default='./dataset',
                        help='Output directory for the dataset')
    parser.add_argument('--n_papers', type=int, default=50,
                        help='Number of papers (default: 50 for tiny dataset)')
    parser.add_argument('--n_authors', type=int, default=20,
                        help='Number of authors (default: 20)')
    parser.add_argument('--n_venues', type=int, default=5,
                        help='Number of venues (default: 5)')
    parser.add_argument('--n_fields', type=int, default=10,
                        help='Number of fields (default: 10)')
    
    args = parser.parse_args()
    
    create_tiny_oag_dataset(
        args.output_dir,
        args.n_papers,
        args.n_authors,
        args.n_venues,
        args.n_fields
    )
