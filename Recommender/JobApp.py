import dash
from dash import dcc, html, Input, Output, State, callback_context
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import umap
import plotly.graph_objs as go
from dash.exceptions import PreventUpdate
import networkx as nx
import plotly.express as px


print("start loading existing data")
existing_data_df = pd.read_csv('firmInfo_cleaned_longProsCons_embeddings_utf8.csv')
print("existing data loaded")
# Initialize Dash application
app = dash.Dash(__name__)

# Define style constants
COLORS = {
    'background': '#FFFFFF',
    'button': '#43A047',
    'slider_selected': '#1E88E5',
    'slider_unselected': '#E0E0E0',
    'highlight': '#FB8C00',
    'error': '#FF0000',
    'text_light': '#757575'
}

# Define unified style constants
COMMON_STYLES = {
    'LABEL_STYLE': {
        'fontSize': '14px',
        'fontWeight': '500',
        'marginBottom': '8px',
        'marginTop': '15px',
        'color': COLORS['text_light'],
        'display': 'block',
        'lineHeight': '1.4'
    },
    'INPUT_BOX_STYLE': {
        'width': '250px',
        'padding': '8px',
        'marginBottom': '0px',
        'borderRadius': '4px',
        'border': f'1px solid {COLORS["slider_unselected"]}',
        'fontSize': '14px',
        'color': COLORS['text_light'],
        'backgroundColor': COLORS['background'],
        'lineHeight': '1.4'
    },
    'SCORE_BOX_STYLE': {
        'width': '250px',
        'padding': '8px',
        'marginBottom': '0px',
        'borderRadius': '4px',
        'border': f'1px solid {COLORS["slider_unselected"]}',
        'fontSize': '14px',
        'color': COLORS['text_light'],
        'backgroundColor': COLORS['background'],
        'lineHeight': '1.4'
    },
    'TEXT_AREA_STYLE': {
        'width': '250px',
        'height': '80px',
        'padding': '8px',
        'marginBottom': '0px',
        'borderRadius': '4px',
        'border': f'1px solid {COLORS["slider_unselected"]}',
        'fontSize': '14px',
        'color': COLORS['text_light'],
        'resize': 'none',
        'backgroundColor': COLORS['background'],
        'lineHeight': '1.4'
    },
    'ERROR_STYLE': {
        'color': COLORS['error'],
        'fontSize': '12px',
        'height': '0px',
        'overflow': 'visible',
        'position': 'absolute',
        'marginBottom': '0px'
    }
}

# Custom CSS styles
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
        <style>
            * { font-family: 'Roboto', sans-serif; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Application layout
app.layout = html.Div([
    # Title area
    html.H1(
        "Dream Job Navigator",
        style={
            'fontWeight': 'bold',
            'fontSize': '24px',
            'textAlign': 'center',
            'margin': '20px 0',
            'color': '#4169E1', # Royal blue, contrast with light blue background
            'backgroundColor': '#f0f5ff',
            'padding': '15px',
            'borderRadius': '8px'
        }
    ),
    
    # Main content area
    html.Div([
        # Left interaction area
        html.Div([
            # User information input area
            html.Div([
                html.Label(
                    "Your Dream Firm",
                    style=COMMON_STYLES['LABEL_STYLE']
                ),
                dcc.Input(
                    id='firm-name',
                    value='Dream Firm',
                    disabled=True,
                    style={
                        **COMMON_STYLES['INPUT_BOX_STYLE'],
                        'marginBottom': '0px'
                    }
                ),
                
                # Score input area
                html.Div([
                    *[html.Div([
                        html.Label(
                            score,
                            style=COMMON_STYLES['LABEL_STYLE']
                        ),
                        dcc.Input(
                            id=f'score-{score.lower().replace(" ", "-")}',
                            type='number',
                            value=3.2,
                            min=0,
                            max=5,
                            style=COMMON_STYLES['SCORE_BOX_STYLE']
                        ),
                        html.Div(
                            id=f'error-{score.lower().replace(" ", "-")}',
                            style={
                                'color': COLORS['error'],
                                'fontSize': '12px',
                                'height': '0px',
                                'overflow': 'visible',
                                'position': 'absolute'
                            }
                        )
                    ]) for score in ["Opportunities", "Compensation", "Management", 
                                   "Worklife Balance", "Culture", "Diversity"]]
                ]),
                
                # Reviews input area
                html.Div([
                    html.Label(
                        "Pros",
                        style=COMMON_STYLES['LABEL_STYLE']
                    ),
                    dcc.Textarea(
                        id='pros-input',
                        value='This company provides global learning opportunities and values employee growth.',
                        style={
                            **COMMON_STYLES['TEXT_AREA_STYLE'],
                            'marginBottom': '15px'
                        }
                    ),
                    html.Div(
                        id='error-pros',
                        style={
                            'color': COLORS['error'],
                            'fontSize': '12px',
                            'position': 'absolute',
                            'height': '0',
                            'margin': '0',
                            'padding': '0',
                            'visibility': 'hidden'
                        }
                    ),
                    
                    html.Label(
                        "Cons",
                        style={
                            **COMMON_STYLES['LABEL_STYLE'],
                            'marginTop': '0px'
                        }
                    ),
                    dcc.Textarea(
                        id='cons-input',
                        value='This company cannot reimburse the cost of learning new skills.',
                        style=COMMON_STYLES['TEXT_AREA_STYLE']
                    ),
                    html.Div(
                        id='error-cons',
                        style={
                            'color': COLORS['error'],
                            'fontSize': '12px',
                            'position': 'absolute',
                            'height': '0',
                            'margin': '0',
                            'padding': '0',
                            'visibility': 'hidden'
                        }
                    )
                ]),
                
                # Slider area
                html.Div([
                    html.Label(
                        "Weight",
                        style=COMMON_STYLES['LABEL_STYLE']
                    ),
                    dcc.Slider(
                        id='weight-slider',
                        min=0,
                        max=100,
                        value=50,
                        marks={
                            0: 'Scores',
                            100: 'Reviews'
                        },
                        step=1,
                        className='custom-slider'
                    ),
                    
                    html.Label(
                        "Distribution Resolution",
                        style={
                            **COMMON_STYLES['LABEL_STYLE'],
                            'marginTop': '20px'
                        }
                    ),
                    dcc.Slider(
                        id='cluster-slider',
                        min=2,
                        max=10,
                        value=5,
                        marks={
                            2: '2',
                            4: '4', 
                            6: '6',
                            8: '8',
                            10: '10'
                        },
                        step=1,
                        className='custom-slider'
                    ),
                    # Search button
                    html.Button(
                        'Search',
                        id='search-button',
                        style={
                            'width': '250px',
                            'padding': '8px',
                            'marginTop': '20px',
                            'backgroundColor': COLORS['button'],
                            'color': 'white',
                            'border': 'none',
                            'borderRadius': '4px',
                            'cursor': 'pointer',
                            'fontSize': '14px',
                            'lineHeight': '1.4'
                        }
                    )
                ], style={'marginTop': '20px'})
            ], style={'padding': '20px 15px'})
        ], style={'width': '15%', 'backgroundColor': COLORS['background']}),

        # Company information display area
        html.Div([
            html.Div([
                html.Label(
                    "Selected Firm Info",
                    style=COMMON_STYLES['LABEL_STYLE']
                ),
                html.Div(
                    "The Firm You Selected",
                    id='selected-firm-name',
                    style=COMMON_STYLES['INPUT_BOX_STYLE']
                ),
                
                # Score display area
                html.Div([
                    *[html.Div([
                        html.Label(
                            score,
                            style=COMMON_STYLES['LABEL_STYLE']
                        ),
                        html.Div(
                            "0.0",
                            id=f'selected-{score.lower().replace(" ", "-")}',
                            style=COMMON_STYLES['SCORE_BOX_STYLE']
                        )
                    ]) for score in ["Opportunities", "Compensation", "Management", 
                                   "Worklife Balance", "Culture", "Diversity"]]
                ]),
                
                # Reviews display area (right side company information display area)
                html.Div([
                    html.Label(
                        "Pros",
                        style=COMMON_STYLES['LABEL_STYLE']
                    ),
                    html.Div(
                        'This will display the summary of pros by LLM of the firm you selected.',
                        id='selected-pros',
                        style={
                            **COMMON_STYLES['TEXT_AREA_STYLE'],
                            'marginBottom': '18px',
                            'backgroundColor': '#f5f5f5',
                            'overflowY': 'auto',
                            'userSelect': 'none',
                            'cursor': 'default'
                        }
                    ),
                    html.Div(
                        id='error-pros-display',
                        style={
                            'color': COLORS['error'],
                            'fontSize': '12px',
                            'position': 'absolute',
                            'height': '0',
                            'margin': '0',
                            'padding': '0',
                            'visibility': 'hidden'
                        }
                    ),
                    
                    html.Label(
                        "Cons",
                        style={
                            **COMMON_STYLES['LABEL_STYLE'],
                            'marginTop': '0px'
                        }
                    ),
                    html.Div(
                        'This will display the summary of cons by LLM of the firm you selected.',
                        id='selected-cons',
                        style={
                            **COMMON_STYLES['TEXT_AREA_STYLE'],
                            'backgroundColor': '#f5f5f5',
                            'overflowY': 'auto',
                            'userSelect': 'none',
                            'cursor': 'default'
                        }
                    ),
                    html.Div(
                        id='error-cons-display',
                        style={
                            'color': COLORS['error'],
                            'fontSize': '12px',
                            'position': 'absolute',
                            'height': '0',
                            'margin': '0',
                            'padding': '0',
                            'visibility': 'hidden'
                        }
                    )
                ])
            ], style={'padding': '20px 15px'})
        ], style={'width': '15%', 'backgroundColor': COLORS['background']}),

        # Right visualization area
        html.Div([
            dcc.Loading(
                id="loading-graph",
                type="circle",
                children=[
                    dcc.Graph(
                        id='company-graph',
                        figure={
                            'layout': {
                                'xaxis': {'visible': False},
                                'yaxis': {'visible': False},
                                'plot_bgcolor': 'white',
                                'paper_bgcolor': 'white',
                                'showgrid': False
                            }
                        },
                        style={'height': '100vh'}  # Keep original height, use webpage scrollbar
                    )
                ]
            ),
            # Legend area
            html.Div([
                # Dream Firm legend
                html.Div([
                    html.Div(style={
                        'width': '20px',
                        'height': '20px',
                        'borderRadius': '50%',
                        'backgroundColor': '#ADD8E6',
                        'border': '4px solid rgba(128, 128, 128, 0.8)',
                        'display': 'inline-block',
                        'marginRight': '5px',
                        'verticalAlign': 'middle'
                    }),
                    html.Span('Your Dream Firm', style={**COMMON_STYLES['LABEL_STYLE'], 'verticalAlign': 'middle'})
                ], style={'display': 'inline-block', 'marginRight': '30px'}),
                
                # Typical firm legend
                html.Div([
                    html.Div(style={
                        'width': '20px',
                        'height': '20px',
                        'borderRadius': '50%',
                        'backgroundColor': '#ADD8E6',
                        'border': '2px solid rgba(128, 128, 128, 0.8)',
                        'display': 'inline-block',
                        'marginRight': '5px',
                        'verticalAlign': 'middle'
                    }),
                    html.Span('The Representative Firm Within the Category', style={**COMMON_STYLES['LABEL_STYLE'], 'verticalAlign': 'middle'})
                ], style={'display': 'inline-block', 'marginRight': '30px'}),
                
                # Size and color legend
                html.Div([
                    html.Div(style={
                        'width': '25px',
                        'height': '25px',
                        'borderRadius': '50%',
                        'backgroundColor': '#ADD8E6',
                        'display': 'inline-block',
                        'marginRight': '5px',
                        'verticalAlign': 'middle'
                    }),
                    html.Div(style={
                        'width': '15px',
                        'height': '15px',
                        'borderRadius': '50%',
                        'backgroundColor': '#ADD8E6',
                        'display': 'inline-block',
                        'marginRight': '5px',
                        'verticalAlign': 'middle'
                    }),
                    html.Span('Node Size: Indicates the Average Quantitative Score. Node Color: Indicates the Category', 
                             style={**COMMON_STYLES['LABEL_STYLE'], 'verticalAlign': 'middle'})
                ], style={'display': 'inline-block'})
            ], style={
                'textAlign': 'center',
                'padding': '10px',
                'backgroundColor': 'white',
                'borderRadius': '5px',
                'marginTop': '10px'
            })
        ], style={'width': '70%', 'backgroundColor': COLORS['background']})
        
    ], style={'display': 'flex', 'height': 'calc(100vh - 80px)'})
], style={'backgroundColor': COLORS['background']})





def obtain_user_data(firm_name, opportunities_score, compensation_score, management_score, worklife_balance_score, culture_score, diversity_score, pros_text, cons_text):
    user_data = {
        'firm id': 9999,
        'firm name': firm_name,
        'opportunities': opportunities_score,
        'compensation': compensation_score,
        'management': management_score,
        'worklife_balance': worklife_balance_score,
        'culture': culture_score,
        'diversity': diversity_score,
        'pros_text': pros_text,
        'cons_text': cons_text
    }
    return user_data


def process_user_input(user_data):
    """
    Process user input text and score data, generate user company data frame
    
    Parameters:
    user_data: dict, containing firm_id, firm_name, 6-dimensional score and pros_text, cons_text
    
    Returns:
    pd.DataFrame: DataFrame containing user company information
    """
    # 1. Extract data from user_data
    firm_id = user_data['firm id']
    firm_name = user_data['firm name']
    
    # 2. Get scores
    opportunities = user_data['opportunities']
    compensation = user_data['compensation'] 
    management = user_data['management']
    worklife_balance = user_data['worklife_balance']
    culture = user_data['culture']
    diversity = user_data['diversity']
    
    # 3. Get text
    pros_text = user_data['pros_text']
    cons_text = user_data['cons_text']
    
    # 4. Use sentence transformer to process text
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Load pre-trained model
    
    # Generate embeddings for pros
    pros_embedding = model.encode(pros_text)
    pros_embedding_dict = {f'pros_text_dim{i}': value for i, value in enumerate(pros_embedding)}
    
    # Generate embeddings for cons
    cons_embedding = model.encode(cons_text)
    cons_embedding_dict = {f'cons_text_dim{i}': value for i, value in enumerate(cons_embedding)}
    
    # 5. Create DataFrame
    data = {
        'firm id': firm_id,
        'firm name': firm_name,
        'opportunities': opportunities,
        'compensation': compensation,
        'management': management,
        'worklife_balance': worklife_balance,
        'culture': culture,
        'diversity': diversity,
        'pros_text': pros_text,
        'cons_text': cons_text,
        **pros_embedding_dict,  # Expand pros' 384-dimensional vector
        **cons_embedding_dict   # Expand cons' 384-dimensional vector
    }
    
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    return df


def perform_data_processing(user_data_df, existing_data_df, W_reviews, k):
    """Process user data and existing data, perform dimensionality reduction, normalization, clustering, etc."""
    # 1. Merge data
    combined_data_df = pd.concat([existing_data_df, user_data_df], ignore_index=True)
    
    # 2. UMAP dimensionality reduction
    # Process pros text
    pros_cols = [f'pros_text_dim{i}' for i in range(384)]
    pros_embeddings = combined_data_df[pros_cols].values
    reducer_pros = umap.UMAP(n_components=12, n_jobs=-1)
    pros_umap = reducer_pros.fit_transform(pros_embeddings)
    
    # Process cons text
    cons_cols = [f'cons_text_dim{i}' for i in range(384)]
    cons_embeddings = combined_data_df[cons_cols].values
    reducer_cons = umap.UMAP(n_components=12, n_jobs=-1)
    cons_umap = reducer_cons.fit_transform(cons_embeddings)
    
    # Add dimensionality reduction results to DataFrame
    for i in range(12):
        combined_data_df[f'pros_text_umap{i}'] = pros_umap[:, i]
        combined_data_df[f'cons_text_umap{i}'] = cons_umap[:, i]
    
    # 3. Normalization
    # Normalize scores
    score_cols = ['opportunities', 'compensation', 'management', 
                 'worklife_balance', 'culture', 'diversity']
    for col in score_cols:
        combined_data_df[f'{col}_norm'] = (combined_data_df[col] - combined_data_df[col].min()) / \
                                        (combined_data_df[col].max() - combined_data_df[col].min())
    
    # Normalize UMAP results
    umap_cols = [f'pros_text_umap{i}' for i in range(12)] + [f'cons_text_umap{i}' for i in range(12)]
    for col in umap_cols:
        combined_data_df[f'{col}_norm'] = (combined_data_df[col] - combined_data_df[col].min()) / \
                                        (combined_data_df[col].max() - combined_data_df[col].min())
    
    # 4. Calculate similarity
    # Prepare normalized scores and UMAP features
    norm_scores = combined_data_df[[f'{col}_norm' for col in score_cols]].values
    norm_umap = combined_data_df[[f'{col}_norm' for col in umap_cols]].values
    
    # Get index of user data (last row)
    user_idx = len(combined_data_df) - 1
    
    # Calculate cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    scores_similarity = cosine_similarity(norm_scores)[user_idx]
    reviews_similarity = cosine_similarity(norm_umap)[user_idx]
    
    # Calculate combined similarity
    combined_similarity = (1 - W_reviews) * scores_similarity + W_reviews * reviews_similarity
    combined_data_df['similarity_scores'] = combined_similarity
    
    # 5. Calculate score average
    norm_score_cols = [f'{col}_norm' for col in score_cols]
    combined_data_df['six_dimension_scores_mean'] = combined_data_df[norm_score_cols].mean(axis=1)
    
    # 6. Clustering
    # Prepare clustering features
    cluster_features = np.concatenate([
        (1 - W_reviews) * norm_scores,
        W_reviews * norm_umap
    ], axis=1)
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(cluster_features)
    combined_data_df['cluster_label'] = cluster_labels
    
    # Identify cluster centers
    # Calculate distance to cluster centers
    distances_to_center = kmeans.transform(cluster_features)
    cluster_centers = np.zeros(len(combined_data_df), dtype=bool)
    
    # For each cluster, find the point closest to center
    for i in range(k):
        mask = cluster_labels == i
        if np.any(mask):
            closest_to_center = distances_to_center[mask, i].argmin()
            cluster_centers[np.where(mask)[0][closest_to_center]] = True
    
    combined_data_df['cluster_center'] = cluster_centers
    
    # Calculate within-cluster similarity
    similarity_scores_within_cluster = np.zeros(len(combined_data_df))
    
    # Iterate through each cluster, calculate similarity between companies and center company
    for cluster_id in range(k):
        # Filter companies belonging to current cluster
        cluster_mask = combined_data_df['cluster_label'] == cluster_id
        cluster_firms = combined_data_df[cluster_mask]
        
        # Get center company of the cluster (company closest to cluster center)
        center_firm = cluster_firms[cluster_firms['cluster_center']].iloc[0]
        center_idx = center_firm.name
        
        # Extract feature vectors for center company and cluster companies
        center_scores = norm_scores[center_idx].reshape(1, -1)
        center_umap = norm_umap[center_idx].reshape(1, -1)
        
        cluster_indices = cluster_firms.index
        cluster_scores = norm_scores[cluster_indices]
        cluster_umap = norm_umap[cluster_indices]
        
        # Calculate score similarity and text similarity, combine with weights
        scores_sim = cosine_similarity(cluster_scores, center_scores).flatten()
        reviews_sim = cosine_similarity(cluster_umap, center_umap).flatten()
        cluster_similarity = (1 - W_reviews) * scores_sim + W_reviews * reviews_sim
        
        # Save calculated similarities to result array
        similarity_scores_within_cluster[cluster_indices] = cluster_similarity
    
    # Add within-cluster similarity as new feature to dataframe
    combined_data_df['similarity_scores_within_cluster'] = similarity_scores_within_cluster
    
    return combined_data_df

def create_graph(combined_data_df):
    """Create network graph structure containing nodes and edges"""
    user_edge_threshold=0.05
    center_edge_threshold=0.1
    # 1. Prepare node data
    n_clusters = len(combined_data_df['cluster_label'].unique())
    colors = px.colors.qualitative.Set3[:n_clusters]
    
    # Create graph structure
    G = nx.Graph()
    
    # Store node attributes
    node_attrs = {}
    
    # Add nodes and node attributes
    # Calculate once score range
    min_score = combined_data_df['six_dimension_scores_mean'].min()
    max_score = combined_data_df['six_dimension_scores_mean'].max()
    
    for _, row in combined_data_df.iterrows():
        node_id = row['firm id']
        G.add_node(node_id)
        normalized_score = (row['six_dimension_scores_mean'] - min_score) / (max_score - min_score)
        node_attrs[node_id] = {
            'label': row['firm name'],
            'size': 2 + (normalized_score * 45),
            'color': colors[int(row['cluster_label'])],
            'is_center': row['cluster_center'],
            'cluster': int(row['cluster_label']),
            'customdata': node_id  # Add node ID as customdata
        }
    
    # Add edges with customdata
    for edge in G.edges():
        source, target = edge
        G.edges[edge]['customdata'] = [source, target]  # Store start and end node IDs of edges
    
    # 2. Add edges
    user_node_id = 9999
    
    # Calculate user similarity threshold
    user_similarities = combined_data_df['similarity_scores'].values
    user_sim_threshold = np.percentile(user_similarities, (1 - user_edge_threshold) * 100)
    
    # Add user node to all other nodes
    for _, row in combined_data_df.iterrows():
        if row['firm id'] != user_node_id:  # Exclude user node itself
            similarity = row['similarity_scores']
            edge_alpha = max(0.01, similarity) if similarity >= user_sim_threshold else 0.01
            G.add_edge(user_node_id, row['firm id'],
                      weight=similarity,
                      color=f'rgba(128, 128, 128, {edge_alpha})')
    
    # 3. Add edges from cluster centers to other nodes in the same cluster
    # Get center nodes of each cluster
    cluster_centers = {}
    for _, row in combined_data_df.iterrows():
        if row['cluster_center']:
            cluster_centers[int(row['cluster_label'])] = row['firm id']
    
    # Calculate similarity threshold for each cluster
    for cluster_id in cluster_centers.keys():
        cluster_mask = combined_data_df['cluster_label'] == cluster_id
        cluster_similarities = combined_data_df[cluster_mask]['similarity_scores_within_cluster'].values
        cluster_sim_threshold = np.percentile(cluster_similarities, (1 - center_edge_threshold) * 100)
        
        # Add edges from center to other nodes in the same cluster
        cluster_firms = combined_data_df[cluster_mask]
        center_id = cluster_centers[cluster_id]
        
        for _, row in cluster_firms.iterrows():
            if row['firm id'] != center_id:  # Exclude center node itself
                similarity = row['similarity_scores_within_cluster']
                edge_alpha = max(0.01, similarity) if similarity >= cluster_sim_threshold else 0.1
                G.add_edge(center_id, row['firm id'],
                          weight=similarity * 0.5,
                          color=f'rgba(128, 128, 128, {edge_alpha})')
    
    return G, node_attrs

def visualize_graph(G, node_attrs):
    """Visualize network graph, ensure correct customdata setting"""
    # 1. Create layout
    pos = nx.spring_layout(G, k=1/np.sqrt(len(G.nodes)), 
                         weight='weight',
                         iterations=20)
    
    # 2. Create edge traces
    edge_traces = []
    for edge in G.edges(data=True):
        source, target = edge[0], edge[1]
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        
        edge_trace = go.Scatter(
            x=[x0, x1], y=[y0, y1],
            line=dict(
                width=1, 
                color=edge[2].get('color', 'rgba(128, 128, 128, 0.1)')
            ),
            hoverinfo='none',
            mode='lines',
            showlegend=False,
            customdata=[[source, target]]  # Store start and end node IDs of edges
        )
        edge_traces.append(edge_trace)
    
    # 3. Create node traces
    n_clusters = max(attr['cluster'] for attr in node_attrs.values()) + 1
    node_traces = []
    
    for cluster in range(n_clusters):
        # Get all nodes of the cluster
        cluster_nodes = {node: attr for node, attr in node_attrs.items() 
                        if attr['cluster'] == cluster}
        
        if not cluster_nodes:
            continue
            
        # Collect node data
        x = []
        y = []
        node_ids = []
        marker_sizes = []
        marker_colors = []
        hover_texts = []
        line_widths = []
        line_colors = []
        
        for node, attr in cluster_nodes.items():
            x.append(pos[node][0])
            y.append(pos[node][1])
            node_ids.append(node)  # Store node ID as customdata
            marker_sizes.append(attr['size'])
            marker_colors.append(attr['color'])
            hover_texts.append(attr['label'])
            
            # Set node border style
            if node == 9999:  # User node
                line_widths.append(4)
                line_colors.append('rgba(128, 128, 128, 0.8)')
            elif attr['is_center']:  # Cluster center node
                line_widths.append(2)
                line_colors.append('rgba(128, 128, 128, 0.8)')
            else:  # Normal node
                line_widths.append(1)
                line_colors.append('rgba(255, 255, 255, 0.5)')
        
        # Create node trace for the cluster
        node_trace = go.Scatter(
            x=x, y=y,
            mode='markers',
            hoverinfo='text',
            text=hover_texts,
            marker=dict(
                size=marker_sizes,
                color=marker_colors,
                line=dict(
                    width=line_widths,
                    color=line_colors
                )
            ),
            customdata=node_ids,  # Store node IDs
            showlegend=False
        )
        node_traces.append(node_trace)
    
    # 4. Create layout
    layout = go.Layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        uirevision=True  # Keep view state
    )
    
    # 5. Create figure
    fig = go.Figure(data=[*edge_traces, *node_traces], layout=layout)
    
    return fig

# Add new callback function to handle node click events
@app.callback(
    [Output('company-graph', 'figure', allow_duplicate=True),
     Output('selected-firm-name', 'children'),
     Output('selected-opportunities', 'children'),
     Output('selected-compensation', 'children'), 
     Output('selected-management', 'children'),
     Output('selected-worklife-balance', 'children'),
     Output('selected-culture', 'children'),
     Output('selected-diversity', 'children'),
     Output('selected-pros', 'children'),
     Output('selected-cons', 'children')],
    [Input('company-graph', 'clickData')],
    [State('company-graph', 'figure')],
    prevent_initial_call=True
)
def update_highlight_and_info(clickData, current_figure):
    """
    Handle node click events:
    1. Highlight clicked node and related edges in red
    2. Update company information display area
    """
    if not clickData:
        raise PreventUpdate
    
    # Get clicked node ID
    clicked_node = clickData['points'][0]['customdata']
    
    # Update edges and nodes in the graph
    for trace in current_figure['data']:
        # Process edge traces
        if trace.get('mode') == 'lines':
            edge_nodes = trace['customdata'][0]
            # Reset all edge styles
            trace['line']['color'] = 'rgba(211, 211, 211, 0.3)'  # Default color
            trace['line']['width'] = 1  # Default width
            # If edge connects to clicked node, add red highlight
            if clicked_node in edge_nodes:
                trace['line']['color'] = '#8B0000'  # Red
                trace['line']['width'] = 3
        
        # Process node traces
        elif trace.get('mode') == 'markers':
            # Reset all node border styles
            for i, node_id in enumerate(trace['customdata']):
                if node_id == 9999:  # User node
                    trace['marker']['line']['color'][i] = 'rgba(128, 128, 128, 0.8)'
                    trace['marker']['line']['width'][i] = 4
                elif trace.get('is_center', False):  # Cluster center node
                    trace['marker']['line']['color'][i] = 'rgba(128, 128, 128, 0.8)'
                    trace['marker']['line']['width'][i] = 2
                else:  # Normal node
                    trace['marker']['line']['color'][i] = 'rgba(255, 255, 255, 0.5)'
                    trace['marker']['line']['width'][i] = 1
                # Add red highlight to clicked node
                if node_id == clicked_node:
                    trace['marker']['line']['color'][i] = '#8B0000'  # Red
                    trace['marker']['line']['width'][i] = 3
    
    # Get information of the clicked company
    if clicked_node == 9999:  # User input ideal company
        company_data = existing_data_df[existing_data_df['firm id'] == clicked_node].iloc[0]
    else:
        company_data = existing_data_df[existing_data_df['firm id'] == clicked_node].iloc[0]
    
    return (current_figure,
            company_data['firm name'],
            company_data['opportunities'],
            company_data['compensation'],
            company_data['management'],
            company_data['worklife_balance'],
            company_data['culture'],
            company_data['diversity'],
            company_data['pros_text'],
            company_data['cons_text'])



@app.callback(
    Output('company-graph', 'figure'),
    [Input('search-button', 'n_clicks')],
    [State('firm-name', 'value'),
     State('score-opportunities', 'value'),
     State('score-compensation', 'value'),
     State('score-management', 'value'),
     State('score-worklife-balance', 'value'),
     State('score-culture', 'value'),
     State('score-diversity', 'value'),
     State('pros-input', 'value'),
     State('cons-input', 'value'),
     State('weight-slider', 'value'),
     State('cluster-slider', 'value')]
)
def update_graph(n_clicks, firm_name, opportunities_score, compensation_score, 
                management_score, worklife_balance_score, culture_score, 
                diversity_score, pros_text, cons_text, weight_slider, cluster_slider):
    """
    Handle user input and update graph
    
    Parameters:
    n_clicks: int, number of button clicks
    firm_name: str, company name
    *_score: float, score for each dimension
    pros_text: str, pros text
    cons_text: str, cons text
    weight_slider: float, reviews weight (0-100)
    cluster_slider: int, number of clusters
    
    Returns:
    fig: plotly.graph_objs.Figure, updated graph
    """
    # If button is not clicked, return empty graph
    if n_clicks is None:
        raise PreventUpdate
    
    try:
        # 1. Get user input data
        user_data = obtain_user_data(
            firm_name=firm_name,
            opportunities_score=opportunities_score,
            compensation_score=compensation_score,
            management_score=management_score,
            worklife_balance_score=worklife_balance_score,
            culture_score=culture_score,
            diversity_score=diversity_score,
            pros_text=pros_text,
            cons_text=cons_text
        )
        
        # 2. Process user input, generate user data DataFrame
        user_data_df = process_user_input(user_data)
        
        # 3. Process data, perform dimensionality reduction, clustering, etc.
        # Convert weight_slider value to 0-1 range
        W_reviews = weight_slider / 100.0
        combined_data_df = perform_data_processing(
            user_data_df=user_data_df,
            existing_data_df=existing_data_df,
            W_reviews=W_reviews,
            k=cluster_slider
        )
        
        # 4. Create graph structure
        G, node_attrs = create_graph(combined_data_df)
        
        # 5. Visualize graph
        fig = visualize_graph(G, node_attrs)
        
        return fig
        
    except Exception as e:
        print(f"Error in update_graph: {str(e)}")
        # Return an empty graph or error prompt graph
        return go.Figure(
            data=[],
            layout=go.Layout(
                title=f"Error: {str(e)}",
                showlegend=False
            )
        )


# Callback function to update trace when node is clicked



if __name__ == '__main__':
    app.run_server(debug=True)




