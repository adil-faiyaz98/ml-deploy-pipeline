# knowledge_graphs/ai_knowledge_graph.py
import logging
import networkx as nx
import matplotlib.pyplot as plt

logging.basicConfig(filename='logs/ai_knowledge_graph.log', level=logging.INFO)

# Create a directed graph
G = nx.DiGraph()

# Define relationships
relationships = [
    ("Data Collection", "Model Training"),
    ("Model Training", "Model Evaluation"),
    ("Model Evaluation", "Model Deployment"),
    ("Model Deployment", "Model Monitoring"),
    ("Human Feedback", "Model Improvement"),
    ("Model Monitoring", "Model Improvement")
]

# Add relationships to the graph
G.add_edges_from(relationships)

# Visualize the knowledge graph
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=3000, font_size=10)
plt.savefig("logs/knowledge_graph.png")
logging.info("AI Knowledge Graph generated.")
print("AI Knowledge Graph Created.")
