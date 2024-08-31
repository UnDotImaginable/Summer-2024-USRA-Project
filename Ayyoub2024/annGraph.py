
from graphviz import Digraph
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input


def ann_viz(model, view=True, filename="network.gv", title="My Neural Network"):
    """Visualize a Sequential model.

    # Arguments
        model: A Keras model instance.
        view: whether to display the model after generation.
        filename: where to save the visualization. (a .gv file)
        title: A title for the graph
    """


    input_layer = model.input_shape[1]
    hidden_layers_nr = 0
    layer_types = []
    hidden_layers = []
    output_layer = 0

    input_labels = ['day_of_year', 'time', 'Temperature', 'Dew Point', 'Wind Direction', 
                    'Precipitation', 'Wind Gust', 'Solar Radiation', 'Relative Humidity', 
                    'Cloud Cover', 'Sea Level Pressure']

    for layer in model.layers:
        if layer == model.layers[0]:
            hidden_layers_nr += 1
            if isinstance(layer, keras.layers.Dense):
                hidden_layers.append(layer.units)
                layer_types.append("Dense")
        else:
            if layer == model.layers[-1]:
                output_layer = layer.units
            else:
                hidden_layers_nr += 1
                if isinstance(layer, keras.layers.Dense):
                    hidden_layers.append(layer.units)
                    layer_types.append("Dense")

    last_layer_nodes = input_layer
    nodes_up = input_layer
    if not isinstance(model.layers[0], keras.layers.Dense):
        last_layer_nodes = 1
        nodes_up = 1
        input_layer = 1

    g = Digraph('g', filename=filename)
    n = 0
    g.graph_attr.update(rankdir='TB', splines="false", nodesep='.2', ranksep='5')  # Adjusted values

    # Set the size for all nodes
    node_attr = {'width': '0.5', 'height': '0.5', 'fixedsize': 'true'}

    # Input Layer
    with g.subgraph(name='cluster_input') as c:
        if isinstance(model.layers[0], keras.layers.Dense):
            the_label = title + '\n\n\n\nInput Layer'
            c.attr(color='white')
            for i, label in enumerate(input_labels):
                n += 1
                c.node(str(n), shape='circle', style='filled', color='#2ecc71', fontcolor='#2ecc71',  **node_attr)
            c.attr(label=the_label)
        else:
            raise ValueError("ANN Visualizer: Layer not supported for visualizing")

    # Hidden Layers
    for i in range(hidden_layers_nr):
        with g.subgraph(name="cluster_" + str(i+1)) as c:
            if layer_types[i] == "Dense":
                c.attr(color='white')
                the_label = ""
                c.attr(labeljust="right", labelloc="b", label=the_label)
                for j in range(hidden_layers[i]):
                    n += 1
                    c.node(str(n), shape="circle", style="filled", color="#3498db", fontcolor="#3498db", **node_attr)
                    for h in range(nodes_up - last_layer_nodes + 1, nodes_up + 1):
                        g.edge(str(h), str(n))
                last_layer_nodes = hidden_layers[i]
                nodes_up += hidden_layers[i]

    # Output Layer
    with g.subgraph(name='cluster_output') as c:
        if isinstance(model.layers[-1], keras.layers.Dense):
            c.attr(color='white')
            for i in range(1, output_layer + 1):
                n += 1
                c.node(str(n), shape="circle", style="filled", color="#e74c3c", fontcolor="#e74c3c", **node_attr)
                for h in range(nodes_up - last_layer_nodes + 1, nodes_up + 1):
                    g.edge(str(h), str(n))
            c.attr(label='Output Layer', labelloc="bottom")
            c.node_attr.update(color="#2ecc71", style="filled", fontcolor="#2ecc71", shape="circle")

    g.attr(arrowShape="none")
    g.edge_attr.update(arrowhead="none", color="#707070")

    if view:
        g.view()
    g.save()








# Define the model
model = Sequential()
model.add(Input(shape=(11,)))                 # Input layer with 11 features
model.add(Dense(64, activation='relu'))       # First hidden layer with 64 neurons
model.add(Dense(32, activation='relu'))       # Second hidden layer with 32 neurons
model.add(Dense(16, activation='relu'))       # Third hidden layer with 16 neurons
model.add(Dense(1, activation='linear'))      # Output layer with 1 neuron (Wind Speed)

# Call the ann_viz function to visualize the model
ann_viz(model, title="Artificial Neural Network Model")
