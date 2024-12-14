from graphviz import Digraph

# function used to generate architecture diagrams of our models
def create_architecture_diagram(model_version):
    dot = Digraph(comment=f'DeepLOB {model_version} Architecture')
    dot.attr(rankdir='TB')

    # colors
    input_color = '#E6F3FF'      
    conv_color = '#FFE6E6'      
    incep_color = '#E6FFE6'  
    attention_color = '#FFE6FF'
    lstm_color = '#FFFFD9'      
    trans_color = '#FFDAB9'     
    output_color = '#F0F0F0'

    if model_version == 'v2':
        dot.node('input', 'Input\n(batch, 1, 100, 40)', style='filled', fillcolor=input_color)
        dot.node('conv1', 'Conv Block 1\n(Conv2d + BatchNorm)', style='filled', fillcolor=conv_color)
        dot.node('conv2', 'Conv Block 2\n(Conv2d + BatchNorm)', style='filled', fillcolor=conv_color)
        dot.node('conv3', 'Conv Block 3\n(Conv2d + BatchNorm)', style='filled', fillcolor=conv_color)
        dot.node('incep', 'Inception Blocks\n(3 Paths: Conv + Pool)', style='filled', fillcolor=incep_color)
        dot.node('attention', 'Multihead Attention\n(8 Heads)', style='filled', fillcolor=attention_color)
        dot.node('lstm', 'LSTM\n(1 layer, hidden=64)', style='filled', fillcolor=lstm_color)
        dot.node('fc', 'Fully Connected\n(Classification)', style='filled', fillcolor=output_color)

        dot.edge('input', 'conv1', penwidth='1.5')
        dot.edge('conv1', 'conv2', penwidth='1.5')
        dot.edge('conv2', 'conv3', penwidth='1.5')
        dot.edge('conv3', 'incep', penwidth='1.5')
        dot.edge('incep', 'attention', penwidth='1.5')
        dot.edge('attention', 'lstm', penwidth='1.5')
        dot.edge('lstm', 'fc', penwidth='1.5')

    elif model_version == 'transformer':
        dot.node('input', 'Input\n(batch, 1, seq_len, 40)', style='filled', fillcolor=input_color)
        dot.node('conv', 'Conv Reduction\n(Conv1d + MaxPool)', style='filled', fillcolor=conv_color)
        dot.node('input_proj', 'Input Projection\n(Linear Layer)', style='filled', fillcolor=incep_color)
        dot.node('pos', 'Positional Encoding', style='filled', fillcolor=attention_color)
        dot.node('trans', 'Transformer Encoder\n(3 layers, d_model=32)', style='filled', fillcolor=trans_color)
        dot.node('fc1', 'Fully Connected\n(ReLU, hidden=64)', style='filled', fillcolor=lstm_color)
        dot.node('fc2', 'Fully Connected\n(Output Layer)', style='filled', fillcolor=output_color)

        dot.edge('input', 'conv', penwidth='1.5')
        dot.edge('conv', 'input_proj', penwidth='1.5')
        dot.edge('input_proj', 'pos', penwidth='1.5')
        dot.edge('pos', 'trans', penwidth='1.5')
        dot.edge('trans', 'fc1', penwidth='1.5')
        dot.edge('fc1', 'fc2', penwidth='1.5')
    
    elif model_version == 'baseline':
        dot.node('input', 'Input\n(batch, 1, 100, 40)', style='filled', fillcolor=input_color)
        dot.node('conv1', 'Conv Block 1\n(Conv2d + BatchNorm + ReLU)', style='filled', fillcolor=conv_color)
        dot.node('conv2', 'Conv Block 2\n(Conv2d + BatchNorm + Tanh)', style='filled', fillcolor=conv_color)
        dot.node('conv3', 'Conv Block 3\n(Conv2d + BatchNorm + ReLU)', style='filled', fillcolor=conv_color)
        dot.node('incep', 'Inception Blocks\n(3 Paths: Conv + Pool)', style='filled', fillcolor=incep_color)
        dot.node('lstm', 'LSTM\n(1 layer, hidden=64)', style='filled', fillcolor=lstm_color)
        dot.node('fc', 'Fully Connected\n(Classification)', style='filled', fillcolor=output_color)

        dot.edge('input', 'conv1', penwidth='1.5')
        dot.edge('conv1', 'conv2', penwidth='1.5')
        dot.edge('conv2', 'conv3', penwidth='1.5')
        dot.edge('conv3', 'incep', penwidth='1.5')
        dot.edge('incep', 'lstm', penwidth='1.5')
        dot.edge('lstm', 'fc', penwidth='1.5')

    # save the diagram
    dot.render(f"DeepLOB_{model_version}_architecture", format="png", cleanup=True)

if __name__ == "__main__":
    print('running')
    create_architecture_diagram('transformer')
