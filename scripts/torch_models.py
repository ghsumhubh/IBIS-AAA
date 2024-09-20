import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler

def get_default_params(params=None):
    if params is None:
        params = {}

    params['model_type'] = params.get('model_type', 'cnn')

    params['steps_per_epoch'] = params.get('steps_per_epoch', None)
    params['patience'] = params.get('patience', 50)
    params['experiment'] = params.get('experiment', None)
    params['validation_split'] = params.get('validation_split', 0.1)
    params['epochs'] = params.get('epochs', 200)
    params['batch_size'] = params.get('batch_size', 2048)

    if params['experiment'] in [None, 'PBM', 'HTS']:
        params['loss'] = params.get('loss', nn.MSELoss())
    elif params['experiment'] in ['SMS', 'CHS']:
        params['loss'] = params.get('loss', nn.BCELoss())
    else:
        raise ValueError('Experiment not implemented')

    # Optimizer will be initialized later with model parameters
    #params['optimizer_args'] = params.get('optimizer_args', {'lr': 0.0001})
    params['optimizer_class'] = params.get('optimizer_class', optim.Adam)
    params['learning_rate'] = params.get('learning_rate', 0.0001)

    
    params['n_filters'] = params.get('n_filters', 64)
    params['sizes_to_use'] = params.get('sizes_to_use', [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    params['conv_activation'] = params.get('conv_activation', nn.ReLU())
    params['dropout_rate'] = params.get('dropout_rate', 0.5)
    params['dense_list'] = params.get('dense_list', [(128, 0.2), (64, 0), (32, 0), (32, 0)])
    params['dense_activation'] = params.get('dense_activation', nn.ReLU())
    params['kernel_initializer_conv'] = params.get('kernel_initializer_conv', nn.init.kaiming_normal_)
    params['kernel_initializer_dense'] = params.get('kernel_initializer_dense', nn.init.kaiming_normal_)
    params['kernel_initializer_last_layer'] = params.get('kernel_initializer_last_layer', nn.init.kaiming_normal_)
    params['metrics'] = params.get('metrics', [spearman_correlation])
    params['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



    return params


# Define a dummy spearman_correlation function for placeholder
def spearman_correlation(output, target):
    # Implement your metric here
    return torch.tensor(0.0)



class CNNModel(nn.Module):
    def __init__(self, params):
        super(CNNModel, self).__init__()
        self.params = params
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=4, out_channels=params['n_filters'], kernel_size=size)
            for size in params['sizes_to_use']
        ])
        
        self.conv_activation = params['conv_activation'] if isinstance(params['conv_activation'], nn.Module) else params['conv_activation']()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.drop1 = nn.Dropout(params['dropout_rate'])
        
        # Fully connected layers
        dense_layers = []
        input_size = params['n_filters'] * len(params['sizes_to_use'])
        for size, rate in params['dense_list']:
            dense_layers.append(nn.Linear(input_size, size))
            dense_layers.append(params['dense_activation'] if isinstance(params['dense_activation'], nn.Module) else params['dense_activation']())
            if rate > 0:
                dense_layers.append(nn.Dropout(rate))
            input_size = size
        self.dense = nn.Sequential(*dense_layers)
        
        # Output layer
        self.output_layer = nn.Linear(input_size, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change to (batch_size, channels, seq_length)
        conv_out = [self.pool(self.conv_activation(conv(x))).squeeze(-1) for conv in self.convs]
        x = torch.cat(conv_out, dim=1)
        x = self.drop1(x)
        x = self.dense(x)
        x = self.output_layer(x)
        return x
    
def get_cnn_model(params):
    return CNNModel(params)


def copy_model_weights(original_model, n_count, params=None):
    params = get_default_params()
    new_model = get_cnn_model(params)
    new_model.load_state_dict(original_model.state_dict())
    return new_model
    
def get_trained_model(train_data, train_labels,params):
    params = get_default_params()
    params['n_nucleotides']= train_data.shape[1]
    model = get_cnn_model(params=params)
    model = train_model(model, train_data, train_labels, params)
    return model


def train_model(model, train_data, train_labels, params):
    device = params['device']
    model = model.to(device)
    print(f'Using {device}')
    
    # Enable cuDNN autotuner to find the best algorithms for your hardware
    torch.backends.cudnn.benchmark = True
    
    # Using mixed precision training
    scaler = GradScaler()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Split the data into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.04, random_state=42)

    # Move data to GPU
    train_data = torch.tensor(train_data, dtype=torch.float32).to(device)
    train_labels = torch.tensor(train_labels, dtype=torch.float32).to(device)
    val_data = torch.tensor(val_data, dtype=torch.float32).to(device)
    val_labels = torch.tensor(val_labels, dtype=torch.float32).to(device)

    # Create datasets and data loaders
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True) #num_workers=8
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True,num_workers=8) #num_workers=8


    val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
    #val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False,num_workers=8)


    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(params['epochs']):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast():
                outputs = model(inputs)
                labels = labels.unsqueeze(1)
                loss = criterion(outputs, labels)
            
            # Backward pass with mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)

        # Validate the model
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                with autocast():
                    outputs = model(inputs)
                    labels = labels.unsqueeze(1)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{params['epochs']}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")

        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= params['patience']:
                print("Early stopping")
                model.load_state_dict(torch.load('best_model.pth'))
                break

    return model


def check_if_protein_model_exists(experiment, protein, custom_name=None):
    path = f'models/{experiment}/{protein}_model'
    if custom_name is not None:
        path += f'_{custom_name}'
    path += '.pth'
    return os.path.exists(path)

def save_protein_model(model, experiment, protein, custom_name=None):
    path = f'models/{experiment}/{protein}_model'
    if custom_name is not None:
        path += f'_{custom_name}'
    path += '.pth'
    torch.save(model.state_dict(), path)

def load_protein_model(model_class, params, experiment, protein, custom_name=None):
    path = f'models/{experiment}/{protein}_model'
    if custom_name is not None:
        path += f'_{custom_name}'
    path += '.pth'
    model = model_class(params)
    model.load_state_dict(torch.load(path))
    return model