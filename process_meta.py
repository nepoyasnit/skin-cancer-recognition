import types
import torch
import torch.nn as nn
import torch.nn.functional as F

def modify_meta(mdlParams,model):
    # Define FC layers
    if len(mdlParams['fc_layers_before']) > 1:
        model.meta_before = nn.Sequential(nn.Linear(mdlParams['meta_array'].shape[1],mdlParams['fc_layers_before'][0]),
                                    nn.BatchNorm1d(mdlParams['fc_layers_before'][0]),
                                    nn.ReLU(),
                                    nn.Dropout(p=mdlParams['dropout_meta']),
                                    nn.Linear(mdlParams['fc_layers_before'][0],mdlParams['fc_layers_before'][1]),
                                    nn.BatchNorm1d(mdlParams['fc_layers_before'][1]),
                                    nn.ReLU(),
                                    nn.Dropout(p=mdlParams['dropout_meta']))
    else:
        model.meta_before = nn.Sequential(nn.Linear(mdlParams['meta_array'].shape[1],mdlParams['fc_layers_before'][0]),
                                    nn.BatchNorm1d(mdlParams['fc_layers_before'][0]),
                                    nn.ReLU(),
                                    nn.Dropout(p=mdlParams['dropout_meta']))
    # Define fc layers after
    if len(mdlParams['fc_layers_after']) > 0:
        if 'efficient' in mdlParams['model_type']:
            num_cnn_features = model._fc.in_features
        elif 'wsl' in mdlParams['model_type']:
            num_cnn_features = model.fc.in_features
        else:
            num_cnn_features = model.last_linear.in_features
        model.meta_after = nn.Sequential(nn.Linear(mdlParams['fc_layers_before'][-1]+num_cnn_features,mdlParams['fc_layers_after'][0]),
                                    nn.BatchNorm1d(mdlParams['fc_layers_after'][0]),
                                    nn.ReLU())
        classifier_in_features = mdlParams['fc_layers_after'][0]
    else:
        model.meta_after = None
        classifier_in_features = mdlParams['fc_layers_before'][-1]+model._fc.in_features
    # Modify classifier
    if 'efficient' in mdlParams['model_type']:
        model._fc = nn.Linear(classifier_in_features, mdlParams['numClasses'])
    elif 'wsl' in mdlParams['model_type']:
        model.fc = nn.Linear(classifier_in_features, mdlParams['numClasses'])
    else:
        model.last_linear = nn.Linear(classifier_in_features, mdlParams['numClasses'])
    # Modify forward pass
    def new_forward(self, inputs):
        x, meta_data = inputs
        # Normal CNN features
        if 'efficient' in mdlParams['model_type']:
            # Convolution layers
            cnn_features = self.extract_features(x)
            # Pooling and final linear layer
            cnn_features = F.adaptive_avg_pool2d(cnn_features, 1).squeeze(-1).squeeze(-1)
            if self._dropout:
                cnn_features = F.dropout(cnn_features, p=self._dropout, training=self.training)
        elif 'wsl' in mdlParams['model_type']:
            cnn_features = self.conv1(x)
            cnn_features = self.bn1(cnn_features)
            cnn_features = self.relu(cnn_features)
            cnn_features = self.maxpool(cnn_features)

            cnn_features = self.layer1(cnn_features)
            cnn_features = self.layer2(cnn_features)
            cnn_features = self.layer3(cnn_features)
            cnn_features = self.layer4(cnn_features)

            cnn_features = self.avgpool(cnn_features)
            cnn_features = torch.flatten(cnn_features, 1)
        else:
            cnn_features = self.layer0(x)
            cnn_features = self.layer1(cnn_features)
            cnn_features = self.layer2(cnn_features)
            cnn_features = self.layer3(cnn_features)
            cnn_features = self.layer4(cnn_features)
            cnn_features = self.avg_pool(cnn_features)
            if self.dropout is not None:
                cnn_features = self.dropout(cnn_features)
            cnn_features = cnn_features.view(cnn_features.size(0), -1)
        # Meta part
        #print(meta_data.shape,meta_data)
        meta_features = self.meta_before(meta_data)

        # Cat
        features = torch.cat((cnn_features,meta_features),dim=1)
        #print("features cat",features.shape)
        if self.meta_after is not None:
            features = self.meta_after(features)
        # Classifier
        if 'efficient' in mdlParams['model_type']:
            output = self._fc(features)
        elif 'wsl' in mdlParams['model_type']:
            output = self.fc(features)
        else:
            output = self.last_linear(features)
        return output
    model.forward  = types.MethodType(new_forward, model)
    return model
