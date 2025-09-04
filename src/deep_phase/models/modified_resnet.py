import torch
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

weight_dict = {
    'resnet18': ResNet18_Weights.IMAGENET1K_V1,
    'resnet34': ResNet34_Weights.IMAGENET1K_V1,
    'resnet50': ResNet50_Weights.IMAGENET1K_V2,
}
def build_network(
        out_classes=3, device=torch.device('cpu'),
        resnet='resnet34', freeze=True, latent=2,
        starting_network=None,
    ):
    net = torch.hub.load("pytorch/vision:v0.10.0", resnet, weights=weight_dict[resnet])

    # load previous network if provided
    if starting_network:
        state_without_fc = {
            key: value
            for key, value in torch.load(starting_network, map_location=device, weights_only=True).items()
            if not key.startswith('fc')
        }
        net.load_state_dict(state_without_fc, strict=False)

    # add encoder/decoder to last fc layer
    if isinstance(latent, int):
        net.fc = torch.nn.Sequential(
            torch.nn.Linear(net.fc.in_features, latent),
            torch.nn.Linear(latent, out_classes)
        )
    else:
        net.fc = make_fc_layer(
            latent,
            net.fc.in_features,
            out_classes,
        )
    net.type(torch.float32).to(device)

    if freeze:  # freeze all but last layer and fc layer
        for name, param in net.named_parameters():
            if "fc" not in name and "layer4" not in name:
                param.requires_grad = False

    return net

def save_network(net, path):
    net.eval()
    torch.save(net.state_dict(), path)


def load_network(net, path, device=torch.device('cpu')):
    net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return net

def make_fc_layer(fc_layers, in_features, out_features):
    layers = [torch.nn.ReLU()]
    last_features = in_features
    for fc_layer in fc_layers:
        layers += [
            torch.nn.Linear(last_features, fc_layer),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
        ]
        last_features = fc_layer
    layers += [
        torch.nn.Linear(last_features, out_features),
    ]
    return torch.nn.Sequential(*layers)

class BranchedResnetEnd(torch.nn.Module):
    def __init__(self, categories, in_features, latent=3, fc_layers=[512, 256, 128, 56], device=None):
        super(BranchedResnetEnd, self).__init__()
        top_classes = len(categories.keys())
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(in_features, latent),
            torch.nn.Linear(latent, top_classes),
        )
        self.nested_layers = torch.nn.ModuleList([
            make_fc_layer(fc_layers, in_features, len(values)).to(device) 
            for values in categories.values()
        ])

    def forward(self, x):
        top_class = self.fc(x)
        return torch.cat([
            module(x) + x_el[:, None]
            for module, x_el in zip(self.nested_layers, top_class.T)
        ], dim=1), top_class

    def __getitem__(self, item):
        # used in evaluate to determine how many hidden features are present
        if item == 0:
            return self.fc[1]  # skip dropout layer
        else:
            raise ValueError()


class BranchedResnetEndFeedback(torch.nn.Module):
    def __init__(self, categories, in_features, latent=3, fc_layers=[512, 256, 128, 56], device=None):
        super(BranchedResnetEndFeedback, self).__init__()
        top_classes = len(categories.keys())
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(in_features, latent),
            torch.nn.Linear(latent, top_classes),
        )
        self.nested_layers = torch.nn.ModuleList([
            make_fc_layer(fc_layers[:-1], in_features, fc_layers[-1]).to(device)
            for _ in categories
        ])
        self.join_layers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(fc_layers[-1] + top_classes, len(values))
            )
            for values in categories.values()
        ])

    def forward(self, x):
        top_class = self.fc(x)
        if True:  # feedback with sum
            return torch.cat([
                joining(torch.cat((nested(x), top_class), dim=1)) + x_el[:, None]
                for nested, joining, x_el in zip(self.nested_layers, self.join_layers, top_class.T)
            ], dim=1), top_class
        else:
            return torch.cat([
                joining(torch.cat((nested(x), top_class), dim=1))
                for nested, joining in zip(self.nested_layers, self.join_layers)
            ], dim=1), top_class

    def __getitem__(self, item):
        # used in evaluate to determine how many hidden features are present
        if item == 0:
            return self.fc[1]  # skip dropout layer
        else:
            raise ValueError()


def build_multiclass_network(categories, device=torch.device('cpu'),
                             resnet='resnet34', freeze=False, latent=2,
                             starting_network=None):
    top_classes = len(categories.keys())
    # get resnet size setup
    network = build_network(out_classes=top_classes, device=device,
                            resnet=resnet, freeze=freeze, latent=latent,
                            starting_network=starting_network,
                            )
    in_features = network.fc[0].in_features

    network.fc = BranchedResnetEnd(categories, in_features, device=device, latent=latent)
    
    network.type(torch.float32).to(device)
    
    return network
