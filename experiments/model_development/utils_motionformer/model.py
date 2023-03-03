"""
model.py: contains the main CNNLSTM custom model.
"""
import torch
import torch.nn as nn
import torchvision.models as models

# Torchvision existing models
CNN_MODELS = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "alexnet": models.alexnet,
}

# Output sizes of each supported CNN model
CNN_OUTPUT_SIZES = {
    "resnet18": 512,
    "resnet34": 512,
    "alexnet": 4096,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Identity(nn.Module):
    """
    An Identity neural network block.
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """
        Forward pass of an Identity block block.

        :param Tensor x: any input.
        :return: the same input.
        :rtype: Tensor
        """
        return x

class LSTMBlock(nn.Module):
    """
    An LSTM block consisting of bidriectional RNNs.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        """
        Creates a bidirectional LSTM block

        :param int input_size: the input size to the LSTM.
        :param int hidden_size: the hidden size of an LSTM layer.
        :param int num_layers: the number of layers in the LSTM.
        :param int num_classes: the number of output classes.
        """
        super(LSTMBlock, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        """
        Forward pass of an LSTM block.

        :param Tensor x: Input videos, should be of shape [batch_size, frames_per_video, input_size].
        :return: predictions of shape [batch_size, num_classes].
        :rtype: Tensor
        """
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class CNNLSTM(nn.Module):
    """
    Wrapper around CNNs with the FC layer replaced by an LSTM block.
    """
    def __init__(self, lstm_hidden_size, lstm_num_layers, num_classes, cnn_architecture="resnet18", pretrained=True):
        """
        Loads the appropriate CNN model, disables the output layer, and adds an LSTM block.

        :param int lstm_hidden_size: the hidden size of an LSTM layer.
        :param int lstm_num_layers: the number of layers in the outptu LSTM.
        :param int num_classes: the number of output classes.
        :param str cnn_architecture: the name of the particular CNN architecture.
        :param bool pretrained: whether to load a pretrained CNN or not.
        """
        super(CNNLSTM, self).__init__()
        self.cnn = CNN_MODELS[cnn_architecture](pretrained=pretrained)
        set_last_identity(self.cnn, cnn_architecture)
        self.lstm = LSTMBlock(CNN_OUTPUT_SIZES[cnn_architecture], lstm_hidden_size, lstm_num_layers, num_classes)

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN model and the LSTM block.

        :param Tensor videos: Input videos, should be of shape [batch_size, frames_per_video, num_channels, width, height].
        :return: predictions of shape [batch_size, num_classes].
        :rtype: Tensor
        """
        batch_size, timesteps, C, H, W = videos.size()
        c_in = videos.view(batch_size*timesteps, C, H, W)  
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out = self.lstm(r_in)
        return r_out

def set_last_identity(model, cnn_architecture):
    """
    Updates a CNN model's last layer with an Identity layer.

    :param nn.Module cnn_model: an instantiated CNN model whose last layer is to be replaced.
    :param string cnn_architecture: the name of the model, must one of the CNN_MODELS.
    """

    if "resnet" in cnn_architecture:
        model.fc = Identity()
    elif "alexnet" in cnn_architecture:
        model.classifier[-1] = Identity()
    else:
        pass

if __name__ == "__main__":
    # Loads a Resnet18 + LSTM model
    cnn_architecture = "resnet18"
    model = CNNLSTM(512, 2, 3, cnn_architecture=cnn_architecture)

    # Shows a description of the CNNLSTM model architecture
    from torchinfo import summary
    summary(model, input_size=(1, 205, 3, 224, 224))