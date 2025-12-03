import torch


class BaseModel(torch.nn.Module):
    def load(self, path, device):
        """Load model from file.

        Args:
            path (str): file path
        """
        # parameters = torch.load(path, map_location=torch.device('cpu'))
        parameters = torch.load(path, map_location=torch.device(device))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)
