from .df_loader import DataLoader
from .serialization import Serialization

serializer = Serialization()
dataloader = DataLoader(serializer)