import queue
import torch
import torch.nn as nn

class MemoryModule(nn.Module):
    def __init__(self, dim=256, queue_size=48000, momentum=0):
        """
        memory module for domain adaptation
        """
        super().__init__()

        self.K = queue_size
        self.m = momentum

        self.queue_labels = nn.Parameter(data=torch.zeros(self.K, dtype=torch.long), requires_grad=False)
        self.queue = nn.Parameter(data=torch.zeros(self.K , dim), requires_grad=False)
        self.queue_ptr = nn.Parameter(torch.zeros(1,dtype=torch.long), requires_grad=False)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, key_labels): 

        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)

        assert self.K % batch_size == 0

        self.queue[ptr:ptr + batch_size, :] = keys
        self.queue_labels[ptr:ptr + batch_size] = key_labels
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, source_features, source_labels):

        self.key_features = source_features.detach()
        self.key_labels = source_labels.detach()

        # dequeue and enqueue
        self._dequeue_and_enqueue(self.key_features, self.key_labels)
