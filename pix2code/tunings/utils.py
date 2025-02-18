from typing import *
import torch.nn as nn


class InjectModule(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def handle_inputs(self, *inputs):
        return inputs
    
    def handle_outputs(self, outputs):
        return outputs
    

class HijackModule(nn.Module):

    def __init__(self, 
                 hijack_model: nn.Module, 
                 fn_handle_inputs: Optional[Callable] = None, 
                 fn_handle_output: Optional[Callable] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hijack_model = hijack_model
        self.fn_handle_inputs = fn_handle_inputs
        self.fn_handle_output = fn_handle_output

    def forward(self, *inputs):
        if self.fn_handle_inputs is not None:
            inputs = self.fn_handle_inputs(*inputs)
        outputs = self.hijack_model(*inputs)
        if self.fn_handle_output is not None:
            outputs = self.fn_handle_output(outputs)
        return outputs
    
