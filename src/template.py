import math
import torch
from torch import Tensor
from torch.nn.parameter import Parameter #, UninitializedParameter
from torch.nn import functional as F
from torch.nn import init
from torch.nn.modules import Module
class nD_Linear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
        This module supports :ref:`TensorFloat32<tf32_on_ampere>`.
        Args:
            tensor_features: size of first dimension of tensor
            in_features: size of each input sample
            out_features: size of each output sample
            bias: If set to ``False``, the layer will not learn an additive bias.
                Default: ``True``
        Shape:
            - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
              additional dimensions and :math:`H_{in} = \text{in\_features}`
            - Output: :math:`(N, tensor_feature, *, H_{out})` where all but the last dimension
              are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
        Attributes:
            weight: the learnable weights of the module of shape
                :math:`(\text{out\_features}, \text{in\_features})`. The values are
                initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
                :math:`k = \frac{1}{\text{in\_features}}`
            bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                    If :attr:`bias` is ``True``, the values are initialized from
                    :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                    :math:`k = \frac{1}{\text{in\_features}}`
        """
    __constants__ = ['in_features', 'out_features']
    tensor_features: int
    in_features: int
    out_features: int
    weight: Tensor
    def __init__(self, tensor_features: int, in_features: int, out_features: int, bias: bool = True) -> None:
        super(nD_Linear, self).__init__()
        self.tensor_features = tensor_features
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(tensor_features, out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(tensor_features, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
    def forward(self, input: Tensor) -> Tensor:
        return (self.weight@input.T).permute(2,0,1)+ self.bias
    def extra_repr(self) -> str:
        return 'tensor_features = {}, in_features={}, out_features={}, bias={}'.format(
            self.tensor_features, self.in_features, self.out_features, self.bias is not None
        )
if __name__ == '__main__':
    nz = 3
    nu = 2
    nz1 = nz+1
    model = nD_Linear(nu, nz, nz1)
    num_forward = 7
    actions = [int(torch.rand(1) > 0.5) for i in range(num_forward)]
    z0_in_data = torch.ones(num_forward, nz)
    z1_label = 2*torch.ones(num_forward, nz1)
    z1_est = model(z0_in_data)
    def loss(out_data, label, action_indexes):
        # example of only evaluating loss on some of the indexes. Probably want to vectorize
        return torch.sum((out_data[torch.arange(len(action_indexes)), action_indexes, :] - label)**2)
    loss_value = loss(z1_est, z1_label, actions)
    print(loss_value)