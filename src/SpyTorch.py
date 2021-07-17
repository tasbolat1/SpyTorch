import torch
import torch.nn.functional as F
import numpy as np

class spikeLayer(torch.nn.Module):
    '''
    This class defines main function of the SpyTorch. 
    
    **Important:** 
    All inputs must follow (Batch, Channels, Height, Width, Time) or ``NCHWT`` format.
    '''

    def __init__(self):
        super(spikeLayer, self).__init__()

        self.Ts = 1 # time step
        self.tauMem = 1 # tau synaptic
        self.tauSyn = 1 # tau membrane
        self.numSteps = 5 # length of simulated signal
        self._spike = SurrGradSpike.apply

        # calculate kernel parameters
        self.alpha = float(np.exp(-self.Ts/self.tauSyn))
        self.beta = float(np.exp(-self.Ts/self.tauMem))
        

    def dense(self, inFeatures, outFeatures, weightScale=1):
        '''
        Applies linear trasnformation to the incoming data. 
        This function behaves similar to ``torch.nn.Linear`` applied to each time instance.

        Arguments:
            * ``inFeatures`` (``int``, tuple of two ints, tuple of three ints): 
              dimension of input features (Width, Height, Channel) that represents the number of input neurons.
            * ``outFeatures`` (``int``): number of output neurons.
            * ``weightScale``: scale factor of default initialized weights. Applied to initialize weights
        Usage:
        
        >>> fc = SpyTorch.dense(64, 16)          # takes (N, 64, 1, 1, T) tensor and outputs (N, 16, 1, 1, T)
        >>> fc = SpyTorch.dense((32, 32, 2), 8) # takes (N, 2, 32, 32, T) tensor and outputs (N, 8, 1, 1, T)
        >>> output = fcl(input)                   
        '''
        return _dense(inFeatures, outFeatures, weightScale)

    def psp(self, inputs):
        '''
        This function calculates synaptic current and membrane potential for LIF neuron

        Membrane potential is not activated!
        '''
        
        B, C, H, W, _ = inputs.size()
        syn_current = torch.zeros((B,C,H,W), device=inputs.device, dtype=inputs.dtype)
        mem_current = torch.zeros((B,C,H,W), device=inputs.device, dtype=inputs.dtype)

        mem = []
        syn = []

        for t in np.arange(0, self.numSteps, self.Ts):
            syn_next = self.alpha*syn_current + inputs[...,t]
            mem_next = self.beta*mem_current + syn_current

            syn.append(syn_next)
            mem.append(mem_next)

            syn_current = syn_next
            mem_current = mem_next

        syn = torch.stack(syn, dim=-1)
        mem = torch.stack(mem, dim=-1)
        
        return syn, mem

    def spike(self, mem, threshold=0):

        spike_out = torch.zeros_like(mem)
        for t in np.arange(0, self.numSteps, self.Ts):
            mem_threshold = mem[..., t] - 1.0
            out = self._spike(mem_threshold, threshold)
            rst = out.detach()
            mem[..., t] = mem[..., t]*(1.0-rst)
            spike_out[..., t] = out

        return mem, spike_out


    def spike_lif(self, threshold=0, reccurent=False, rec_inFeatures=None, rec_outFeatures=None, rec_weight_scale=1.0):
        '''
        This function calculates synaptic current and membrane potential for LIF neuron

        Membrane potential is activated!
        '''

        if reccurent:
            return _spike_lif(threshold=0, reccurent=True, rec_inFeatures=None, rec_outFeatures=None, rec_weight_scale=1.0)
        else:
            return _spike_lif(threshold=0)


    def pool(self):
        # TODO
        pass

    def conv(self):
        # TODO
        pass

    def unpool(self):
        # TODO
        pass

    def axonal_delay(self):
        # TODO
        pass

class _spike_lif(torch.nn.Module):
    def __init__(self,  threshold=0, reccurent=False, rec_inFeatures=None, rec_outFeatures=None, rec_weight_scale=1.0):
        super(_spike_lif, self).__init__()

        self.threshold = threshold
        self.reccurrent = reccurent
        
        # add reccurrent weight
        if self.reccurrent:

            # check that these are assigned for reccurency
            assert (rec_inFeatures is not None) and (rec_outFeatures is not None), "please describe recurrent weight"

            self.rec_weight = torch.nn.Parameter(torch.Tensor(rec_inFeatures, rec_outFeatures), requires_grad=True)
            torch.nn.init.normal_(self.rec_weight, mean=0.0, std=rec_weight_scale/np.sqrt(rec_outFeatures))

    def forward(self, inputs):

        B, C, H, W, _ = inputs.size()
        syn_current = torch.zeros((B,C,H,W), device=inputs.device, dtype=inputs.dtype)
        mem_current = torch.zeros((B,C,H,W), device=inputs.device, dtype=inputs.dtype)

        mem = []
        syn = []
        spike_out = []

        for t in np.arange(0, self.numSteps, self.Ts):
            
            # activate neuron
            mem_threshold = mem - 1.0
            out = self._spike(mem_threshold, self.threshold)

            # calculate synaptic current and membrane potential
            syn_next = self.alpha*syn_current + inputs[...,t]

            # add reccurence if needed
            if self.reccurrent:
                syn_next = torch.einsum("abcd,be->aecd", syn_next, self.rec_weight)
            
            mem_next = (self.beta*mem_current + syn_current)*(1.0-out.detach())
            
            syn.append(syn_next)
            mem.append(mem_next)
            spike_out.append(out)

            syn_current = syn_next
            mem_current = mem_next

        syn = torch.stack(syn, dim=-1)
        mem = torch.stack(mem, dim=-1)
        spike_out = torch.stack(spike_out, dim=-1)

        return syn, mem, spike_out


class _dense(torch.nn.Conv3d):
    def __init__(self, inFeatures, outFeatures, weightScale):

        if type(inFeatures) == int:
            kernel = (1, 1, 1)
            inChannels = inFeatures 
        elif len(inFeatures) == 2:
            kernel = (inFeatures[1], inFeatures[0], 1)
            inChannels = 1
        elif len(inFeatures) == 3:
            kernel = (inFeatures[1], inFeatures[0], 1)
            inChannels = inFeatures[2]
        else:
            raise Exception('inFeatures should not be more than 3 dimension. It was: {}'.format(inFeatures.shape))
        
        if type(outFeatures) == int:
            outChannels = outFeatures
        else:
            raise Exception('outFeatures should not be more than 1 dimesnion. It was: {}'.format(outFeatures.shape))
        
        super(_dense, self).__init__(inChannels, outChannels, kernel, bias=False)

        # create learnable weights and initialize them from Normal distribution
        self.weight = torch.nn.Parameter(self.weight)
        torch.nn.init.normal_(self.weight, mean=0.0, std=weightScale/np.sqrt(outFeatures))

    def forward(self, input):
        return F.conv3d(input, 
                        self.weight, self.bias, self.stride, 
                        self.padding, self.dilation, self.groups)

    
class SurrGradSpike(torch.autograd.Function):
    """
    This class implements neuron's ``spike`` function along with backpropogation 
    using surrogate gradient. The forward function is simply threshold nonlinear function
    while backpropogation is the normalized negative part of a past sigmoid.
    For more information, please refer Zenke & Ganguli (2018).
    """
    def __init__(self, scale=100):
        # controls steepness of surrogate gradient
        scale = scale # ? Forward and Backward functions are static, thus no need self.

    @staticmethod
    def forward(ctx, input, threshold=0):
        """
        This 

        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which 
        we need to later backpropagate our error signals. To achieve this we use the 
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > threshold] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the 
        surrogate gradient of the loss with respect to the input. 
        Here we use the normalized negative part of a fast sigmoid 
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SurrGradSpike.scale*torch.abs(input)+1.0)**2
        return grad