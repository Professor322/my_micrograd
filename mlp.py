import random
import Value

class Module:
    def parameters(self):
        return []
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0


class Neuron(Module):
    # Neuron will expect:
    # - how many inputs to the single neuron (n_in)
    def __init__(self, n_in):
        self.w = []
        for w in range(0, n_in):
            self.w.append(Value(random.uniform(-1, 1)))
        self.b = Value(random.uniform(-1, 1))
    
    def __call__(self, x):
        res = Value(self.b.data)
        for i in range(0, len(x)):
            res = res + self.w[i] * x[i]
        return res.tanh()
    
    def parameters(self):
        return self.w + [self.b]
    

class Layer(Module):
    # Layer will expect:
    # - how many inputs to the single neuron (n_in)
    # - how many neurons to create in the layer (n_out)
    def __init__(self, n_in, n_out):
        self.neurons = []
        for _ in range(0, n_out):
            self.neurons.append(Neuron(n_in))

    def __call__(self, x):
        res = []
        for neuron in self.neurons:
            res.append(neuron(x))
        return res
    
    def parameters(self):
        res = []
        for neuron in self.neurons:
            res.extend(neuron.parameters())
        return res


class MLP(Module):
    # MLP will expect:
    #  - how many inputs to the single neuron n_in
    #  - vector of layers length n_out
    def __init__(self, n_in, n_out):
        layer_sizes = [n_in] + n_out
        self.layers = []
        for i in range(0, len(n_out)):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1]))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        res = []
        for layer in self.layers:
            res.extend(layer.parameters())
        return res
        