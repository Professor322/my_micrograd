import numpy as np

class Value:
    # children = ()  is an empty tuple
    def __init__(self, data, op = '', children = ()):
        self.data = data
        self.grad = 0
        #underscore means that variable is for internal use
        self._op = op
        # create set out of incoming children, so we will not duplicate
        # in case of same variable used twice in the arithmetic operation i.e (x * x)
        self._children = set(children)
        self._backward = lambda : None

    def tanh(self):
        result = Value(np.tanh(self.data), 'tanh', (self, ))
        def _backward():
            self.grad += (1 - np.tanh(result.data) ** 2) * result.grad

        result._backward = _backward
        return result


    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data + other.data, '+', (self, other))
        def _backward():
            self.grad += result.grad * 1
            other.grad += result.grad * 1

        result._backward = _backward
        return result

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data * other.data, '*', (self, other))
        def _backward():
            self.grad += other.data * result.grad
            other.grad += self.data * result.grad
        result._backward = _backward
        return result

    def __pow__(self, other):
        result = Value(self.data ** other, f'**{other}', (self,))
        def _backward():
            self.grad += other * self.data ** (other - 1) * result.grad

        result._backward = _backward
        return result
    
    def __truediv__(self, other):
        return self * other ** -1
    
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -other
    
    def __rmul__(self, other):
        return self * other
    
    def __radd__(self, other):
        return self + other
        
    def __repr__(self):
        return f'Value(data:{self.data}, grad:{self.grad:.3f})'
    
    def backward(self):
        visited = set()
        topological_order = []
        def visit(value):
            if value in visited:
                return
            visited.add(value)
            for child in value._children:
                visit(child)
            topological_order.append(value)
        visit(self)
        self.grad = 1
        for val in reversed(topological_order):
            val._backward()