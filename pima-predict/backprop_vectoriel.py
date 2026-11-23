import numpy as np

x  = np.array([2.0])          # (1,)
y  = 10.0
w1 = np.array([[0.5], [0.5]]) # (2×1)
b1 = np.array([0.0, 0.0])     # (2,)
w2 = np.array([[1.0, 1.0]])   # (1×2)
b2 = np.array([0.0])          # (1,)


def activate(function_name: str, x) : 
    if function_name == "Relu" :
        return np.maximum(0,x)
    else :
        return None

def derivate(function_name: str, x ):
    return None

def forward_pass(x,y,w1,w2,b1,b2):
    z1 = w1 @ x + b1 
    a1 = activate("Relu",z1)
    z2 = w2 @ a1 + b2 
    L = (z2 - y) ** 2
    
    return z1,z2,a1,L 

def backward_prop(z1,z2,a1,L,w2,y,x):
    
    return None
    


"""def train(x,y,w1,w2,b1,b2):
    
    lr = 0.001
    iterations = 5000
    for i in range(iterations):
        z1,z2,a1,L = forward_pass(x,y,w1,w2,b1,b2)
        dz2,dw2,db2,dz1,dw1,db1 = backward_prop(z1,z2,a1,L,w2,y,x)
        
        w1 -= lr * dw1
        b1 -= lr * db1
        w2 -= lr * dw2
        b2 -= lr * db2  
        
        if i % 500 == 0:
            print(f"iter {i:4d}   L = {L:.4f}")
            
    print(f"final: L = {L:.6f}   z2 = {z2:.4f}   (cible {y})")
         
    
    
train(x,y,w1,w2,b1,b2)"""
    

    
print( forward_pass(x,y,w1,w2,b1,b2))