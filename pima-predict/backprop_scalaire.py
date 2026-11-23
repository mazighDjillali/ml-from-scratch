x  = 2.0      # entrée
y  = 10.0     # cible

# poids initiaux (on les choisit, pour l'instant)
w1 = 0.5
b1 = 0.0
w2 = 1.0
b2 = 0.0


def activate(function_name: str, x:float) : 
    if function_name == "Relu" :
        return max(0,x)
    else :
        return None

def derivate(function_name: str, x: float ):
    if function_name == "Relu" :
        return 1 if x > 0 else 0
    else :
        return None

def forward_pass(x,y,w1,w2,b1,b2):
    z1 = w1 * x + b1 
    a1 = activate("Relu",z1)
    z2 = w2 * a1 + b2 
    L = (z2 - y) ** 2
    
    return z1,z2,a1,L 

def backward_prop(z1,z2,a1,L,w2,y,x):
    dz2 = 2 * (z2 - y)
    dw2 = dz2 * a1
    db2  = dz2 
    dz1  = dz2 * w2 * derivate("Relu",z1)
    dw1 = dz1 * x
    db1 = dz1 
    
    return dz2,dw2,db2,dz1,dw1,db1
    


def train(x,y,w1,w2,b1,b2):
    
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
         
    
    
train(x,y,w1,w2,b1,b2)
    

    
    