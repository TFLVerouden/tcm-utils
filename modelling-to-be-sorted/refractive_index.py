"""
Calculating the refractive index of solutions using Lorentz-Lorentz
"""

import numpy as np
def Lorentz_Lorentz(n1=1.33,n2=1.4539,phi1=0.99):
    x = phi1 * (n1**2-1)/(n1**2+2) + (1-phi1) * (n2**2-1)/(n2**2+2)
    
    n12 = np.sqrt((-1-2*x)/(x-1))
    return x,n12

x,n12 = Lorentz_Lorentz(phi1=0.997)
print(x,n12)
