import numpy as np
import matplotlib.pyplot as plt


#Gaussian distribution
def gauss(x, mean, std):
    return (1.0/np.sqrt(2*np.pi*std**2))*np.exp(-(x-mean)**2/(2*std**2))


# draw sin function
def DrawSin(x, save_name):
    # write your code here
    plt.plot(np.sin(x))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Udom Phay')
    plt.savefig(save_name)
    plt.close()
	
	
	
# draw cos function	
def DrawCos(x, save_name):
    # write your code here
    plt.plot(np.cos(x))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Udom Phay')
    plt.savefig(save_name)
    plt.close()
	
	
	
# draw multiple gaussian together on one figure
def DrawGMM(x, A, save_name):
    # write your code here	
    for key in A:
        m = (float(key))
        s = (float(A[key]))
        plt.plot(gauss(x, m, s))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Udom Phay')
    plt.savefig(save_name)
    plt.close()

	
	
	