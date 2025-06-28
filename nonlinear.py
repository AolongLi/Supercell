import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import time
from joblib import Parallel,delayed
from collections.abc import Callable
import os

def zero_delta(E,Ef,gamma):
    return (1/np.pi)*(gamma/((E-Ef)**2+gamma**2))

def zero_fd(E,Ef):
    f=((E-Ef)<0)+0
    return f

def DF_dis(E,Ef,gamma):
    f=1/(np.exp((E-Ef)/(gamma))+1)
    return f

def shiftxyy(omega: float, kvecs: np.ndarray, hamiltonian: Callable[[np.ndarray], np.ndarray], lambdaa: float, Sint: float, Ef: float, gamma: float, energy_scale: float, use_parallel: bool = False, n_jobs: int = -1) -> float:
    
    # kvecs is a N*2 array, where N is the number of k-vectors
    kvecs_length=kvecs.shape[0]
    ones_kvecs_length=np.ones((kvecs_length,1))

    # Generate the offset of point k for calculating the derivative
    dkx0=np.array([[lambdaa,0]])
    dky0=np.array([[0,lambdaa]])
    dkxdky0=np.array([[lambdaa,lambdaa]])
    dkx=np.kron(dkx0,ones_kvecs_length)
    dky=np.kron(dky0,ones_kvecs_length)
    dkxdky=np.kron(dkxdky0,ones_kvecs_length)

    # Generate the Hamiltonian for the shifted k-vectors
    Htotal = hamiltonian(kvecs)
    Htotal_zx = hamiltonian(kvecs+dkx)
    Htotal_fx = hamiltonian(kvecs-dkx)
    Htotal_zy = hamiltonian(kvecs+dky)
    Htotal_fy = hamiltonian(kvecs-dky)
    Htotal_zxy = hamiltonian(kvecs+dkxdky)
    Htotal_fxy = hamiltonian(kvecs-dkxdky)

    # solve the eigenvalue problem
    evalue,evector=LA.eigh(Htotal)
    evalue_zx,evector_zx=LA.eigh(Htotal_zx)
    evalue_fx,evector_fx=LA.eigh(Htotal_fx)
    evalue_zy,evector_zy=LA.eigh(Htotal_zy)
    evalue_fy,evector_fy=LA.eigh(Htotal_fy)
    evalue_zxy,evector_zxzy=LA.eigh(Htotal_zxy)
    evalue_fxy,evector_fxfy=LA.eigh(Htotal_fxy)

    basissize=evalue.shape[1]

    # Calculate the skwness tensor
    Quantum_tensor=np.zeros((kvecs_length,basissize,basissize),dtype='float')
    
    # 定义处理单个k点的函数
    def process_single_k(k):
        if k==0:
            start=time.time()
        if k%100==0:
            print('k=%d/%d'%(k,kvecs_length))
            
        k_tensor = np.zeros((basissize, basissize), dtype='float')
        for m in range(basissize):
            for n in range(basissize):
                if m<n:
                    vn=evector[k,:,n]
                    vzy_n=evector_zy[k,:,n]
                    vfy_n=evector_fy[k,:,n]
                    vzx_n=evector_zx[k,:,n]
                    vfx_n=evector_fx[k,:,n]
                    vzxy_n=evector_zxzy[k,:,n]
                    vfxy_n=evector_fxfy[k,:,n]

                    vzy_m=evector_zy[k,:,m]
                    vfy_m=evector_fy[k,:,m]
                    vzx_m=evector_zx[k,:,m]
                    vfx_m=evector_fx[k,:,m]

                    a1=np.dot(vn.conj(),vzx_m)
                    a2=np.dot(vn.conj(),vfx_m)

                    term1_1=(a1*np.dot(vzx_m.conj(),vzxy_n)-a2*np.dot(vfx_m.conj(),vzxy_n))*np.dot(vzxy_n.conj(),vn)
                    term1_2=(-a1*np.dot(vzx_m.conj(),vzx_n)+a2*np.dot(vfx_m.conj(),vzx_n))*np.dot(vzx_n.conj(),vn)
                    term1_3=(-a1*np.dot(vzx_m.conj(),vzy_n)+a2*np.dot(vfx_m.conj(),vzy_n))*np.dot(vzy_n.conj(),vn)
                    term1_4=2*a1*np.dot(vzx_m.conj(),vn)-2*a2*np.dot(vfx_m.conj(),vn)
                    term1_5=(-a1*np.dot(vzx_m.conj(),vfx_n)+a2*np.dot(vfx_m.conj(),vfx_n))*np.dot(vfx_n.conj(),vn)
                    term1_6=(-a1*np.dot(vzx_m.conj(),vfy_n)+a2*np.dot(vfx_m.conj(),vfy_n))*np.dot(vfy_n.conj(),vn)
                    term1_7=(a1*np.dot(vzx_m.conj(),vfxy_n)-a2*np.dot(vfx_m.conj(),vfxy_n))*np.dot(vfxy_n.conj(),vn)
                    term1=(1/(4*lambdaa**3))*(term1_1+term1_2+term1_3+term1_4+term1_5+term1_6+term1_7)
                    
                    
                    b1=np.dot(vn.conj(),vzx_m)
                    b2=np.dot(vn.conj(),vfx_m)
                    b3=np.dot(vzx_n.conj(),vn)
                    b4=np.dot(vfx_n.conj(),vn)
                    b5=np.dot(vzx_m.conj(),vzy_m)
                    b6=np.dot(vzx_m.conj(),vfy_m)
                    b7=np.dot(vfx_m.conj(),vzy_m)
                    b8=np.dot(vfx_m.conj(),vfy_m)
                    b9=np.dot(vzy_m.conj(),vfx_n)
                    b10=np.dot(vzy_m.conj(),vzx_n)
                    b11=np.dot(vfy_m.conj(),vzx_n)
                    b12=np.dot(vfy_m.conj(),vfx_n)

                    term2=(1/(8*lambdaa**3))*(b1*b5*b10*b3-b1*b5*b9*b4-b1*b6*b11*b3+b1*b6*b12*b4\
                        -b2*b7*b10*b3+b2*b7*b9*b4+b2*b8*b11*b3-b2*b8*b12*b4)
                    
                    k_tensor[m,n]=np.imag(term1+term2)
        
        if k==0:
            end=time.time()
            print('Each k cost time %f s'%(end-start))
            
        return k, k_tensor

    # 根据use_parallel选择串行或并行处理
    if use_parallel:
        print(f"使用并行计算，n_jobs={n_jobs}")
        # 并行处理所有k点
        results = Parallel(n_jobs=n_jobs)(delayed(process_single_k)(k) for k in range(kvecs_length))
        # 将结果组装到Quantum_tensor中
        for k, k_tensor in results:
            Quantum_tensor[k] = k_tensor
    else:
        # 串行处理
        for k in range(kvecs_length):
            _, k_tensor = process_single_k(k)
            Quantum_tensor[k] = k_tensor
    
    skewness_tensor=Quantum_tensor-Quantum_tensor.transpose(0,2,1)

    # calculate delta energy
    delta_E = evalue.reshape(kvecs_length,basissize,1) - evalue.reshape(kvecs_length,1,basissize)

    # calculate the Fermi-Dirac distribution
    fermi=DF_dis(evalue,Ef,gamma)
    delta_fermi = fermi.reshape(kvecs_length,basissize,1) - fermi.reshape(kvecs_length,1,basissize)

    delt_func = zero_delta(omega, delta_E, gamma)

    term1=skewness_tensor[:,basissize//2:basissize,:basissize//2]*\
        delta_fermi[:,basissize//2:basissize,:basissize//2]\
            *delt_func[:,basissize//2:basissize,:basissize//2]

    term2=skewness_tensor[:,:basissize//2,basissize//2:basissize]*\
        delta_fermi[:,:basissize//2,basissize//2:basissize]\
            *delt_func[:,:basissize//2,basissize//2:basissize]
    term=term1+term2

    mesh = (Sint)*term.sum(axis=(1,2))*(energy_scale)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(kvecs[:, 0], kvecs[:, 1], c=mesh, cmap='viridis', 
                        s=50, alpha=0.8, edgecolors='none')
    plt.colorbar(scatter, label='shift current')
    plt.xlabel('kx')
    plt.ylabel('ky')
    plt.title(f'omega = {omega:.3f}')
    plt.tight_layout()

    # 创建保存图像的目录
    save_dir = f"image/lambda_{lambdaa:.3f}_gamma_{gamma:.3f}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存图像
    plt.savefig(f'{save_dir}/kvecs_mesh_scatter_xyy_omega_{omega:.3f}.png', dpi=300)
    plt.close()

    result = (Sint)*np.sum(term)*(energy_scale)

    return result



def shiftyyy(omega: float, kvecs: np.ndarray, hamiltonian: Callable[[np.ndarray], np.ndarray], lambdaa: float, Sint: float, Ef: float, gamma: float, energy_scale: float, use_parallel: bool = False, n_jobs: int = -1) -> float:
    
    # kvecs is a N*2 array, where N is the number of k-vectors
    kvecs_length=kvecs.shape[0]
    ones_kvecs_length=np.ones((kvecs_length,1))

    # Generate the offset of point k for calculating the derivative
    dky0=np.array([[0,lambdaa]])
    dky=np.kron(dky0,ones_kvecs_length)

    # Generate the Hamiltonian for the shifted k-vectors
    Htotal = hamiltonian(kvecs)
    Htotal_zy = hamiltonian(kvecs+dky)
    Htotal_fy = hamiltonian(kvecs-dky)


    # solve the eigenvalue problem
    evalue,evector=LA.eigh(Htotal)
    evalue_zy,evector_zy=LA.eigh(Htotal_zy)
    evalue_fy,evector_fy=LA.eigh(Htotal_fy)

    basissize=evalue.shape[1]

    # Calculate the skwness tensor
    Quantum_tensor=np.zeros((kvecs_length,basissize,basissize),dtype='complex')

    # 定义处理单个k点的函数
    def process_single_k(k):
        if k==1:
            start=time.time()
        if k%2000==0:
            print('k=%d/%d'%(k,kvecs_length))
            
        k_tensor = np.zeros((basissize, basissize), dtype='complex')
        for m in range(basissize//2):
            for n in range(basissize//2-1,basissize):
                # if m<n:
                    
                vn=evector[k,:,n]
                vzy_m=evector_zy[k,:,m]
                vfy_m=evector_fy[k,:,m]
                vzy_n=evector_zy[k,:,n]
                vfy_n=evector_fy[k,:,n]

                # part_a
                a1=np.dot(vn.conj(),vzy_m)
                a2=np.dot(vn.conj(),vfy_m)
                a3=np.dot(vzy_n.conj(),vn)
                a4=np.dot(vfy_n.conj(),vn)

                a5=np.dot(vzy_m.conj(),vzy_n)
                a6=np.dot(vzy_m.conj(),vfy_n)
                a7=np.dot(vfy_m.conj(),vzy_n)
                a8=np.dot(vfy_m.conj(),vfy_n)

                b1=np.dot(vzy_m.conj(),vfy_m)

                re1=a1*a5*a3
                re2=a1*a6*a4
                re3=a2*a7*a3
                re4=a2*a8*a4

                term1=(1/(2*lambdaa**3))*(re1-2*a1*a1.conj()+re2-re3+2*a2*a2.conj()-re4)
                
                term2_1=re1-re2-a1*b1*a7*a3+a1*b1*a8*a4
                term2_2=-a2*b1.conj()*a5*a3+a2*b1.conj()*a6*a4+re3-re4

                term2=(1/(8*lambdaa**3))*(term2_1+term2_2)
                
                k_tensor[m,n]=np.imag(term1+term2)
        
        if k==1:
            end=time.time()
            print('each k cost time %f s'%(end-start))
            
        return k, k_tensor

    # 根据use_parallel选择串行或并行处理
    if use_parallel:
        print(f"使用并行计算，n_jobs={n_jobs}")
        # 并行处理所有k点
        results = Parallel(n_jobs=n_jobs)(delayed(process_single_k)(k) for k in range(kvecs_length))
        # 将结果组装到Quantum_tensor中
        for k, k_tensor in results:
            Quantum_tensor[k] = k_tensor
    else:
        # 串行处理
        for k in range(kvecs_length):
            _, k_tensor = process_single_k(k)
            Quantum_tensor[k] = k_tensor

    skewness_tensor=(Quantum_tensor-Quantum_tensor.transpose(0,2,1))

    # calculate delta energy
    delta_E = evalue.reshape(kvecs_length,basissize,1) - evalue.reshape(kvecs_length,1,basissize)

    # calculate the Fermi-Dirac distribution
    fermi=DF_dis(evalue,Ef,gamma)
    delta_fermi = fermi.reshape(kvecs_length,basissize,1) - fermi.reshape(kvecs_length,1,basissize)

    delt_func = zero_delta(omega, delta_E, gamma)

    term1=skewness_tensor[:,basissize//2:basissize,:basissize//2]*\
        delta_fermi[:,basissize//2:basissize,:basissize//2]*\
            delt_func[:,basissize//2:basissize,:basissize//2]

    term2=skewness_tensor[:,:basissize//2,basissize//2:basissize]*\
        delta_fermi[:,:basissize//2,basissize//2:basissize]*\
            delt_func[:,:basissize//2,basissize//2:basissize]
    term=term1+term2

    mesh = (Sint)*term.sum(axis=(1,2))*(energy_scale)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(kvecs[:, 0], kvecs[:, 1], c=mesh, cmap='viridis', 
                        s=50, alpha=0.8, edgecolors='none')
    plt.colorbar(scatter, label='shift current')
    plt.xlabel('kx')
    plt.ylabel('ky')
    plt.title(f'omega = {omega:.3f}')
    plt.tight_layout()

    # 创建保存图像的目录
    save_dir = f"image/lambda_{lambdaa:.3f}_gamma_{gamma:.3f}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存图像
    plt.savefig(f'{save_dir}/kvecs_mesh_scatter_yyy_omega_{omega:.3f}.png', dpi=300)
    plt.close()

    result = (Sint)*np.sum(term)*(energy_scale)

    return result



def shiftxxx(omega: float, kvecs: np.ndarray, hamiltonian: Callable[[np.ndarray], np.ndarray], lambdaa: float, Sint: float, Ef: float, gamma: float, energy_scale: float, use_parallel: bool = False, n_jobs: int = -1) -> float:
    
    # kvecs is a N*2 array, where N is the number of k-vectors
    kvecs_length=kvecs.shape[0]
    ones_kvecs_length=np.ones((kvecs_length,1))

    # Generate the offset of point k for calculating the derivative
    dkx0=np.array([[lambdaa,0]])
    dkx=np.kron(dkx0,ones_kvecs_length)

    # Generate the Hamiltonian for the shifted k-vectors
    Htotal = hamiltonian(kvecs)
    Htotal_zx = hamiltonian(kvecs+dkx)
    Htotal_fx = hamiltonian(kvecs-dkx)


    # solve the eigenvalue problem
    evalue,evector=LA.eigh(Htotal)
    evalue_zx,evector_zx=LA.eigh(Htotal_zx)
    evalue_fx,evector_fx=LA.eigh(Htotal_fx)

    basissize=evalue.shape[1]

    # Calculate the skwness tensor
    Quantum_tensor=np.zeros((kvecs_length,basissize,basissize),dtype='complex')

    # 定义处理单个k点的函数
    def process_single_k(k):
        if k==1:
            start=time.time()
        if k%2000==0:
            print('k=%d/%d'%(k,kvecs_length))
            
        k_tensor = np.zeros((basissize, basissize), dtype='complex')
        for m in range(basissize//2):
            for n in range(basissize//2-1,basissize):
                # if m<n:
                    
                vn=evector[k,:,n]
                vzx_m=evector_zx[k,:,m]
                vfx_m=evector_fx[k,:,m]
                vzx_n=evector_zx[k,:,n]
                vfx_n=evector_fx[k,:,n]

                # part_a
                a1=np.dot(vn.conj(),vzx_m)
                a2=np.dot(vn.conj(),vfx_m)
                a3=np.dot(vzx_n.conj(),vn)
                a4=np.dot(vfx_n.conj(),vn)

                a5=np.dot(vzx_m.conj(),vzx_n)
                a6=np.dot(vzx_m.conj(),vfx_n)
                a7=np.dot(vfx_m.conj(),vzx_n)
                a8=np.dot(vfx_m.conj(),vfx_n)

                b1=np.dot(vzx_m.conj(),vfx_m)

                re1=a1*a5*a3
                re2=a1*a6*a4
                re3=a2*a7*a3
                re4=a2*a8*a4

                term1=(1/(2*lambdaa**3))*(re1-2*a1*a1.conj()+re2-re3+2*a2*a2.conj()-re4)
                
                term2_1=re1-re2-a1*b1*a7*a3+a1*b1*a8*a4
                term2_2=-a2*b1.conj()*a5*a3+a2*b1.conj()*a6*a4+re3-re4

                term2=(1/(8*lambdaa**3))*(term2_1+term2_2)
                
                k_tensor[m,n]=np.imag(term1+term2)
        
        if k==1:
            end=time.time()
            print('each k cost time %f s'%(end-start))
            
        return k, k_tensor

    # 根据use_parallel选择串行或并行处理
    if use_parallel:
        print(f"使用并行计算，n_jobs={n_jobs}")
        # 并行处理所有k点
        results = Parallel(n_jobs=n_jobs)(delayed(process_single_k)(k) for k in range(kvecs_length))
        # 将结果组装到Quantum_tensor中
        for k, k_tensor in results:
            Quantum_tensor[k] = k_tensor
    else:
        # 串行处理
        for k in range(kvecs_length):
            _, k_tensor = process_single_k(k)
            Quantum_tensor[k] = k_tensor

    skewness_tensor=(Quantum_tensor-Quantum_tensor.transpose(0,2,1))

    # calculate delta energy
    delta_E = evalue.reshape(kvecs_length,basissize,1) - evalue.reshape(kvecs_length,1,basissize)

    # calculate the Fermi-Dirac distribution
    fermi=DF_dis(evalue,Ef,gamma)
    delta_fermi = fermi.reshape(kvecs_length,basissize,1) - fermi.reshape(kvecs_length,1,basissize)

    delt_func = zero_delta(omega, delta_E, gamma)

    term1=skewness_tensor[:,basissize//2:basissize,:basissize//2]*\
        delta_fermi[:,basissize//2:basissize,:basissize//2]*\
            delt_func[:,basissize//2:basissize,:basissize//2]

    term2=skewness_tensor[:,:basissize//2,basissize//2:basissize]*\
        delta_fermi[:,:basissize//2,basissize//2:basissize]*\
            delt_func[:,:basissize//2,basissize//2:basissize]
    term=term1+term2

    mesh = (Sint)*term.sum(axis=(1,2))*(energy_scale)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(kvecs[:, 0], kvecs[:, 1], c=mesh, cmap='viridis', 
                        s=50, alpha=0.8, edgecolors='none')
    plt.colorbar(scatter, label='shift current')
    plt.xlabel('kx')
    plt.ylabel('ky')
    plt.title(f'omega = {omega:.3f}')
    plt.tight_layout()

    # 创建保存图像的目录
    save_dir = f"image/lambda_{lambdaa:.3f}_gamma_{gamma:.3f}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存图像
    plt.savefig(f'{save_dir}/kvecs_mesh_scatter_xxx_omega_{omega:.3f}.png', dpi=300)
    plt.close()

    result = (Sint)*np.sum(term)*(energy_scale)

    return result



def shiftyxx(omega: float, kvecs: np.ndarray, hamiltonian: Callable[[np.ndarray], np.ndarray], lambdaa: float, Sint: float, Ef: float, gamma: float, energy_scale: float, use_parallel: bool = False, n_jobs: int = -1) -> float:
    
    # kvecs is a N*2 array, where N is the number of k-vectors
    kvecs_length=kvecs.shape[0]
    ones_kvecs_length=np.ones((kvecs_length,1))

    # Generate the offset of point k for calculating the derivative
    # for simplicity, we exchange the x and y axis
    dkx0=np.array([[0,lambdaa]])
    dky0=np.array([[lambdaa,0]])
    dkxdky0=np.array([[lambdaa,lambdaa]])
    dkx=np.kron(dkx0,ones_kvecs_length)
    dky=np.kron(dky0,ones_kvecs_length)
    dkxdky=np.kron(dkxdky0,ones_kvecs_length)

    # Generate the Hamiltonian for the shifted k-vectors
    Htotal = hamiltonian(kvecs)
    Htotal_zx = hamiltonian(kvecs+dkx)
    Htotal_fx = hamiltonian(kvecs-dkx)
    Htotal_zy = hamiltonian(kvecs+dky)
    Htotal_fy = hamiltonian(kvecs-dky)
    Htotal_zxy = hamiltonian(kvecs+dkxdky)
    Htotal_fxy = hamiltonian(kvecs-dkxdky)

    # solve the eigenvalue problem
    evalue,evector=LA.eigh(Htotal)
    evalue_zx,evector_zx=LA.eigh(Htotal_zx)
    evalue_fx,evector_fx=LA.eigh(Htotal_fx)
    evalue_zy,evector_zy=LA.eigh(Htotal_zy)
    evalue_fy,evector_fy=LA.eigh(Htotal_fy)
    evalue_zxy,evector_zxzy=LA.eigh(Htotal_zxy)
    evalue_fxy,evector_fxfy=LA.eigh(Htotal_fxy)

    basissize=evalue.shape[1]

    # Calculate the skwness tensor
    Quantum_tensor=np.zeros((kvecs_length,basissize,basissize),dtype='float')
    
    # 定义处理单个k点的函数
    def process_single_k(k):
        if k==0:
            start=time.time()
        if k%100==0:
            print('k=%d/%d'%(k,kvecs_length))
            
        k_tensor = np.zeros((basissize, basissize), dtype='float')
        for m in range(basissize):
            for n in range(basissize):
                if m<n:
                    vn=evector[k,:,n]
                    vzy_n=evector_zy[k,:,n]
                    vfy_n=evector_fy[k,:,n]
                    vzx_n=evector_zx[k,:,n]
                    vfx_n=evector_fx[k,:,n]
                    vzxy_n=evector_zxzy[k,:,n]
                    vfxy_n=evector_fxfy[k,:,n]

                    vzy_m=evector_zy[k,:,m]
                    vfy_m=evector_fy[k,:,m]
                    vzx_m=evector_zx[k,:,m]
                    vfx_m=evector_fx[k,:,m]

                    a1=np.dot(vn.conj(),vzx_m)
                    a2=np.dot(vn.conj(),vfx_m)

                    term1_1=(a1*np.dot(vzx_m.conj(),vzxy_n)-a2*np.dot(vfx_m.conj(),vzxy_n))*np.dot(vzxy_n.conj(),vn)
                    term1_2=(-a1*np.dot(vzx_m.conj(),vzx_n)+a2*np.dot(vfx_m.conj(),vzx_n))*np.dot(vzx_n.conj(),vn)
                    term1_3=(-a1*np.dot(vzx_m.conj(),vzy_n)+a2*np.dot(vfx_m.conj(),vzy_n))*np.dot(vzy_n.conj(),vn)
                    term1_4=2*a1*np.dot(vzx_m.conj(),vn)-2*a2*np.dot(vfx_m.conj(),vn)
                    term1_5=(-a1*np.dot(vzx_m.conj(),vfx_n)+a2*np.dot(vfx_m.conj(),vfx_n))*np.dot(vfx_n.conj(),vn)
                    term1_6=(-a1*np.dot(vzx_m.conj(),vfy_n)+a2*np.dot(vfx_m.conj(),vfy_n))*np.dot(vfy_n.conj(),vn)
                    term1_7=(a1*np.dot(vzx_m.conj(),vfxy_n)-a2*np.dot(vfx_m.conj(),vfxy_n))*np.dot(vfxy_n.conj(),vn)
                    term1=(1/(4*lambdaa**3))*(term1_1+term1_2+term1_3+term1_4+term1_5+term1_6+term1_7)
                    
                    
                    b1=np.dot(vn.conj(),vzx_m)
                    b2=np.dot(vn.conj(),vfx_m)
                    b3=np.dot(vzx_n.conj(),vn)
                    b4=np.dot(vfx_n.conj(),vn)
                    b5=np.dot(vzx_m.conj(),vzy_m)
                    b6=np.dot(vzx_m.conj(),vfy_m)
                    b7=np.dot(vfx_m.conj(),vzy_m)
                    b8=np.dot(vfx_m.conj(),vfy_m)
                    b9=np.dot(vzy_m.conj(),vfx_n)
                    b10=np.dot(vzy_m.conj(),vzx_n)
                    b11=np.dot(vfy_m.conj(),vzx_n)
                    b12=np.dot(vfy_m.conj(),vfx_n)

                    term2=(1/(8*lambdaa**3))*(b1*b5*b10*b3-b1*b5*b9*b4-b1*b6*b11*b3+b1*b6*b12*b4\
                        -b2*b7*b10*b3+b2*b7*b9*b4+b2*b8*b11*b3-b2*b8*b12*b4)
                    
                    k_tensor[m,n]=np.imag(term1+term2)
        
        if k==0:
            end=time.time()
            print('Each k cost time %f s'%(end-start))
            
        return k, k_tensor

    # 根据use_parallel选择串行或并行处理
    if use_parallel:
        print(f"使用并行计算，n_jobs={n_jobs}")
        # 并行处理所有k点
        results = Parallel(n_jobs=n_jobs)(delayed(process_single_k)(k) for k in range(kvecs_length))
        # 将结果组装到Quantum_tensor中
        for k, k_tensor in results:
            Quantum_tensor[k] = k_tensor
    else:
        # 串行处理
        for k in range(kvecs_length):
            _, k_tensor = process_single_k(k)
            Quantum_tensor[k] = k_tensor
            
    skewness_tensor=Quantum_tensor-Quantum_tensor.transpose(0,2,1)

    # calculate delta energy
    delta_E = evalue.reshape(kvecs_length,basissize,1) - evalue.reshape(kvecs_length,1,basissize)

    # calculate the Fermi-Dirac distribution
    fermi=DF_dis(evalue,Ef,gamma)
    delta_fermi = fermi.reshape(kvecs_length,basissize,1) - fermi.reshape(kvecs_length,1,basissize)

    delt_func = zero_delta(omega, delta_E, gamma)

    term1=skewness_tensor[:,basissize//2:basissize,:basissize//2]*\
        delta_fermi[:,basissize//2:basissize,:basissize//2]\
            *delt_func[:,basissize//2:basissize,:basissize//2]

    term2=skewness_tensor[:,:basissize//2,basissize//2:basissize]*\
        delta_fermi[:,:basissize//2,basissize//2:basissize]*\
            delt_func[:,:basissize//2,basissize//2:basissize]
    term=term1+term2

    mesh = (Sint)*term.sum(axis=(1,2))*(energy_scale)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(kvecs[:, 0], kvecs[:, 1], c=mesh, cmap='viridis', 
                        s=50, alpha=0.8, edgecolors='none')
    plt.colorbar(scatter, label='shift current')
    plt.xlabel('kx')
    plt.ylabel('ky')
    plt.title(f'omega = {omega:.3f}')
    plt.tight_layout()

    # 创建保存图像的目录
    save_dir = f"image/lambda_{lambdaa:.3f}_gamma_{gamma:.3f}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存图像
    plt.savefig(f'{save_dir}/kvecs_mesh_scatter_yxx_omega_{omega:.3f}.png', dpi=300)
    plt.close()

    result = (Sint)*np.sum(term)*(energy_scale)

    return result


def get_shift_function(func_name: str):
    """
    根据输入的函数名称字符串返回对应的函数对象
    
    参数:
        func_name: 函数名称字符串，如'shiftxxx', 'shiftxyy', 'shiftyyy', 'shiftyxx'等
    
    返回:
        对应的函数对象
    """
    function_dict = {
        'shiftxxx': shiftxxx,
        'shiftxyy': shiftxyy,
        'shiftyyy': shiftyyy,
        'shiftyxx': shiftyxx
    }
    
    if func_name in function_dict:
        return function_dict[func_name]
    else:
        raise ValueError(f"未找到名为{func_name}的函数，可用的函数有: {', '.join(function_dict.keys())}")


