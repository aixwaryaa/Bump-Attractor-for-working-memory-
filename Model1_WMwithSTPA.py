
import math 
from math import floor, exp, sqrt, pi
import cmath 
import numpy 
import matplotlib.pyplot as plt
from numpy import e, cos, zeros, arange, roll, where, random, ones, mean, reshape, dot, array, flipud, pi, exp, dot, angle, degrees, shape, linspace
import scipy
from scipy import special
import numpy as np 
import time
import scipy.signal
from numpy.fft import rfft, irfft
import os
import sys
import socket
import multiprocessing as mp

from scipy.interpolate import interp1d

#prefs.codegen.target = 'cython'

start_time = time.time()
par = int()


###############
############### Accesory functions
###############


def decode_rE(rE, N=512):
    #Population vector for a given rE
    # return ( angle in radians, absolut angle in radians, abs angle in degrees )
    N=len(rE)
    angles= np.arange(0,2*pi,2*np.pi/N) 
    R = np.sum(np.dot(np.reshape(rE, (1,N)),np.exp(1j*angles)))/np.sum(rE) ## population vector 
    angle_decoded = np.degrees(np.angle(R))
    strength_code = np.abs(R)
    if angle_decoded<0:
        angle_decoded = 360+angle_decoded 
    
    return angle_decoded, strength_code


def model_I0E_constant(value, N=512): 
    y=[value for x in range(N)] 
    return np.reshape(np.array(y), (N,1)) 



###############
############### MODEL STPA function
###############


def model_STPA(totalTime, targ_onset1, targ_onset2, presentation_period, delay1, delay2,iti, angle_pos=120,angle_pos2=random.random()*360,    
    tauE=60, tauI=10, tauf=7000, taud=80, I0E=0.6, I0I=0.4, U=0.4, Gad=0.001, gadapt=0.5, tauad=1500,
    GEE=0.016, GEI=0.015, GIE=0.012 , GII=0.007, sigE=0.06, sigI=0.04,
    kappa_E=100, kappa_I=1.5, k_inhib=0.07, kappa_stim=20,
    N=512, save_RE=True):
    ##

    # Task timings converted to simulation steps
    st_sim =time.time()
    dt=2;
    nsteps=int(floor(totalTime/dt)); 
    origin = np.radians(angle_pos) 
    origin2 = np.radians(angle_pos2) 
    targ_offset1 = targ_onset1 + presentation_period; 
    targon1 = floor(targ_onset1/dt);
    targoff1 = floor(targ_offset1/dt); 
    resp_onset = targ_offset1 + delay1 ; 
    resp_offset = resp_onset + presentation_period;
    respon = floor (resp_onset/dt);
    respoff = floor( resp_offset/dt);
    targ_onset2 = resp_offset + iti
    targ_offset2 = targ_onset2 + presentation_period;
    targon2 = floor(targ_onset2/dt)
    targoff2 = floor(targ_offset2/dt)
    ######

    ###### Definition of the network connectivity WE and WI
    v_E=np.zeros((N)); 
    v_I=np.zeros((N));
    WE=np.zeros((N,N));
    WI=np.zeros((N,N));
    theta =np.arange(0,2*pi,2*pi/N) 

    for i in range(0, N):
        v_E_new=[e**(kappa_E*np.cos(theta[f]))/(2*pi*scipy.special.i0(kappa_E)) for f in range(0, len(theta))]  # use a translationally invariant von Mises function to define the ring connectivity
        v_I_new=[e**(kappa_I*np.cos(theta[f]))/(2*pi*scipy.special.i0(kappa_I)) + k_inhib for f in range(0, len(theta))] 
        #    
        vE_NEW=np.roll(v_E_new,i) 
        vI_NEW=np.roll(v_I_new,i) 
        # 
        WE[:,i]=vE_NEW 
        WI[:,i]=vI_NEW
        #   
    # try to visualize the connectivity that we are using with the matplotlib command imshow


    ###### Definition of the current corresponding to stimulus 1: target
    target=np.zeros((N)) 
    for i in range(0, N):
        target[i]=e**(kappa_stim*cos(theta[i] - origin ))  / (2*pi*scipy.special.i0(kappa_stim))   
    #
    noise_stim = np.random.normal(0, 0.01, N) 
    target = target+ noise_stim
    target=reshape(target, (N,1)) 

    ###### Definition of the current corresponding to stimulus 2: target2
    target2=np.zeros((N)) 
    for i in range(0, N):
        target2[i]=(e**(kappa_stim*cos(theta[i] - origin2 ))  / (2*pi*scipy.special.i0(kappa_stim)))
    #
    noise_stim = np.random.normal(0, 0.01, N) 
    target2 = target2+ noise_stim 
    target2=reshape(target2, (N,1)) 

    #
    ###### Initialize all the variables for the simulation
    mf=1
    rE=np.zeros((N,1));
    rI=np.zeros((N,1)); 
    s1 = np.zeros((N,1));
    u = np.ones((N,1))*U
    x = np.ones((N,1))
    a = np.ones((N,1))*0.002 #a = np.zeros((N,1))
    RE=np.zeros((N,nsteps));
    RI=np.zeros((N,nsteps));
    SE=np.zeros((N,nsteps));
    AD=np.zeros((N,nsteps));
    p_u=np.ones((N,nsteps));
    p_x=np.ones((N,nsteps));
    
    ###
    ###
    ### input-output function converting currents into firing rates, to be used during the simulation
    fnc = lambda x : x*x*(x>0)*(x<1) + array([cmath.sqrt(4*x[i]-3) for i in range(0, len(x))]).real * (x>=1)
    xx=np.linspace(-5,10,100)
    yy=fnc(xx)
    func = interp1d(xx,yy, fill_value='extrapolate') # convert function into interpolated lookup table to make simulations much faster


    ### MAIN SIMULATION LOOP 
    for i in range(0, nsteps):
        # independent noise to each neuron at each time step
        noiseE = sigE*random.randn(N,1);  
        noiseI = sigI*random.randn(N,1);
       
        # calculate the current inputs to each neuron based on the firing rates and the connectivity
        IE= GEE*dot(WE, (rE*u*x)) - GIE*dot(WI,rI) - Gad * a + I0E; 
        II= GEI*dot(WE,rE) +  (I0I-GII*mean(rI))*ones((N,1)); 
        
        ## consider additional inputs based on the task period: stimuli, response
        ## presentation stim 1
        if i>targon1 and i<targoff1:
            IE=IE+target;
            II=II+target;
        ## response 
        if i>respon and i<respoff: 
            IE = IE - 5 # hyperpolarize all neurons to stop the bumps
            II = II - 5 
         ## presentation stim 2
        if i>targon2 and i<targoff2:
            IE =IE + target2 ;
            II=II+ target2 ;

        
        # Euler method applied to the differential rate equations, for each population
        rE = rE + (func(IE) - rE + noiseE)*dt/tauE;
        rI = rI + (func(II) - rI + noiseI)*dt/tauI;

        # Euler method applied to the differential equations of synaptic plasticity: paper mongillo, barak, tsodyks. Science 2008
        u = u + ((U - u) / tauf + U*(1-u)*rE/1000)*dt;
        x = x + ((1 - x)/taud - u*x*rE/1000)*dt;

        # Euler method applied to the differential equation of the adaptation current a
        a = a + (-a/tauad + gadapt * rE/1000) * dt

        ur=np.reshape(u, N)
        xr=np.reshape(x, N)

        #append results to arrays 
        RE[:,i] = np.reshape(rE, N);
        RI[:,i] = np.reshape(rI, N);
        p_u[:,i] = ur;
        p_x[:,i] = xr;
        SE[:,i]= ur*xr;
        AD[:,i] = np.reshape(a,N)
   



    

    
    #
    #### Decode positions
    final_position_bump, final_strength = decode_rE(RE[:,-5], N)
    previous_position_bump, previous_strength =decode_rE(RE[:,respon-1], N) 
    del1s_position_bump, del1s_strength  = decode_rE(RE[:,int(targoff2+np.floor(1000/dt))], N)
    del0s_position_bump, del0s_strength  = decode_rE(RE[:,int(targoff2+np.floor(100/dt))], N)
    del2s_position_bump, del2s_strength  = decode_rE(RE[:,int(targoff2+np.floor(2000/dt))], N)





# Creating the time axis (assuming 'i' represents time steps)
    time_steps = range(rate_151.shape[0])  # Assuming 'i' corresponds to time steps





    p_targ = int((N * np.degrees(origin))/360) 
    p_targ2 = int((N * np.degrees(origin2))/360) 
     
    ### Output
    ###return bias_target, bias_dist, number_of_bumps, angle_separation, RE #rE[p_targ][0], I0E
    if save_RE==True:
        return previous_position_bump, del0s_position_bump,del1s_position_bump,del2s_position_bump,final_position_bump,previous_strength,del0s_strength, del1s_strength,del2s_strength,final_strength, RE, WE , SE, AD
    else:
        return previous_position_bump, del0s_position_bump, del1s_position_bump, del2s_position_bump, final_position_bump,previous_strength,del0s_strength, del1s_strength,del2s_strength,final_strength


def run_simulation(i):

    np.random.seed(os.getpid()+i)
    global par #rep
    log_file    = "simulation_%i_%f_%s_%i" %(os.getpid(), time.time(), socket.gethostname(), i) 
    decoded_posi3 = "output_beh%d.txt" %(par)

    print (log_file)

    #####
    ##### Codes to run simulations
    #####
    Angle_pres1 = 120 #degrees 
    Angle_pres2 = random.random()*360 #degres
    target_onset1 = 1500 #ms
    presentation_period = 250 #ms
    target_offset1 = target_onset1 + presentation_period 
    delay1= np.random.choice([1,2])*1000  #1500
    resp_onset = target_offset1 + delay1 
    resp_offset = resp_onset + presentation_period 
    iti = np.random.choice([1,2])*1000 #ms
    target_onset2 = resp_offset + iti
    target_offset2 = target_onset2 + presentation_period
    delay2= 3000 #ms
    time_simulation=target_offset2 + delay2

    saveRE = True
    
    results= model_STPA(totalTime=time_simulation, presentation_period=presentation_period,   
            targ_onset1= target_onset1, targ_onset2=target_onset2, angle_pos=Angle_pres1 ,angle_pos2=Angle_pres2, delay1=delay1, delay2=delay2, iti=iti ,
            tauE=60, tauI=10, tauf=2000, taud=80, I0E=0.6,
            I0I=0.4, U=0.9,  Gad=1, gadapt =0.01, tauad=1500,
            GEE=0.0235, GEI=0.019, GIE=0.01, GII=0.1,
            sigE=4, sigI=2.2,
            kappa_E=12, kappa_I=1.5, k_inhib=0.07, kappa_stim=12,
            N=512,save_RE=saveRE)
    
    with open (decoded_posi3, 'a') as myfile :
            myfile.write(log_file +"; " +str(np.round(Angle_pres1, 3))+"; "+
            str(np.round(Angle_pres2, 3))+"; " +
            str(delay1)+"; "+str(iti)+"; "+
            str(np.round(results[0], 3))+"; "+
            str(np.round(results[1], 3))+"; "+str(np.round(results[2], 3))+"; "+
            str(np.round(results[3], 3))+"; " +str(np.round(results[4], 3))+"; "+
            str(np.round(results[5], 3))+"; "+str(np.round(results[6], 3))+"; "+
            str(np.round(results[7], 3))+"; "+str(np.round(results[8], 3))+"; "+
            str(np.round(results[9], 3))+'\n')


    if saveRE==True:
        RE = results[10]
        plt.figure()
        plt.imshow(RE)

        plt.figure()
        
        plt.show()
       

    print ('saved')


#####################################################################################################
#                                    RUN SIMULATIONS                                                #
#####################################################################################################

## USE THIS TO RUN IN PARALLEL IN A MULTICORE MACHINE
# numcores=16
# pool = mp.Pool(processes=numcores, maxtasksperchild=1)   
# myseeds = range(0, numcores)
# args = [(sd) for sd in myseeds]
# pool.map(run_simulation, args, chunksize=1)

run_simulation(0)

#run_simulation(1)
print ('all sims finished')
print (time.time() - start_time )