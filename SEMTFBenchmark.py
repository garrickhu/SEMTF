# -*- coding: utf-8 -*-
"""
A python code to compare computation speed between different SEMTF models
Wang1977 : Optical resolution through a turbulent medium with adaptive phase compensations
Charnotskii1993 : Anisoplanatic short-exposure imaging in turbulence
Tofsted2011 : Reanalysis of turbulence effects on short-exposure passive imaging
Dai2007: Zernike annular polynomials and atmospheric turbulence
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma,j1,jv,hyp2f1
from scipy.integrate import quad, dblquad, nquad
import time
import math
import matplotlib



#--------gloabal arguments -------------
A = 2*(gamma(6/5)*24/5)**(5/6)
B=(gamma(11/6))**2/2/(np.pi)**(11/3)




def expztilthyp(r,phi,rho,D,r0,epsi):
    if rho > 0:
        pplus = np.sqrt(r**2+0.25*rho**2+r*rho*np.cos(phi))
        gmaplus = np.arctan2(r*np.sin(phi),r*np.cos(phi)+0.5*rho)
        pminus = np.sqrt(r**2+0.25*rho**2-r*rho*np.cos(phi))
        gmaminus = np.arctan2(r*np.sin(phi),r*np.cos(phi)-0.5*rho)  
        
        coeff = -1*32*rho*A*B*np.pi**(5/3)*2**(-17/3)*D**(2/3)/r0**(5/3)/(1-epsi**4)
        
        if pminus == 0 and pplus == 0:
            Iminus = 0
            Iplus = 0
            
        if pminus == 0 and pplus !=0 :
            Iminus = 0
            term2 = 0
            term4 = 0
            if D/2/pplus <=1:
                term2 = -1*(D/2/pplus)**(-2/3)*gamma(1/6)*gamma(-5/6)*hyp2f1(1/6,-5/6,3,(D/2/pplus)**2)/gamma(3)
            else:
                term2 = (2*pplus/D)*gamma(1/6)*gamma(-11/6)*hyp2f1(1/6,-11/6,2,(2*pplus/D)**2)
                

            if 0< epsi <= 1:
                if D*epsi/2/pplus <=1:
                    term4 = -1*(D*epsi/2/pplus)**(-2/3)*gamma(1/6)*gamma(-5/6)*hyp2f1(1/6,-5/6,3,(D*epsi/2/pplus)**2)/gamma(3)
                else:
                    term4 = (2*pplus/D/epsi)*gamma(1/6)*gamma(-11/6)*hyp2f1(1/6,-11/6,2,(2*pplus/D/epsi)**2)


            term2 = term2 * np.cos(gmaplus)
            term4 = term4 * np.cos(gmaplus)
            Iplus = term2 - epsi**(14/3)*term4
            
        if pminus != 0 and pplus == 0 :
            Iplus = 0 
            term1 = 0 
            term3 = 0 
            if D/2/pminus <= 1:
                term1 = -1*(D/2/pminus)**(-2/3)*gamma(1/6)*gamma(-5/6)*hyp2f1(1/6,-5/6,3,(D/2/pminus)**2)/gamma(3)
            else:
                term1 = (2*pminus/D)*gamma(1/6)*gamma(-11/6)*hyp2f1(1/6,-11/6,2,(2*pminus/D)**2)
            if 0< epsi <= 1:
                if D*epsi/2/pminus <=1:
                    term3 = -1*(D*epsi/2/pminus)**(-2/3)*gamma(1/6)*gamma(-5/6)*hyp2f1(1/6,-5/6,3,(D*epsi/2/pminus)**2)/gamma(3)
                else:
                    term3 = (2*pminus/D/epsi)*gamma(1/6)*gamma(-11/6)*hyp2f1(1/6,-11/6,2,(2*pminus/D/epsi)**2)
            term1 = term1 * np.cos(gmaminus)
            term3 = term3 * np.cos(gmaminus)
            Iminus = term1 - epsi**(14/3)*term3
            
        if pminus != 0 and pplus != 0:
            term1 = 0
            term2 = 0
            term3 = 0
            term4 = 0
            if D/2/pminus <= 1:
                term1 = -1*(D/2/pminus)**(-2/3)*gamma(1/6)*gamma(-5/6)*hyp2f1(1/6,-5/6,3,(D/2/pminus)**2)/gamma(3)
            else:
                term1 = (2*pminus/D)*gamma(1/6)*gamma(-11/6)*hyp2f1(1/6,-11/6,2,(2*pminus/D)**2)
                
            
            if D/2/pplus <=1:
                term2 = -1*(D/2/pplus)**(-2/3)*gamma(1/6)*gamma(-5/6)*hyp2f1(1/6,-5/6,3,(D/2/pplus)**2)/gamma(3)  
            else:
                term2 = (2*pplus/D)*gamma(1/6)*gamma(-11/6)*hyp2f1(1/6,-11/6,2,(2*pplus/D)**2)
                
            
            if 0< epsi <= 1:
                
                if D*epsi/2/pminus <=1:
                    term3 = -1*(D*epsi/2/pminus)**(-2/3)*gamma(1/6)*gamma(-5/6)*hyp2f1(1/6,-5/6,3,(D*epsi/2/pminus)**2)/gamma(3)
                else:
                    term3 = (2*pminus/D/epsi)*gamma(1/6)*gamma(-11/6)*hyp2f1(1/6,-11/6,2,(2*pminus/D/epsi)**2)
                       
                if D*epsi/2/pplus <=1:
                    term4 = -1*(D*epsi/2/pplus)**(-2/3)*gamma(1/6)*gamma(-5/6)*hyp2f1(1/6,-5/6,3,(D*epsi/2/pplus)**2)/gamma(3)
                else:
                    term4 = (2*pplus/D/epsi)*gamma(1/6)*gamma(-11/6)*hyp2f1(1/6,-11/6,2,(2*pplus/D/epsi)**2)
            term1 = term1 * np.cos(gmaminus)
            term2 = term2 * np.cos(gmaplus)
            term3 = term3 * np.cos(gmaminus)
            term4 = term4 * np.cos(gmaplus)
            Iminus = term1 - epsi**(14/3)*term3
            Iplus = term2 - epsi**(14/3)*term4
        
        summ =  Iminus - Iplus
        res = summ*coeff
        #print(res)
    else:
        res = 0
    return(np.exp(res)*r)


def doubleinteztilt(rho,D,r0,epsi):
    #double integration calculation with two intersected circle, Ztilt
    #print('rho',rho)
    #print('D',D)
    #print('r0',r0)
    
    def lthetaq0(theta):
        return(0.5*(np.sqrt(D**2-(rho*np.sin(theta))**2) - rho*np.cos(theta))) #0<= rho <=D, big circle intersection
    def lthetaq0_1(theta):
        return(0.5*(np.sqrt(D**2-(rho*np.sin(theta))**2) + rho*np.cos(theta)))
    
    def lthetaq1(theta):
        return(0.5*(np.sqrt((D*epsi)**2-(rho*np.sin(theta))**2) - rho*np.cos(theta))) #0<= rho <=D*epsi, small circle intersection
    
    def lthetaq2(theta):
        return(0.5*(np.sqrt((D*epsi)**2-(rho*np.sin(theta))**2) + rho*np.cos(theta))) #small circle inside big circle,0<= rho <=D*(1-epsi)/2
    
    def lthetaq3(theta):
        return(0.5*(-1*np.sqrt((D*epsi)**2-(rho*np.sin(theta))**2) + rho*np.cos(theta)))
    
    
    
    if 0<= rho <D:
        res0 = 4*dblquad(expztilthyp,0,np.pi/2,0,lthetaq0,args=(rho,D,r0,epsi))[0] #big circle
    elif rho == D:
        res0 = 0
    
    if 0<= rho <=D*epsi:
        res1 = 4*dblquad(expztilthyp,0,np.pi/2,0,lthetaq1,args=(rho,D,r0,epsi))[0] #small circle
    else:
        res1 = 0 
    
    if 0<= rho <=D*(1-epsi)/2:
        if rho <= D*epsi:
            res2 = dblquad(expztilthyp,0,np.pi,0,lthetaq2,args=(rho,D,r0,epsi))[0] #small circle completely inside big circle
            res2 = 4*res2
        else:
            thetamax = np.arcsin(D*epsi/rho)
            res2 = dblquad(expztilthyp,0,thetamax,lthetaq3,lthetaq2,args=(rho,D,r0,epsi))[0]
            res2 = 4*res2
    else:
        res2 = 0
        
    if D*(1-epsi)/2< rho <=D*(1+epsi)/2:
        if rho < D*epsi:
            x0 = D**2*(1-epsi**2)/8/rho
            y0 = np.sqrt(D**2/4 - (D**2*(1-epsi**2)/8/rho + 0.5*rho)**2)
            th0 = np.arctan2(y0,x0)
        
            res3_1 = dblquad(expztilthyp,0,th0,0,lthetaq0,args=(rho,D,r0,epsi))[0]
            res3_2 = dblquad(expztilthyp,th0,np.pi,0,lthetaq2,args=(rho,D,r0,epsi))[0]
            res3 = 4*(res3_1+res3_2) #计算发现res3=res4，所以不需要计算res4了

            res5=0
            res6=0
            
        elif D*epsi<= rho <=D*(1+epsi)/2:
            res3=0
            x0 = D**2*(1-epsi**2)/8/rho
            y0 = np.sqrt(D**2/4 - (D**2*(1-epsi**2)/8/rho + 0.5*rho)**2)
            th0 = np.arctan2(y0,x0) #交叉点角度
            
            th2 = np.arcsin(D*epsi/rho) #切线角度
            x2 = np.cos(th2)*np.sqrt(rho**2/4-D**2*epsi**2/4)
            
            if x0 > x2:
                res5_1 = dblquad(expztilthyp, 0, th0, lthetaq3 , lthetaq0, args=(rho,D,r0,epsi))[0]
                res5_2 = dblquad(expztilthyp, th0, th2, lthetaq3, lthetaq2,args=(rho,D,r0,epsi))[0]
                res5 = 4*(res5_1+res5_2)
                res6 = 0
            else:
                res5=0
                res6 = 4*dblquad(expztilthyp, 0, th0, lthetaq3, lthetaq0, args=(rho,D,r0,epsi))[0]
            
    else:
        res3 = 0
        res5 = 0
        res6 = 0
        
    res = res0 + res1 -res2 - res3 - res5 - res6
    #print(res0)
    return(res,res0,res1,res2,res3,res5,res6)


def frontz(rho,D,r0,epsi,NN=3):
    #front part ztilt
    temp1 = gamma(1/6)*gamma(7/3)/gamma(29/6)/gamma(17/6)
    temp2 = temp1*np.pi**(19/6)/2
    C=A*B/2
    temp3 = C*256/np.pi/2
    
    #-------------Gauss Hypergeometric Function---------
    coe1 = temp3/D**(1/3)/(1-epsi**4)**2/r0**(5/3)
    temp4 = (np.pi/2)**(11/3)*gamma(1/6)*gamma(-11/6)/2/2/np.pi
    frtz3 = np.exp(-(A/2)*(rho/r0)**(5/3) - coe1*rho**2*(temp2*(1+epsi**(23/3)) - temp4*epsi**4*hyp2f1(1/6,-11/6,3,epsi**2)))
    #----------------------
    
    return(frtz3)


def mtfannular(rho,D,epsi):
    #clear annular aperture mtf
    
    if 0<= rho <= D:
        term1 = 2*(np.arccos(rho/D) - rho*np.sqrt(1-(rho/D)**2)/D)/np.pi
    
    term2 = 0
    term3 = 0
    
    if epsi > 0:
        if 0<= rho <=D*epsi:
            term3 = 2*(np.arccos(rho/D/epsi) - rho*np.sqrt(1-(rho/D/epsi)**2)/D/epsi)/np.pi
        else:
            term3 = 0 
        
        if 0<= rho <= D*(1-epsi)/2:
            term2 = 2*epsi**2
        elif D*(1-epsi)/2 < rho <= D*(1+epsi)/2:
            theta1 = np.arccos((4*rho**2 + D**2 - D**2*epsi**2)/4/rho/D)
            theta2 = np.arccos((4*rho**2 - D**2 + D**2*epsi**2)/4/rho/D/epsi)
            
            term2 = 2*(theta1 + epsi**2*theta2 - 2*np.sin(theta1)*rho/D)/np.pi
        else:
            term2 = 0 
        
    res = (term1 - term2 + epsi**2*term3)/(1-epsi**2)
    return(res)



def simplesemtfz(rho,D,r0,epsi,NN=3):
    #Fried-type SEMTF Ztilt
    telescopemtf = mtfannular(rho, D, epsi)

    
    temp1 = gamma(1/6)*gamma(7/3)/gamma(29/6)/gamma(17/6)
    temp2 = temp1*np.pi**(19/6)/2
    C=A*B/2
    temp3 = C*256/np.pi/2

    
    #-------------Gauss Hypergeometric Function---------
    coe1 = temp3/D**(1/3)/(1-epsi**4)**2/r0**(5/3)
    temp4 = (np.pi/2)**(11/3)*gamma(1/6)*gamma(-11/6)/2/2/np.pi
    avse3 = np.exp(-(A/2)*(rho/r0)**(5/3) + coe1*rho**2*(temp2*(1+epsi**(23/3)) - temp4*epsi**4*hyp2f1(1/6,-11/6,3,epsi**2)))
    #----------------------
    
    ssemtfz3 = telescopemtf*avse3
    
    return(ssemtfz3)
    


def expgtilthyp(r,phi,rho,D,r0,epsi):
    if rho > 0:
        pplus = np.sqrt(r**2+0.25*rho**2+r*rho*np.cos(phi))
        gmaplus = np.arctan2(r*np.sin(phi),r*np.cos(phi)+0.5*rho)
        pminus = np.sqrt(r**2+0.25*rho**2-r*rho*np.cos(phi))
        gmaminus = np.arctan2(r*np.sin(phi),r*np.cos(phi)-0.5*rho)  
        
        coeff = -1*rho*A*B*np.pi**(5/3)*2**(-5/3)*D**(2/3)/r0**(5/3)/(1-epsi**2)
        
        if pminus == 0 and pplus == 0:
            Iminus = 0
            Iplus = 0
        if pminus == 0 and pplus !=0 :
            Iminus = 0
            term2 = 0
            term4 = 0
            if D/2/pplus <=1:
                term2 = -1*(D/2/pplus)**(-2/3)*gamma(1/6)*gamma(-5/6)*hyp2f1(1/6,-5/6,2,(D/2/pplus)**2)
            else:
                term2 = -1*(2*pplus/D)*gamma(1/6)*gamma(-5/6)*hyp2f1(1/6,-5/6,2,(2*pplus/D)**2)
            if 0< epsi <= 1:
                if D*epsi/2/pplus <=1:
                    term4 = -1*(D*epsi/2/pplus)**(-2/3)*gamma(1/6)*gamma(-5/6)*hyp2f1(1/6,-5/6,2,(D*epsi/2/pplus)**2)
                else:
                    term4 = -1*(2*pplus/D/epsi)*gamma(1/6)*gamma(-5/6)*hyp2f1(1/6,-5/6,2,(2*pplus/D/epsi)**2)
            term2 = term2 * np.cos(gmaplus)
            term4 = term4 * np.cos(gmaplus)
            Iplus = term2 - epsi**(8/3)*term4
        
        if pminus != 0 and pplus == 0 :
            Iplus = 0 
            term1 = 0 
            term3 = 0 
            if D/2/pminus <= 1:
                term1 = -1*(D/2/pminus)**(-2/3)*gamma(1/6)*gamma(-5/6)*hyp2f1(1/6,-5/6,2,(D/2/pminus)**2)
            else:
                term1 = -1*(2*pminus/D)*gamma(1/6)*gamma(-5/6)*hyp2f1(1/6,-5/6,2,(2*pminus/D)**2)
            if 0< epsi <= 1:
                if D*epsi/2/pminus <=1:
                    term3 = -1*(D*epsi/2/pminus)**(-2/3)*gamma(1/6)*gamma(-5/6)*hyp2f1(1/6,-5/6,2,(D*epsi/2/pminus)**2)
                else:
                    term3 = -1*(2*pminus/D/epsi)*gamma(1/6)*gamma(-5/6)*hyp2f1(1/6,-5/6,2,(2*pminus/D/epsi)**2)
            
            term1 = term1 * np.cos(gmaminus)
            term3 = term3 * np.cos(gmaminus)
            Iminus = term1 - epsi**(8/3)*term3
        
        if pminus != 0 and pplus != 0:
            term1 = 0
            term2 = 0
            term3 = 0
            term4 = 0
            if D/2/pminus <= 1:
                term1 = -1*(D/2/pminus)**(-2/3) * gamma(1/6)*gamma(-5/6)*hyp2f1(1/6,-5/6,2,(D/2/pminus)**2)
            else:
                term1 = -1*(2*pminus/D)*gamma(1/6)*gamma(-5/6)*hyp2f1(1/6,-5/6,2,(2*pminus/D)**2)
            
            if D/2/pplus <=1:
                term2 = -1*(D/2/pplus)**(-2/3)*gamma(1/6)*gamma(-5/6)*hyp2f1(1/6,-5/6,2,(D/2/pplus)**2)
            else:
                term2 = -1*(2*pplus/D)*gamma(1/6)*gamma(-5/6)*hyp2f1(1/6,-5/6,2,(2*pplus/D)**2)
            
            if 0< epsi <= 1:
                
                if D*epsi/2/pminus <=1:
                    term3 = -1*(D*epsi/2/pminus)**(-2/3)*gamma(1/6)*gamma(-5/6)*hyp2f1(1/6,-5/6,2,(D*epsi/2/pminus)**2)
                else:
                    term3 = -1*(2*pminus/D/epsi)*gamma(1/6)*gamma(-5/6)*hyp2f1(1/6,-5/6,2,(2*pminus/D/epsi)**2)
                    
                if D*epsi/2/pplus <=1:
                    term4 = -1*(D*epsi/2/pplus)**(-2/3)*gamma(1/6)*gamma(-5/6)*hyp2f1(1/6,-5/6,2,(D*epsi/2/pplus)**2)
                else:
                    term4 = -1*(2*pplus/D/epsi)*gamma(1/6)*gamma(-5/6)*hyp2f1(1/6,-5/6,2,(2*pplus/D/epsi)**2)
    
            
            term1 = term1 * np.cos(gmaminus)
            term2 = term2 * np.cos(gmaplus)
            term3 = term3 * np.cos(gmaminus)
            term4 = term4 * np.cos(gmaplus)
            Iminus = term1 - epsi**(8/3)*term3
            Iplus = term2 - epsi**(8/3)*term4
        
        summ =  Iminus - Iplus
        res = summ*coeff
        #print(res)
    else:
        res = 0
    return(np.exp(res)*r)



def doubleintegtilt(rho,D,r0,epsi):
    #Double integration G-tilt
    #print('rho',rho)
    #print('D',D)
    #print('r0',r0)
    
    def lthetaq0(theta):
        return(0.5*(np.sqrt(D**2-(rho*np.sin(theta))**2) - rho*np.cos(theta))) #0<= rho <=D, big circle intersection
    def lthetaq0_1(theta):
        return(0.5*(np.sqrt(D**2-(rho*np.sin(theta))**2) + rho*np.cos(theta)))
    
    def lthetaq1(theta):
        return(0.5*(np.sqrt((D*epsi)**2-(rho*np.sin(theta))**2) - rho*np.cos(theta))) #0<= rho <=D*epsi, small circle intersection
    
    def lthetaq2(theta):
        return(0.5*(np.sqrt((D*epsi)**2-(rho*np.sin(theta))**2) + rho*np.cos(theta))) #small circle inside big circle,0<= rho <=D*(1-epsi)/2
    
    def lthetaq3(theta):
        return(0.5*(-1*np.sqrt((D*epsi)**2-(rho*np.sin(theta))**2) + rho*np.cos(theta)))
    
    
    
    if 0<= rho <D:
        res0 = 4*dblquad(expgtilthyp,0,np.pi/2,0,lthetaq0,args=(rho,D,r0,epsi))[0] #big circle
    elif rho == D:
        res0 = 0
    
    if 0<= rho <=D*epsi:
        res1 = 4*dblquad(expgtilthyp,0,np.pi/2,0,lthetaq1,args=(rho,D,r0,epsi))[0] #small circle
    else:
        res1 = 0 
    
    if 0<= rho <=D*(1-epsi)/2:
        if rho <= D*epsi:
            res2 = dblquad(expgtilthyp,0,np.pi,0,lthetaq2,args=(rho,D,r0,epsi))[0] #small circle completely inside big circle
            res2 = 4*res2
        else:
            thetamax = np.arcsin(D*epsi/rho)
            res2 = dblquad(expgtilthyp,0,thetamax,lthetaq3,lthetaq2,args=(rho,D,r0,epsi))[0]
            res2 = 4*res2
    else:
        res2 = 0
        
    if D*(1-epsi)/2< rho <=D*(1+epsi)/2:
        if rho < D*epsi:
            x0 = D**2*(1-epsi**2)/8/rho
            y0 = np.sqrt(D**2/4 - (D**2*(1-epsi**2)/8/rho + 0.5*rho)**2)
            th0 = np.arctan2(y0,x0)
        
            res3_1 = dblquad(expgtilthyp,0,th0,0,lthetaq0,args=(rho,D,r0,epsi))[0]
            res3_2 = dblquad(expgtilthyp,th0,np.pi,0,lthetaq2,args=(rho,D,r0,epsi))[0]
            res3 = 4*(res3_1+res3_2) #计算发现res3=res4，所以不需要计算res4了
            
            res5=0
            res6=0
            
        elif D*epsi<= rho <=D*(1+epsi)/2:
            res3=0
            x0 = D**2*(1-epsi**2)/8/rho
            y0 = np.sqrt(D**2/4 - (D**2*(1-epsi**2)/8/rho + 0.5*rho)**2)
            th0 = np.arctan2(y0,x0)
            
            th2 = np.arcsin(D*epsi/rho)
            x2 = np.cos(th2)*np.sqrt(rho**2/4-D**2*epsi**2/4)
            
            if x0 > x2:
                res5_1 = dblquad(expgtilthyp, 0, th0, lthetaq3 , lthetaq0, args=(rho,D,r0,epsi))[0]
                res5_2 = dblquad(expgtilthyp, th0, th2, lthetaq3, lthetaq2,args=(rho,D,r0,epsi))[0]
                res5 = 4*(res5_1+res5_2)
                res6 = 0
            else:
                res5=0
                res6 = 4*dblquad(expgtilthyp, 0, th0, lthetaq3, lthetaq0, args=(rho,D,r0,epsi))[0]
            
    else:
        res3 = 0
        res5 = 0
        res6 = 0
        
    res = res0 + res1 -res2 - res3 - res5 - res6
    #print(res0)
    return(res,res0,res1,res2,res3,res5,res6)


def frontg(rho,D,r0,epsi,NN=3):
    temp1 = gamma(1/6)*gamma(4/3)/gamma(17/6)/gamma(11/6)
    temp2 = temp1*np.pi**(7/6)/2
    C=A*B/2
    temp3 = C*16*np.pi
    
    coe1 = (temp3/2)/D**(1/3)/(1-epsi**2)**2/r0**(5/3)
    temp4 = (np.pi/2)**(5/3)*gamma(1/6)*gamma(-5/6)/2/np.pi
    frtg3 = np.exp(-(A/2)*(rho/r0)**(5/3) - coe1*rho**2*(temp2*(1+epsi**(11/3)) + temp4*epsi**2*hyp2f1(1/6,-5/6,2,epsi**2)))
    
    return(frtg3)



def simplesemtfg(rho,D,r0,epsi,NN=3):
    telescopemtf = mtfannular(rho, D, epsi) #清澈环形光瞳mtf
    
    temp1 = gamma(1/6)*gamma(4/3)/gamma(17/6)/gamma(11/6)
    temp2 = temp1*np.pi**(7/6)/2
    C=A*B/2
    temp3 = C*16*np.pi
    
    coe1 = (temp3/2)/D**(1/3)/(1-epsi**2)**2/r0**(5/3)
    temp4 = (np.pi/2)**(5/3)*gamma(1/6)*gamma(-5/6)/2/np.pi
    avse3 = np.exp(-(A/2)*(rho/r0)**(5/3) + coe1*rho**2*(temp2*(1+epsi**(11/3)) + temp4*epsi**2*hyp2f1(1/6,-5/6,2,epsi**2)))
    ssemtfg3 = telescopemtf*avse3
    
    return(ssemtfg3)
    


def VQX(Q,D,r0):
    #Tofsted2011 V(Q,D/r0) argument
    q = np.log2(4)
    qa = 1.35*(q+1.5)
    qb = 1.45*(q-0.15)
    def bigsigma(q):
        return((np.exp(q)-1)/(np.exp(q)+1))
    A = 0.84+0.116*bigsigma(qa)
    B = 0.805+0.265*bigsigma(qb)
    X=D/r0
    x = np.log10(X)
    return(A+B*np.exp(-(x+1)**3/3.5)/10)




def guangmingdai(q,D,r0,epsi):
    #q从0到1
    if epsi == 0:
        telescopetf = 2*(np.arccos(q)-q*np.sqrt(1-q**2))/np.pi #circular
        avsedai = telescopetf*np.exp(-3.44195*(q*D/r0)**(5/3)*(1-0.9503*q**(1/3) - 0.5585*q**(4/3) + 0.5585*q**(7/3)))
    else:
        telescopetf = mtfannular(q, D, epsi) #annular
        def a2epsisquare(D,r0,epsi):
            temp1 = 0.023*gamma(1/6)*np.pi**(8/3)/2**(2/3)/(1+epsi**2)/(1-epsi**2)**2/gamma(17/6)
            temp2 = (1+epsi**(23/3))*gamma(14/3)/gamma(17/6)/gamma(29/6) - epsi**4*hyp2f1(1/6,-11/6,3,epsi**2)
            return(temp1*temp2*(D/r0)**(5/3))
        a22 = a2epsisquare(D, r0, epsi)
        
        if epsi == 0.25:
            a2a8 = -0.01474*(D/r0)**(5/3)
        elif epsi == 0.5:
            a2a8 = -0.01392*(D/r0)**(5/3)
        elif epsi == 0.75:
            a2a8 = -0.00885*(D/r0)**(5/3)
        
        D3epsi = 6.8839*(q*D/r0)**(5/3) - 16*q**2*a22/(1+epsi**2) - 32*np.sqrt(2)*q**2*(6*(1+epsi**2)*q**2 - 6*(1+epsi**2)*q +1 +epsi**2-np.sqrt(2)*epsi**4)*a2a8/(1-epsi**2)/np.sqrt(1+6*epsi**2+10*epsi**4+6*epsi**6+epsi**8)
        
        avsedai = telescopetf*np.exp(-0.5*D3epsi)
    return(avsedai)


#-------------------wang 1977------------------------
def delta0plus(r,theta,rho):
    y=np.arctan2(r*np.sin(theta),r*np.cos(theta)+0.5*rho)
    return(y)

def Delplus(r,theta,rho):
    y=(r**2+0.25*rho**2+r*rho*np.cos(theta))**(1/2)
    return(y)

def itaplus(delta,r,theta,rho):
    y=(0.25-(Delplus(r,theta,rho))**2*(np.sin(delta-delta0plus(r,theta,rho)))**2)**(1/2)-Delplus(r,theta,rho)*np.cos(delta-delta0plus(r,theta,rho))
    return(y)

def P2plusinte(delta,r,theta,rho,D):
    inte = 3*(itaplus(delta,r,theta,rho))**(14/3)*np.cos(delta)/14+(r/D)*3*(itaplus(delta,r,theta,rho))**(11/3)*np.cos(theta)/11+0.5*(rho/D)*(3/11)*(itaplus(delta,r,theta,rho))**(11/3)
    return(inte)

def P2plus(r,theta,rho,D,r0):
    res = quad(P2plusinte,0,2*np.pi,args=(r,theta,rho,D),limit=2000)
    #print(res,res[0]*32*6.88*(D/r0)**(5/3)*rho/D/np.pi)
    return(res)


def delta0minus(r,theta,rho):
    
    #print(r*np.cos(theta)-0.5*rho,r*np.sin(theta))
    y=np.arctan2(r*np.sin(theta),r*np.cos(theta)-0.5*rho) 
    return(y)

def Delminus(r,theta,rho):
    y=(r**2+0.25*rho**2-r*rho*np.cos(theta))**(1/2)
    return(y)

def itaminus(delta,r,theta,rho):
    y=(0.25-(Delminus(r,theta,rho))**2*(np.sin(delta-delta0minus(r,theta,rho)))**2)**(1/2)-Delminus(r,theta,rho)*np.cos(delta-delta0minus(r,theta,rho))
    return(y)

def P2minusinte(delta,r,theta,rho,D):
    inte = (3/14)*(itaminus(delta,r,theta,rho))**(14/3)*np.cos(delta) + (r/D)*3*(itaminus(delta,r,theta,rho))**(11/3)*np.cos(theta)/11 - 0.5*(rho/D)*(3/11)*(itaminus(delta,r,theta,rho))**(11/3)
    return(inte)

def P2minus(r,theta,rho,D,r0):
    res = quad(P2minusinte,0,2*np.pi,args=(r,theta,rho,D),limit=2000)
    #print(res,res[0]*32*6.88*(D/r0)**(5/3)*rho/D/np.pi)
    return (res)

def innerinte(r,theta,rho,D,r0):
    p2jia = P2plus(r,theta,rho,D,r0)
    p2jian = P2minus(r,theta,rho,D,r0)
    coeff = 32*A*(D/r0)**(5/3)*rho/D/np.pi
    inner = np.exp((-p2jia[0]+p2jian[0])*coeff)*r
    return(inner)

def finalinte(rho,D,r0):
    #print('rho',rho)
    def lthetaq(theta):
        return(0.5*(np.sqrt(D**2-(rho*np.sin(theta))**2)-rho*np.cos(theta)))
    
    res = dblquad(innerinte, 0, np.pi/2, 0, lthetaq, args=(rho,D,r0))
    
    res0 = 4*res[0] #扩展到四个象限
    #print('result:',res0)
    return(res0)

def Wangstructuralfunction(r,theta,rho,D,r0):
    #Wang的结构函数
    p2jia = P2plus(r,theta,rho,D,r0)
    p2jian = P2minus(r,theta,rho,D,r0)
    coeff = 32*6.88*(D/r0)**(5/3)*rho/D/np.pi
    firstterm = A*(rho/r0)**(5/3)
    secondterm = 7.182*rho**2/D**(1/3)/r0**(5/3)
    return(firstterm + secondterm - 2*coeff*(p2jian[0] - p2jia[0] ))

#-------------------------------------------------------------------------------------------




#---------Tofsted2014 SEMTF---------------------------------------------------
#Extended high-angular-frequency analysis of turbulence effects on short-exposure imaging
def PP(n, z):
    if n == 0:
        return 1
    elif n == 1:
        return z
    
    dp = [0] * (n + 1)
    dp[0] = 1  # PP(0, z) = 1
    dp[1] = z  # PP(1, z) = z
    
    for i in range(2, n + 1):
        dp[i] = (2 * (i - 1) + 1) * z * dp[i - 1] - (i - 1) * dp[i - 2]
        dp[i] /= i
    
    return dp[n]


def LL(n,w):
    return(np.sqrt(2*n+1)*PP(n,2*w-1))

def sigma(x):
    return((np.exp(x)-1)/(np.exp(x)+1))

def sigmaS(x,A,B,C,D):
    res = A+B*sigma(C*(x-D))
    return(res)

def VM(Q):
    return(sigmaS(np.log2(Q),0.221,0.07,1.539,0.15))

def sigmaP(x,A,B1,B2,C1,C2,D):
    if x<= D:
        res = A + B1*sigma(C1*(x-D))
    else:
        res = A + B2*sigma(C2*(x-D))
    return(res)

def V0(Q):
    return(sigmaP(np.log2(Q),0.879,0.373,0.085,0.373,1.57,-0.984))

def xw(q):
    return(sigmaS(q,-0.004,0.091,1.75,-0.05))

def wp(x):
    return(sigmaP(x,0.351,-0.133,-0.318,3.451,1.407,1.274))

def wpQX(Q,X):
    return(wp(np.log10(X) - xw(np.log2(Q))))

def V1x(x):
    return(sigmaP(x,0.122,0.044,0.082,4.296,2.323,1.179))

def AR(q):
    return(sigmaS(q,1.066,0.352,1.596,0.15))

def xP(q):
    return(sigmaS(q,-0.005,0.089,1.55,-0.15))

def VP(Q,X):
    return(AR(np.log2(Q))*V1x(np.log10(X) - xP(np.log2(Q))))

def CC(m,up):
    if m == 0:
        am = 20.2124
        bm = 19.6324
        cm = 0.0031
        dm = 0.5751
        em = 1.0686
    elif m == 2:
        am = 2.1118
        bm = 2.4928
        cm = 0.0042
        dm = -0.3515
        em = 0.9808
    elif m == 4:
        am = -0.3413
        bm = -0.3355
        cm = 0.0051
        dm = 0.0051
        em = 0.7996
    elif m == 6:
        am = -0.3529
        bm = -0.3769
        cm = 0.0
        dm = 0.0131
        em = 0.9041
    elif m == 8:
        am = -0.3696
        bm = -0.3982
        cm = 0.0
        dm = 0.0153
        em = 0.9504
    elif m == 10:
        am = 0.01009
        bm = 0.0
        cm = 0.0
        dm = -0.00452
        em = 0.0
    elif m == 12:
        am = 0.0055
        bm = 0.0
        cm = 0.0
        dm = -0.00257
        em = 0.0
    cm = am*(up-em) + bm*np.sqrt((up-em)**2+cm) + dm
    return(cm)

def DD(m,up):
    if m == 0:
        am = 6.4372
        bm = 5.8919
        cm = 0.0005
        dm = 0.5788
        em = 1.0341
    elif m == 2:
        am = -0.30273
        bm = 0.0
        cm = 0.0
        dm = -0.00057
        em = 0.0
    elif m == 4:
        am = 0.03335
        bm = -0.0
        cm = 0.0
        dm = -0.00413
        em = 0.0
    elif m == 6:
        am = -0.01271
        bm = -0.0
        cm = 0.0
        dm = 0.00485
        em = 0.0
    elif m == 8:
        am = 0.00425
        bm = 0.0
        cm = 0.0
        dm = -0.00154
        em = 0.0
    elif m == 10:
        am = -0.00452
        bm = 0.0
        cm = 0.0
        dm = 0.00186
        em = 0.0
    elif m == 12:
        am = 0.00055
        bm = 0.0
        cm = 0.0
        dm = -0.00019
        em = 0.0
    dm = am*(up-em) + bm*np.sqrt((up-em)**2+cm) + dm
    return(dm)


def Vbar(Q,X,w):
    wp = wpQX(Q,X)
    up = VP(Q,X)/VM(Q)
    if w <= wp:
        summ = 0
        for n in [0,2,4,6,8,10,12]:
            summ = summ + CC(n,up) * LL(n,0.5*w/wp)
    elif w> wp:
        summ = 0
        for n in [0,2,4,6,8,10,12]:
            summ = summ + DD(n,up) * LL(n,1-0.5*(1-w)/(1-wp))
    return(summ)

def Vdelta(Q,X,w):
    return(Vbar(Q,X,w)*VM(Q))

def VQXw(Q,X,w):
    return(V0(Q) + Vdelta(Q,X,w))


#------------------------------------------------------------------------------------




#--------Charnotskii1993-------eq(41) change 8 to4----------------------------
def Dcharnotskii(x,D,r0):
    if x<=1:
        res = (0.751*0.273*6.88/0.73)*(D/r0)**(5/3)*(3.5505*x**(5/3) - 8*0.931945*x**2*hyp2f1(1/6,-5/6,2,x**2) +4*x**2*gamma(1/6)*gamma(4/3)/gamma(17/6)/gamma(11/6)/2/np.sqrt(np.pi))
    else:
        res = (0.751*0.273*6.88/0.73)*(D/r0)**(5/3)*(3.5505*x**(5/3) - 8*0.931945*x**(5/3)*hyp2f1(1/6,-5/6,2,x**(-2)) +4*x**2*gamma(1/6)*gamma(4/3)/gamma(17/6)/gamma(11/6)/2/np.sqrt(np.pi))
    return(res)


#------------------------------------------------------------------------








if __name__=='__main__':
    
    
    
    D=1 #Diameter
    r0=0.1 #seeing
    X=D/r0 # Tofsted X argument
    
    nop = 96 #number of frequency point 
    epsi=0.0 #obstruction ratio
    qq = np.linspace(0,D,nop) #rho from 0 to D
    q = np.linspace(0,1,nop) #Normalized frequency from 0 to 1
    
    
    start = time.perf_counter()
    latterpartz = []
    latterpartg = []
    wang = []
    telescopetf = []
    for i in qq:
        latterpartz.append(doubleinteztilt(i, D, r0, epsi)[0])
        latterpartg.append(doubleintegtilt(i, D, r0, epsi)[0])
        telescopetf.append(mtfannular(i, D, epsi)) #
    
    
    frontpartz = frontz(qq, D, r0, epsi) 
    frontpartg = frontg(qq, D, r0, epsi)
    semtfzt = (4/np.pi)/(D**2*(1-epsi**2))*frontpartz*latterpartz #ztilt semtf
    semtfgt = (4/np.pi)/(D**2*(1-epsi**2))*frontpartg*latterpartg #gtilt semtf
    
    time1 = time.perf_counter()
    print('Our Numerical Z-tilt + G-tilt SEMTF together: ', time1-start)
    
    wangrho = np.linspace(0,0.995,nop)
    for j in wangrho: ##when j == 1 there will be error, when nop is too big there will be error too.
        wang.append(finalinte(j, D, r0))
    wangfrontpartz = frontz(wangrho, D, r0, epsi)
    wangmtf = (4/np.pi/D**2)*wangfrontpartz*wang
    time2 = time.perf_counter()
    print('Wang SEMTF: ', time2-time1)
    
    
    
    #------------Tofsted2011------------------------
    Q=16
    vv = VQX(Q,D,r0) 
    semtftofsted = telescopetf * np.exp(-3.44*(D/r0)**(5/3) * (q**(5/3) - vv*q**2)) 
    #-------------------------------------------------
    time3 = time.perf_counter()
    print('Tofsted SEMTF: ',time3-time2)
    
    #-------------Guangming Dai------------------

    avsedai = []
    for jj in q:
        avsedai.append(guangmingdai(jj, D, r0, epsi))
    
    time4 = time.perf_counter()
    print('Dai SEMTF: ', time4-time3)
    #--------------------------------------------   

    
    #------Charnotskii---------------------------------------------
    
    Charnotskii = []
    for kk in q:
        Charnotskii.append(np.exp(-0.5*Dcharnotskii(jj, D, r0)))
    Charnotskii = telescopetf * np.array(Charnotskii)
    time5 = time.perf_counter()
    print('Charnotskii SEMTF: ', time5-time4)
    #-------------------------------------------------------------
    
    
    

    
    fig,ax = plt.subplots()
    ax.plot(qq/D,semtfzt,linestyle=(5, (5, 5)),label='Numerical Z-tilt')   
    ax.plot(qq/D,semtfgt,linestyle=(0, (3, 1, 1, 1, 1, 1)),label='Numerical G-tilt')
    ax.plot(q,avsedai,linestyle = (0, (5, 1)), label='Dai')
    ax.plot(q,semtftofsted,linestyle=(0, (3, 5, 1, 5)),label='Tofsted')
    ax.plot(qq,wangmtf,'x',markersize=3,label='wang')
    ax.plot(q,Charnotskii, label='Charnotskii')
    ax.legend()
    ax.set_title(r'$r_0 = $ '+ str(round(r0*100))+' cm, '+r'ϵ = '+str(round(epsi,2)))
    ax.set_yscale('log')
    ax.set_ylim(5e-6,1.1)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('SEMTF')
    plt.show()
    
        
    
    end=time.perf_counter()
    print('Total time:', end-start)
    
    
    
    
    




