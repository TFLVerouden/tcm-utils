import numpy as np

def coughmachine_dim(L_change=0):
    """
    returns the difference in length between the exiting point of the spraytec and the first transparent screen of the spraytec
    """
    optical_table =0.92 #m
    h_coughmachine = 0.154 #m, to the place where films are situated
    total_h = h_coughmachine+optical_table
    L= L_change + 0.25 #difference from this position to spraytec
    return total_h,L

def spraytec_dim(lift_height=0.346):
    """
    returns the difference in height between the film in the cough machine and the bottom of the transparent screen of the spraytec
    """
    table_height = 0.553 -0.346# height without lift on
    print(table_height)
    spraytec_ontable =0.472 # lower point, heigher point = 0.514
    height_spraytec_dif = 0.514 -0.472 #difference between top and bototm of plastic screen, this is not complete measurement volume
    total_height_spraytec = table_height +lift_height +spraytec_ontable
    return total_height_spraytec,height_spraytec_dif

h_coughmachine,L_coughmachine = coughmachine_dim()
h_spraytec, dh_spraytec = spraytec_dim()

heightdiff = h_coughmachine - h_spraytec
print(heightdiff)
print(L_coughmachine)
print(dh_spraytec)

