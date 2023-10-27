import pandas as pd
import numpy as np
import math


def TD_CC_TSB(A, n, alpha, delta):
    # A is the data matrix, n=1 did nothing
    # alpha is in the paper, delta is just a small number.
    n_bicluster = []
    # the for loop need to be removed, since it did nothing. But it will the return data structure, which will impact
    # the function residualScoreCol_add.
    for i in range (0, n):
        num_features = A.shape[0]
        num_time_points = A.shape[1]

        # I and J are the index holder of the data matrix.
        I = list(np.arange(0,num_features))
        J = list(np.arange(0,num_time_points))
        H = residualScore(A, I, J)
        #intial temp_H and H
        temp_H = 0
        H = 10000
        #print "deletion process"
        while abs(H - temp_H) > delta:
            temp_H = H
            I = delelteRow(A, I, J, H, alpha)
            H = residualScore(A, I, J)
            #print "Hrow", H
            J = delelteCol(A, I, J, H, alpha)
            H = residualScore(A, I, J)

            #print "Hcol", H
        #print "Insertion process"
        insert = True
        while insert == True:
            I_before = len(I)
            J_before = len(J)
            #print "rows before start insertion", I
            #print "columns before start insertion", J
            J = insertCol(A, I, J, H,alpha)
            H = residualScore(A, I, J)
            I = insertRow(A, I, J, H,alpha)
            H = residualScore(A, I, J)
            I_after = len(I)
            J_after = len(J)
            #print "the length of rows after insertion", I_after
            #print "the length of columns after insertion", J_after
            if (I_before == I_after) and (J_before == J_after):
                insert = False

        # store the bicluster index
        n_bicluster.append([i, I, J])

    return n_bicluster


def delelteRow(A, I, J, H, alpha):
    for i in I:
        H_row = residualScoreRow_del(A, I, J, i)
        #print i, "row score", H_row
        ratio = 1./H*H_row
        #print "H score is", H
        #print "Ri ratio is", ratio
        if ratio > alpha:
            I.remove(i)
    #print "currently rows after deleting rows", I
    return I


def delelteCol(A, I, J, H, alpha):
    J_prime = [min(J), max(J)]
    for j in J_prime:
        H_col = residualScoreCol_del(A, I, J, j)
        ratio = 1./H*H_col
        #print j, "column socre", H_col
        #print "H score is", H
        #print "Rj ratio is", ratio
        if ratio > alpha:
            J.remove(j)
    #print "currently columns after deleting column",J
    return J

def insertCol(A, I, J, H,alpha):
    J_prime = [min(J)-1, max(J)+1]
    for j in J_prime:

        if (j >= A.shape[1]) or (j < 0):
            return J
        else:
            H_col = residualScoreCol_add(A, I, J, j)
            #print "column for insertion", j
            ratio = 1./H*H_col
            #print "column socore", H_col
            #print "Rj ration", ratio
            if ratio < alpha:
                J.append(j)
            #print "currently columns after inserting column",J
            return J


def insertRow(A, I, J, H,alpha):
    num_features = A.shape[0]
    full_I = list(np.arange(0,num_features))
    for i in full_I:
        if i not in I:
            H_row = residualScoreRow_add(A, I, J, i)
            ratio = 1./H*H_row
            #print i, "row score", H_row
            #print "Ri ratio", ratio
            if ratio < alpha:
                I.append(i)
    #print "currently rows after inserting rows",I
    return I



def residualScore(A, I, J):
    I.sort()
    J.sort()
    # get sub matrix based on index holder I(row) and J(column)
    sub_A = A[np.ix_(I,J)]
    # sub matrix mean
    a_I_J = sub_A.mean()
    temp = 0
    # loop the row of submatrix
    for i_sub in range (0, sub_A.shape[0]):
        a_i_J = sub_A.mean(1)[i_sub,0]
        # loop the column of the submatrix
        for j_sub in range(0, sub_A.shape[1]):
            a_i_j = sub_A[i_sub,j_sub]
            a_I_j = sub_A.mean(0)[0,j_sub]
            temp += (a_i_j - a_i_J - a_I_j + a_I_J)**2
    I_num = len(I)
    J_num = len(J)
    H = 1./(I_num*J_num)*temp
    return H


def residualScoreRow_del(A, I, J, i):
    I.sort()
    J.sort()
    # get row index of submatrix, which is the row i in full matrix
    i_sub = I.index(i)
    sub_A = A[np.ix_(I,J)]
    a_I_J = sub_A.mean()
    a_i_J = sub_A.mean(1)[i_sub,0]
    temp = 0
    for j_sub in range(0, len(J)):
        a_i_j = sub_A[i_sub,j_sub]
        a_I_j = sub_A.mean(0)[0,j_sub]
        temp += (a_i_j - a_i_J - a_I_j + a_I_J)**2
    J_num = len(J)
    H_row = 1./J_num*temp
    return H_row


def residualScoreCol_del(A, I, J, j):
    I.sort()
    J.sort()
    # get column index of submatrix, which is the column j in full matrix
    j_sub = J.index(j)
    sub_A = A[np.ix_(I,J)]
    a_I_J = sub_A.mean()
    a_I_j = sub_A.mean(0)[0,j_sub]
    temp = 0
    for i_sub in range (0, len(I)):
        a_i_J = sub_A.mean(1)[i_sub,0]
        a_i_j = sub_A[i_sub,j_sub]
        temp += (a_i_j - a_i_J - a_I_j + a_I_J)**2
    I_num = len(I)
    H_column = 1./(I_num)*temp

    return H_column

def residualScoreRow_add(A, I, J, i):
    I.sort()
    J.sort()
    i_sub = len(I)
    sub_A = A[np.ix_(I,J)]
    a_I_J = sub_A.mean()
    I_add = list(I)
    I_add.append(i)
    sub_A_add = A[np.ix_(I_add,J)]

    a_i_J = sub_A_add.mean(1)[i_sub,0]
    temp = 0
    for j_sub in range(0, len(J)):
        a_i_j = sub_A_add[i_sub,j_sub]
        a_I_j = sub_A.mean(0)[0,j_sub]
        temp += (a_i_j - a_i_J - a_I_j + a_I_J)**2
    J_num = len(J)
    H_row = 1./J_num*temp
    return H_row


def residualScoreCol_add(A, I, J, j):
    I.sort()
    J.sort()
    sub_A = A[np.ix_(I,J)]
    a_I_J = sub_A.mean()
    j_sub = len(J)
    J_add = list(J)
    J_add.append(j)
    sub_A_add = A[np.ix_(I,J_add)]
    a_I_j = sub_A_add.mean(0)[0,j_sub]
    temp = 0
    for i_sub in range (0, len(I)):
        a_i_J = sub_A.mean(1)[i_sub,0]
        a_i_j = sub_A_add[i_sub,j_sub]
        temp += (a_i_j - a_i_J - a_I_j + a_I_J)**2
    I_num = len(I)
    H_column = 1./(I_num)*temp

    return H_column

def bicluster_out(fileindate):

    # filein is the file path and name.
    # fileindate is the date read from stockdatelist. In this setup, it is easier to loop over all file with date as variable.
    filein = 'stockreturn1min'+str(fileindate)+'.csv'
    f = pd.read_csv (filein)
    ret = f.drop(['date','time'], axis=1)
    ret = ret.set_index('KEY1')
    ret_tran = ret.transpose()
    A = ret_tran.as_matrix()
    a = np.asmatrix(A)
    a1 = a.copy()
    num_stock_explained = 0
    # number of bicluster
    i = 0
    temp = 100
    # here is 80 is the beta in our paper
    while (num_stock_explained < 80):
        result = TD_CC_TSB(a1, 1, alpha= 1.2, delta= 1e-12)
        I = result[0][1]
        temp = len(I)
        num_stock_explained += temp
        J = result[0][2]

        # if there is new biclucter identified.
        if len(I) > 1:
            ff =ret_tran.iloc[I,J]
            ee = ret_tran.iloc[I,:]
            i += 1
            # output the identified new bicluster
            outfile = str(fileindate)+'_'+'bicluster'+str(i)+'.csv'
            ff.to_csv(outfile)

            # Eliminate all the identified biclusters from the orignal data matrix
            ret_tran = ret_tran[~ret_tran.isin(ee).all(1)]
            A = ret_tran.as_matrix()
            a = np.asmatrix(A)
            a1 = a.copy()
        else:
            break
    return i, num_stock_explained

# read the date list
tradingdayList_full = pd.read_csv ('stockdatelist.csv')
tradingdayList_full = list(tradingdayList_full.date.unique())
date_bicluster = []
# loop over the all the date
for e in tradingdayList_full:
    n, num = bicluster_out(e)
    # record the date (e), the number of identifed biclusters (n), and the number of the stock explained (num)
    date_bicluster.append([e, n, num])
bic = pd.DataFrame(date_bicluster)
# output the file recording the the date (e), the number of identifed biclusters (n), and the number of the stock explained (num)
bic.to_csv('date_bicluster.csv')
