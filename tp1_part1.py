import numpy as np
import matplotlib.pyplot as plt

# les arrays sont batis avec les dimensions suivantes :
# pluie, arroseur, watson, holmes
# et chaque dimension : faux, vrai

prob_pluie = np.array([0.8, 0.2]).reshape(2, 1, 1, 1)
print("Pr(Pluie)={}\n".format(np.squeeze(prob_pluie)))

prob_arroseur = np.array([0.9, 0.1]).reshape(1, 2, 1, 1)
print("Pr(Arroseur)={}\n".format(np.squeeze(prob_arroseur)))

watson = np.array([[0.8, 0.2], [0, 1]]).reshape(2, 1, 2, 1)
print("Pr(Watson | Pluie)={}\n".format(np.squeeze(watson)))

holmes = np.array([[1, 0], [0.1, 0.9], [0, 1], [0, 1]]).reshape(2, 2, 1, 2)
print("Pr(Holmes | Pluie, Arroseur)={}\n".format(np.squeeze(holmes)))

# prob gazon watson mouille.
a = (watson * prob_pluie)[:,:,1,:].sum(0).squeeze()
print("Pr(Watson=1)={}\n".format(a))

# prob gazon watson mouille si gazon holmes mouille.
b = ((watson * holmes * prob_pluie * prob_arroseur)[:,:,1,1].sum((0, 1)).squeeze()) \
    /((holmes * prob_pluie * prob_arroseur)[:,:,:,1].sum((0, 1)).squeeze())
print("Pr(Watson=1 | Holmes=1)={}\n".format(b))

# prob gazon watson mouille si gazon holmes mouille - arroseur.
c = ((watson * holmes * prob_pluie)[:,0,1,1].sum(0).squeeze())/((holmes * prob_pluie)[:,0,:,1].sum(0).squeeze())
print("Pr(Watson=1 | Holmes=1, Arroseur=0)={}\n".format(c))

# prob gazon watson mouille si - arroseur.
d = (watson * prob_pluie)[:,0,1,:].sum(0).squeeze()
print("Pr(Watson=1 | Arroseur=0)={}\n".format(d))

# prob gazon watson mouille si pluie.
e = watson[1,:,1,:].squeeze()
print("Pr(Watson=1 | Pluie=1)={}\n".format(e))
