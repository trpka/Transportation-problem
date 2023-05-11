# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 19:08:45 2022

@author: Vladimir
"""

#%% Metod severozapadnog ugla sa iterativnim postupkom
import numpy as np
import math

#Balansiranje problema 
def balansiranje_problema(ponuda, potraznja, cene, penali = None):
    ukupna_ponuda = sum(ponuda)
    ukupna_potraznja = sum(potraznja)
    
    if ukupna_ponuda < ukupna_potraznja:
        if penali is None:
            raise Exception('Supply less than demand, penalties required')
        nova_ponuda = ponuda + [ukupna_potraznja - ukupna_ponuda]
        nove_cene = cene + [penali]
        return nova_ponuda, potraznja, nove_cene
    if ukupna_ponuda > ukupna_potraznja:
        nova_potraznja = potraznja + [ukupna_ponuda - ukupna_potraznja]
        nove_cene = cene + [[0 for _ in potraznja]]
        return ponuda, nova_potraznja, nove_cene
    return ponuda, potraznja, cene



def severozapadni_ugao(ponuda, potraznja):
    i = 0
    j = 0
    pocetnoResenje_SZ = []
    ponudaCopy = ponuda.copy()
    potraznjaCopy = potraznja.copy()
    while len(pocetnoResenje_SZ) < len(ponuda) + len(potraznja) - 1:
        pon = ponudaCopy[i]
        potr = potraznjaCopy[j]
        v = min(pon, potr)
        ponudaCopy[i] -= v
        potraznjaCopy[j] -= v
        pocetnoResenje_SZ.append(((i, j), v))
        if ponudaCopy[i] == 0 and i < len(ponuda) - 1:
            i += 1
        elif potraznjaCopy[j] == 0 and j < len(potraznja) - 1: 
            j += 1
    return pocetnoResenje_SZ


def potencijali(pocetnoResenje, cene):
    u = [None] * len(cene) 
    v = [None] * len(cene[0]) 
    u[0] = 0
    cena = 0
    pocetnoResenjeCopy = pocetnoResenje.copy()
    while len(pocetnoResenjeCopy) > 0:
        for indeksi, resenje in enumerate(pocetnoResenjeCopy):
            i, j = resenje[0]
            if u[i] is None and v[j] is None: continue
        
            cena = cene[i][j]
            if u[i] is None:
                u[i] = cena - v[j]
            else: 
                v[j] = cena - u[i]
            pocetnoResenjeCopy.pop(indeksi)
            break
    print(u)
    print(v)
    return u, v

#Ovde se racunau tezine za nebazne promenljive tj vrednosti za nebazne promenljive
def nove_cene_nebaznih_promenljivih(pocetnoResenje, cene, u, v):
    nove_cene = [] 
    for i, red in enumerate(cene): #iterira po redu
        for j, cena in enumerate(red): #iterira po elementu u redu
           #Ukoliko se u kvadraticu ne nalazi broj vratice True 
            nebazno = all([index[0] != i or index[1] != j for index,vrednost in pocetnoResenje])
            #print(nebazno)
            if nebazno:
                nove_cene.append(((i, j), cena - u[i] - v[j])) #dodeljivanje vrednosti tezinama
    print(nove_cene)           
    return nove_cene
    
#Ovde samo pita da li je potrebno dalje ici kroz algoritam, tj ukoliko je o = true, znaci da ima
#vrednosti koje su manje od 0, znaci ima negativnih i treba raditi optimizaciju 
def provera(nove_cene):
    for index, vrednost in nove_cene:
        if vrednost < 0: 
            return True
    return False
        

#Ovde nam trazi najnegativniji element za koji treba da radimo optimizaciju tj polje
 #za najnegativniji element
def najmanja_nebazna_cena(nove_cene):
    najmanje_vrednosti = []
    
    for index, vrednost in nove_cene:
        najmanje_vrednosti.append(vrednost)
    
    najmanja_vrednost = min(najmanje_vrednosti)
    
  
    
    for index, vrednost in nove_cene:
        if(vrednost == najmanja_vrednost):
            return index
        
#Ovde izgleda trazi gde treba provuci tj kakvu konturu je neophodno napraviti

#Ovde nam trebaju cvorovi kroz koje ce proci petlja
def potencijalniCvorovi(petlja, nijePosecen):
    poslednjiCvor = petlja[-1]
    cvoroviURedu = [n for n in nijePosecen if n[0] == poslednjiCvor[0]]
    cvoroviUKoloni = [n for n in nijePosecen if n[1] == poslednjiCvor[1]] 
    if len(petlja) < 2:
        return cvoroviURedu + cvoroviUKoloni
    else:
        prepre_cvor = petlja[-2] 
        potez_red = prepre_cvor[0] == poslednjiCvor[0]
        if potez_red: 
            return cvoroviUKoloni
        return cvoroviURedu
    
    
    
def pronadjiPetlju(pocetniIndexi, pocetnoPolje, petlja):
    if len(petlja) > 3:
        kraj = len(potencijalniCvorovi(petlja, [pocetnoPolje])) == 1 
        if kraj: 
            return petlja
        
    nisuPoseceni = list(set(pocetniIndexi) - set(petlja))
    moguciCvorovi = potencijalniCvorovi(petlja, nisuPoseceni)
    for sledeci_cvor in moguciCvorovi: 
        sledecaPetlja = pronadjiPetlju(pocetniIndexi, pocetnoPolje, petlja + [sledeci_cvor])
        if sledecaPetlja: 
            return sledecaPetlja


def novoResenje(pocetnoResenje, petlja):
    plus = petlja[0::2]
    minus = petlja[1::2]
    get_bv = lambda pos: next(v for p, v in pocetnoResenje if p == pos)
    minIndeks = sorted(minus, key=get_bv)[0]
    minVrednost = get_bv(minIndeks)
    
    novoResenje = []
    for p, v in [p for p in pocetnoResenje if p[0] != minIndeks] + [(petlja[0], 0)]:
        if p in plus:
            v += minVrednost
        elif p in minus:
            v -= minVrednost
        novoResenje.append((p, v))
        
    return novoResenje

def transportniProblem(ponuda, potraznja, cene, penali):
    balansiranaPonuda, balansiranaPotraznja, balansiraneCene = (balansiranje_problema(ponuda, potraznja, cene, penali))
    pocetnoResenje = severozapadni_ugao(balansiranaPonuda, balansiranaPotraznja)
    optimalnoResenje = optimalno(pocetnoResenje, balansiraneCene)

    matrica = np.zeros((len(balansiraneCene), len(balansiraneCene[0])))
    for (i, j), v in optimalnoResenje:
        matrica[i][j] = v
    trosakTransporta = odrediTrosak(balansiraneCene, matrica)
    return matrica, trosakTransporta

def optimalno(pocetnoResenje, cene):
    u,v = potencijali(pocetnoResenje, cene)
    cij = nove_cene_nebaznih_promenljivih(pocetnoResenje, cene, u, v)

    if(provera(cij)):
        najnegativnije = najmanja_nebazna_cena(cij)
        petlja = pronadjiPetlju([p for p,v in pocetnoResenje], najnegativnije, [najnegativnije])
        return optimalno(novoResenje(pocetnoResenje, petlja), cene)
    return pocetnoResenje

def odrediTrosak(cene, resenje):
    suma = 0
    for i, red in enumerate(cene):
        for j, cena in enumerate(red):
            suma += cena*resenje[i][j]
    return suma


def transportniProblem_najmanje_cene(ponuda, potraznja, cene, penali):
    cene = [[20,11,15,13],
            [17,14,12,13],
            [15,12,18,18]]
    balansiranaPonuda, balansiranaPotraznja, balansiraneCene = (balansiranje_problema(ponuda, potraznja, cene, penali))
    pocetnoResenje = najmanja_cena(balansiranaPonuda, balansiranaPotraznja, cene)
    optimalnoResenje = optimalno_najmanje_cene(pocetnoResenje, balansiraneCene)

    matrica = np.zeros((len(balansiraneCene), len(balansiraneCene[0])))
    for (i, j), v in optimalnoResenje:
        matrica[i][j] = v
    trosakTransporta = odrediTrosak_najmanje_cene(balansiraneCene, matrica)
    return matrica, trosakTransporta



def optimalno_najmanje_cene(pocetnoResenje, cene):
    cene = [[20,11,15,13],
            [17,14,12,13],
            [15,12,18,18]]
    u,v = potencijali(pocetnoResenje, cene)
    cij = nove_cene_nebaznih_promenljivih(pocetnoResenje, cene, u, v)

    if(provera(cij)):
        najnegativnije = najmanja_nebazna_cena(cij)
        petlja = pronadjiPetlju([p for p,v in pocetnoResenje], najnegativnije, [najnegativnije])
        return optimalno_najmanje_cene(novoResenje(pocetnoResenje, petlja), cene)
    return pocetnoResenje

def odrediTrosak_najmanje_cene(cene, resenje):
    suma = 0
    cene = [[20,11,15,13],
            [17,14,12,13],
            [15,12,18,18]]
    for i, red in enumerate(cene):
        for j, cena in enumerate(red):
            suma += cena*resenje[i][j]
    return suma

# Metod najmanjih cena 

import numpy as np
import math

def najmanja_cena(ponuda, potraznja, cene):
    i = 0
    j = 0
    ponudaCopy = ponuda.copy()
    potraznjaCopy = potraznja.copy()
    pocetno_res_nc=[]
    ceneCopy = cene.copy()
    
    while len(pocetno_res_nc) != len(ponuda) * len(potraznja):
        minimum = np.min(ceneCopy, axis=None)
        for ii, red in enumerate(ceneCopy):
            for jj, cena in enumerate(red):
                if cena == minimum:
                    i,j = ii, jj
                   
        
        najmanja_vrednost = min(ponudaCopy[i], potraznjaCopy[j])
        pocetno_res_nc.append(((i, j), najmanja_vrednost))
        ponudaCopy[i] -= najmanja_vrednost
        potraznjaCopy[j] -= najmanja_vrednost
        
        ceneCopy[i][j] = math.inf
    
    
    pocetnoResenjeA = []
    for p, v in enumerate(pocetno_res_nc):
        if(v[1] != 0):
            pocetnoResenjeA.append(v)
            
    return pocetnoResenjeA


# Poyivi funckija
cene0 = [[20,11,15,13],
        [17,14,12,13],
        [15,12,18,18]]

ponuda0 = [2,6,7]
potraznja0 = [3,3,4,5]
penali0=[]


ponuda, potraznja, cene = balansiranje_problema(ponuda0,potraznja0,cene0, penali0)
pocetnoResenje_SZ = severozapadni_ugao(ponuda, potraznja)

u,v = potencijali(pocetnoResenje_SZ, cene)
print(pocetnoResenje_SZ)
print(u,v)
print("\n")

tezinski_koeficijenti = nove_cene_nebaznih_promenljivih(pocetnoResenje_SZ, cene, u, v)
print(tezinski_koeficijenti)

nc_nebaznih = nove_cene_nebaznih_promenljivih(pocetnoResenje_SZ, cene, u, v)
end_of_algorihm = provera(nc_nebaznih)
print("Da li je potrebno nastaviti sa algoritmom: " + " " + str(end_of_algorihm))

najnegativnije = najmanja_nebazna_cena(nc_nebaznih)
print("Pozicija najnegativnijeg elementa"  + " " + str(najnegativnije))

petlja = pronadjiPetlju([p for p,v in pocetnoResenje_SZ], najnegativnije, [najnegativnije])
print("")
print("Polja kroz koje prolazi kontura: " + " " + str(petlja))

novo = novoResenje(pocetnoResenje_SZ, petlja)

penali = []
transport, cena = transportniProblem(ponuda, potraznja, cene, penali)
print('\n\nMatrica:\n',transport,'\n\nCena transporta:' ,cena)

print("############# Metod najmanjih cena ")

ponuda, potraznja, cene = balansiranje_problema(ponuda0,potraznja0,cene0, penali0)
pocetno_res_nc = najmanja_cena(ponuda, potraznja, cene)
print(pocetno_res_nc)

cene = [[20,11,15,13],
        [17,14,12,13],
        [15,12,18,18]]

u_nc,v_nc = potencijali(pocetno_res_nc, cene)
print("\n")

tezinski_koeficijenti_nc = nove_cene_nebaznih_promenljivih(pocetno_res_nc, cene, u_nc, v_nc)
print(tezinski_koeficijenti_nc)

nc_nebaznih_nc = nove_cene_nebaznih_promenljivih(pocetno_res_nc, cene, u_nc, v_nc)
end_of_algorihm_nc = provera(nc_nebaznih_nc)
print("Da li je potrebno nastaviti sa algoritmom: " + " " + str(end_of_algorihm_nc))

najnegativnije_nc = najmanja_nebazna_cena(nc_nebaznih_nc)
print("Pozicija najnegativnijeg elementa"  + " " + str(najnegativnije_nc))

petlja_nc = pronadjiPetlju([p for p,v in pocetno_res_nc], najnegativnije_nc, [najnegativnije_nc])
print("")
print("Polja kroz koje prolazi kontura: " + " " + str(petlja_nc))

novo_nc = novoResenje(pocetno_res_nc, petlja_nc)
print(novo_nc)

penali = []
transport, cena = transportniProblem_najmanje_cene(ponuda0, potraznja0, cene, penali)
print('\n\nMatrica:\n',transport,'\n\nCena transporta:' ,cena)

print("############# Vogelova metoda")

#Ova f-ja racuna razlike po vrstama
def razlika_po_redovima(cene, potraznja):
    cene1 = np.transpose(cene)  
    cene1 = [sorted(i) for i in cene1] 
   # print(cene)
    red_razlika = [] 
    for i in range(len(cene1)):
            if(potraznja[i] != 0): 
                red_razlika.append(cene1[i][1] - cene1[i][0])
            else:
                red_razlika.append(0)
    return red_razlika
  #  print(red_razlika)

#Ova f-ja racuna razlike po kolonama
def razlika_po_kolonama(cene, ponuda):

    cene = [sorted(i) for i in cene]
    razlika_kolona=[]
    for i in range(len(cene)):
            if(ponuda[i] != 0):
                razlika_kolona.append(cene[i][1]-cene[i][0])
            else:
                razlika_kolona.append(0)
    return razlika_kolona


#Ovde nam se trazi najveca razlika
def indeks_sa_najvecom_razlikom(razlika_kolona, red_razlika):
    max_red = max(red_razlika)
    #print("Najveci u redu" + " " + str(max_red))
    max_kolona = max(razlika_kolona)
    red = False
    
    if(max_red > max_kolona):
        index_max = red_razlika.index(max_red) 
        red=True
       
    else:
        index_max=razlika_kolona.index(max_kolona)
    #print("INdex max" + " " + str(index_max))        
    return index_max,red
    
#Ovde pronalazi najmanji element kroz koji ce pokusati da provuce najvecu cenu 
def Uzmi_minimum_u_matrici_red(index_max, ponuda, potraznja, red, cene):
    minimalni_index = 0
    
    if(red == True):
        l = []
        for ind1 in range(0, len(cene)):
           l.append(cene[ind1][index_max]) #uzima sve elemente iz kolone/reda
        lm = min(l)
        print("Lista l" + " " + str(l))  

        minimalni_index = l.index(min(l))
        return minimalni_index,index_max
    else:
        print(cene[index_max])
        lm=min(cene[index_max])
        l=cene[index_max]
        minimalni_index = l.index(lm)
        print("Lista l" + " " + str(l))  
       
        return index_max, minimalni_index
    
    


def nadjiPocetnoResenje(pocetnoResenje_VOGEL, cene, ponuda, potraznja, i, j, red):
   
    if(ponuda[i] > potraznja[j]):
        pocetnoResenje_VOGEL[i][j]=potraznja[j]
        ponuda[i]=ponuda[i]-potraznja[j]
        potraznja[j]=0
        for ind in range(0,len(ponuda)):
          cene[ind][j]=99999

    else:
        pocetnoResenje_VOGEL[i][j]=ponuda[i]
        potraznja[j]=potraznja[j]-ponuda[i]
        ponuda[i]=0
        for ind in range(0, len(ponuda)):
            cene[i][ind] = 99 
    return pocetnoResenje_VOGEL,cene

cene = [[20,11,15,13],
        [17,14,12,13],
        [15,12,18,18]]
ponuda = [2,6,7]
potraznja = [3,3,4,5]
pocetnoResenje_VOGEL=np.zeros((3,4))

#U ovom delu je ograniceno koliko metoda treba da se izvrsava, tj dokle god je potraznja
#razlicita od 0
while(max(potraznja)!=0):

    potraznjaA = [i for i, v in enumerate(potraznja) if v == 0]
    print(potraznjaA)
    if len(potraznjaA)==len(potraznja)-1:
        # Ovdje je dio kada ostane ne kraju jedna kolona
        potraznja0 = [i for i, element in enumerate(potraznja) if element != 0]

        red=True
        index_max=potraznja0[0]
        for k in range(len(ponuda)):
            pocetnoResenje_VOGEL[k][index_max]+=ponuda[k]
        potraznja[index_max]=0


    else:
        razlika_kolona = razlika_po_kolonama(cene,ponuda)
        red_razlika = razlika_po_redovima(cene,potraznja)
        index_max,red = indeks_sa_najvecom_razlikom(razlika_kolona,red_razlika)
        i,j=Uzmi_minimum_u_matrici_red(index_max,ponuda,potraznja,red, cene)

        pocetnoResenje_VOGEL,cene=nadjiPocetnoResenje(pocetnoResenje_VOGEL,cene,ponuda,potraznja,i,j,red)
  
    
print(pocetnoResenje_VOGEL)

p=[]
for i, red in enumerate(pocetnoResenje_VOGEL):
        for j, vrednost in enumerate(red):
            if(vrednost!=0):
                p.append(((i,j),vrednost))
                
#%% Vogelova metoda dictionari

from collections import defaultdict

""""costs  = {'W': {'A': 20, 'B': 11, 'C': 15, 'D': 13},
          'X': {'A': 17, 'B': 14, 'C': 12, 'D': 13},
          'Y': {'A': 15, 'B': 12, 'C': 18, 'D': 18}}
demand = {'A': 3, 'B': 3, 'C': 4, 'D': 5}
cols = sorted(demand.keys())
supply = {'W': 2, 'X': 6, 'Y': 7}"""


costs  = {'W': {'A': 10, 'B': 12, 'C': 0},
          'X': {'A': 8, 'B': 4, 'C': 3},
          'Y': {'A': 6, 'B': 9, 'C': 4},
          'Z': {'A': 7, 'B': 8, 'C': 5}}
demand = {'A': 10, 'B': 40, 'C': 30}
cols = sorted(demand.keys())
supply = {'W': 20, 'X': 30, 'Y': 20, 'Z': 10}
res = dict((k, defaultdict(int)) for k in costs)
g = {}
for x in supply:
    g[x] = sorted(costs[x].keys(), key=lambda g: costs[x][g])
for x in demand:
    g[x] = sorted(costs.keys(), key=lambda g: costs[g][x])

while g:
    d = {}
    for x in demand:
        d[x] = (costs[g[x][1]][x] - costs[g[x][0]][x]) if len(g[x]) > 1 else costs[g[x][0]][x]
    s = {}
    for x in supply:
        s[x] = (costs[x][g[x][1]] - costs[x][g[x][0]]) if len(g[x]) > 1 else costs[x][g[x][0]]
    f = max(d, key=lambda n: d[n])
    t = max(s, key=lambda n: s[n])
    t, f = (f, g[f][0]) if d[f] > s[t] else (g[t][0], t)
    v = min(supply[f], demand[t])
    res[f][t] += v
    demand[t] -= v
    if demand[t] == 0:
        for k, n in supply.items():
            if n != 0:
                g[k].remove(t)
        del g[t]
        del demand[t]
    supply[f] -= v
    if supply[f] == 0:
        for k, n in demand.items():
            if n != 0:
                g[k].remove(f)
        del g[f]
        del supply[f]

for n in cols:
    #print("\t" , n),
    print( n),
#print
cost = 0
for g in sorted(costs):
    print( g, "\t"),
    for n in cols:
        y = res[g][n]
        if y != 0:
            print( y),
        cost += y * costs[g][n]
        #print ("\t"),
    #print
print ("\nUkupan trosak = ", cost)

print(res)

#%%
from tkinter import *
import numpy as np
import math

#Balansiranje problema 
def balansiranje_problema(ponuda, potraznja, cene, penali = None):
    ukupna_ponuda = sum(ponuda)
    ukupna_potraznja = sum(potraznja)
    
    if ukupna_ponuda < ukupna_potraznja:
        if penali is None:
            raise Exception('Supply less than demand, penalties required')
        nova_ponuda = ponuda + [ukupna_potraznja - ukupna_ponuda]
        nove_cene = cene + [penali]
        return nova_ponuda, potraznja, nove_cene
    if ukupna_ponuda > ukupna_potraznja:
        nova_potraznja = potraznja + [ukupna_ponuda - ukupna_potraznja]
        nove_cene = cene + [[0 for _ in potraznja]]
        return ponuda, nova_potraznja, nove_cene
    return ponuda, potraznja, cene


def potencijali(pocetnoResenje, cene):
    u = [None] * len(cene) 
    v = [None] * len(cene[0]) 
    u[0] = 0
    cena = 0
    pocetnoResenjeCopy = pocetnoResenje.copy()
    while len(pocetnoResenjeCopy) > 0:
        for indeksi, resenje in enumerate(pocetnoResenjeCopy):
            i, j = resenje[0]
            if u[i] is None and v[j] is None: continue
        
            cena = cene[i][j]
            if u[i] is None:
                u[i] = cena - v[j]
            else: 
                v[j] = cena - u[i]
            pocetnoResenjeCopy.pop(indeksi)
            break
    print(u)
    print(v)
    return u, v

#Ovde se racunau tezine za nebazne promenljive tj vrednosti za nebazne promenljive
def nove_cene_nebaznih_promenljivih(pocetnoResenje, cene, u, v):
    nove_cene = [] 
    for i, red in enumerate(cene): #iterira po redu
        for j, cena in enumerate(red): #iterira po elementu u redu
           #Ukoliko se u kvadraticu ne nalazi broj vratice True 
            nebazno = all([index[0] != i or index[1] != j for index,vrednost in pocetnoResenje])
            print(nebazno)
            if nebazno:
                nove_cene.append(((i, j), cena - u[i] - v[j])) #dodeljivanje vrednosti tezinama
    print(nove_cene)           
    return nove_cene
    
#Ovde samo pita da li je potrebno dalje ici kroz algoritam, tj ukoliko je o = true, znaci da ima
#vrednosti koje su manje od 0, znaci ima negativnih i treba raditi optimizaciju 
def provera(nove_cene):
    for index, vrednost in nove_cene:
        if vrednost < 0: 
            return True
    return False
        

#Ovde nam trazi najnegativniji element za koji treba da radimo optimizaciju tj polje
 #za najnegativniji element
def najmanja_nebazna_cena(nove_cene):
    najmanje_vrednosti = []
    
    for index, vrednost in nove_cene:
        najmanje_vrednosti.append(vrednost)
    
    najmanja_vrednost = min(najmanje_vrednosti)
    
  
    
    for index, vrednost in nove_cene:
        if(vrednost == najmanja_vrednost):
            return index
        
#Ovde izgleda trazi gde treba provuci tj kakvu konturu je neophodno napraviti

#Ovde nam trebaju cvorovi kroz koje ce proci petlja
def potencijalniCvorovi(petlja, nijePosecen):
    poslednjiCvor = petlja[-1]
    cvoroviURedu = [n for n in nijePosecen if n[0] == poslednjiCvor[0]]
    cvoroviUKoloni = [n for n in nijePosecen if n[1] == poslednjiCvor[1]] 
    if len(petlja) < 2:
        return cvoroviURedu + cvoroviUKoloni
    else:
        prepre_cvor = petlja[-2] 
        potez_red = prepre_cvor[0] == poslednjiCvor[0]
        if potez_red: 
            return cvoroviUKoloni
        return cvoroviURedu
    
    
    
def pronadjiPetlju(pocetniIndexi, pocetnoPolje, petlja):
    if len(petlja) > 3:
        kraj = len(potencijalniCvorovi(petlja, [pocetnoPolje])) == 1 
        if kraj: 
            return petlja
        
    nisuPoseceni = list(set(pocetniIndexi) - set(petlja))
    moguciCvorovi = potencijalniCvorovi(petlja, nisuPoseceni)
    for sledeci_cvor in moguciCvorovi: 
        sledecaPetlja = pronadjiPetlju(pocetniIndexi, pocetnoPolje, petlja + [sledeci_cvor])
        if sledecaPetlja: 
            return sledecaPetlja


def novoResenje(pocetnoResenje, petlja):
    plus = petlja[0::2]
    minus = petlja[1::2]
    get_bv = lambda pos: next(v for p, v in pocetnoResenje if p == pos)
    minIndeks = sorted(minus, key=get_bv)[0]
    minVrednost = get_bv(minIndeks)
    
    novoResenje = []
    for p, v in [p for p in pocetnoResenje if p[0] != minIndeks] + [(petlja[0], 0)]:
        if p in plus:
            v += minVrednost
        elif p in minus:
            v -= minVrednost
        novoResenje.append((p, v))
        
    return novoResenje

def New_Window():
    ws = Toplevel()
    ws.title("Metod najmanjih cena")

    canvas =Canvas(ws, width = 500, height = 800,  relief = 'raised')
    canvas.pack()
    
    lab1 =Label(ws, text='Metod najmanjih cena')
    lab1.config(font=('helvetica', 14))
    canvas.create_window(250, 25, window=lab1)
    
    lab2 =Label(ws, text='Unesite potražnje:')
    lab2.config(font=('helvetica', 10))
    canvas.create_window(250, 100, window=lab2)
    
    entry1 =Entry (ws) 
    canvas.create_window(250, 140, window=entry1)
    
    potraznja0 = []
    def potraznja_f ():
        a1 = entry1.get()
        potraznja0.append(a1)
        for i in range(0, len(potraznja0)):
            potraznja0[i] = int(potraznja0[i])
        
          
    button = Button(ws, text='Dodaj potražnju', command=potraznja_f, bg='brown', fg='white', font=('helvetica', 9, 'bold'))
    canvas.create_window(250, 180, window=button)
    
    
    lab3 =Label(ws, text='Unesite ponude:')
    lab3.config(font=('helvetica', 10))
    canvas.create_window(250, 220, window=lab3)
    
    entry2 =Entry (ws) 
    canvas.create_window(250, 260, window=entry2)
    
    ponuda0 = []
    def ponuda_f ():
        a2 = entry2.get()
        ponuda0.append(a2)
        for i in range(0, len(ponuda0)):
            ponuda0[i] = int(ponuda0[i])
        
        
    
    button2 = Button(ws, text='Dodaj ponudu', command=ponuda_f, bg='brown', fg='white', font=('helvetica', 9, 'bold'))
    canvas.create_window(250, 300, window=button2)
    
    
    text_var = []
    entries = []
    matrix = []
    #potraznja = []
    potraznja_string = []
    
    
    # callback function to get your StringVars
    def get_mat(rows,cols):
        for i in range(rows):
            matrix.append([])
            for j in range(cols):
                matrix[i].append(text_var[i][j].get())
        
        
        print(matrix)
        #matrix = [[int(num) for num in sub] for sub in matrix]
        
        return matrix 
    
    def konverzija():
        cene = [[int(num) for num in sub] for sub in matrix]
        print("Konerzija")
        print(cene)
        return cene
    
    
    def prikazi_matricu(r,c):
        Label(ws, text="Unesite cene :", font=('arial', 10)).place(x=200, y=365)
    
        x2 = 0
        y2 = 0
        rows, cols = (r,c)
        for i in range(rows):
            # append an empty list to your two arrays
            # so you can append to those later
            text_var.append([])
            entries.append([])
            for j in range(cols):
                # append your StringVar and Entry
                text_var[i].append(StringVar())
                entries[i].append(Entry(ws, textvariable=text_var[i][j],width=3))
                entries[i][j].place(x=200 + x2, y=410 + y2)
                x2 += 30
    
            y2 += 30
            x2 = 0
        
#    button6= Button(ws,text="Prikazi matricu", bg='bisque3', width=15, command= lambda: prikazi_matricu(len(ponuda), len(potraznja)))
#    button6.place(x=190,y=330)
    button6 = Button(ws, text='Prikaži matricu', command= lambda: prikazi_matricu(len(ponuda), len(potraznja)), bg='brown', fg='white', font=('helvetica', 9, 'bold'))
    canvas.create_window(250, 340, window=button6)
            
            
#    button3= Button(ws,text="Potvrdi2", bg='bisque3', width=15, command=lambda : get_mat(len(ponuda), len(potraznja)))
#    button3.place(x=110,y=520)
    button3 = Button(ws, text='Unesi cene', command=lambda : get_mat(len(ponuda), len(potraznja)), bg='brown', fg='white', font=('helvetica', 9, 'bold'))
    canvas.create_window(420, 430, window=button3)
 
    
#    button5= Button(ws,text="Potvrdi3", bg='bisque3', width=15, command=konverzija)
#    button5.place(x=300,y=520)
    button5 = Button(ws, text='Potvrdi', command=konverzija, bg='brown', fg='white', font=('helvetica', 9, 'bold'))
    canvas.create_window(420, 470, window=button5)
    
    
    def najmanja_cena(ponuda, potraznja, cene):
        i = 0
        j = 0
        ponudaCopy = ponuda.copy()
        potraznjaCopy = potraznja.copy()
        pocetno_res_nc=[]
        ceneCopy = cene.copy()
        
        while len(pocetno_res_nc) != len(ponuda) * len(potraznja):
            minimum = np.min(ceneCopy, axis=None)
            for ii, red in enumerate(ceneCopy):
                for jj, cena in enumerate(red):
                    if cena == minimum:
                        i,j = ii, jj
                       
            
            najmanja_vrednost = min(ponudaCopy[i], potraznjaCopy[j])
            pocetno_res_nc.append(((i, j), najmanja_vrednost))
            ponudaCopy[i] -= najmanja_vrednost
            potraznjaCopy[j] -= najmanja_vrednost
            
            ceneCopy[i][j] = math.inf
        
        
        pocetnoResenjeA = []
        for p, v in enumerate(pocetno_res_nc):
            if(v[1] != 0):
                pocetnoResenjeA.append(v)
        
        pocetnoResenjeA_copy = pocetnoResenjeA.copy()
        str1 = str(pocetnoResenjeA_copy).strip('[]')
        lab3 =Label(ws, text= 'Promenljive su sledeće ',font=('helvetica', 10))
        canvas.create_window(250, 580, window=lab3)
        lab4 =Label(ws, text= str1,font=('helvetica', 10))
        canvas.create_window(250, 600, window=lab4)
                
        return pocetnoResenjeA
    
    def transportniProblem_najmanje_cene(ponuda, potraznja, cene, penali):
        cene = [[20,11,15,13],
                [17,14,12,13],
                [15,12,18,18]]
        balansiranaPonuda, balansiranaPotraznja, balansiraneCene = (balansiranje_problema(ponuda, potraznja, cene, penali))
        pocetnoResenje = najmanja_cena(balansiranaPonuda, balansiranaPotraznja, cene)
        optimalnoResenje = optimalno_najmanje_cene(pocetnoResenje, balansiraneCene)
    
        matrica = np.zeros((len(balansiraneCene), len(balansiraneCene[0])))
        for (i, j), v in optimalnoResenje:
            matrica[i][j] = v
        trosakTransporta = odrediTrosak_najmanje_cene(balansiraneCene, matrica)
        return matrica, trosakTransporta
    
    
    
    def optimalno_najmanje_cene(pocetnoResenje, cene):
        cene = [[20,11,15,13],
                [17,14,12,13],
                [15,12,18,18]]
        u,v = potencijali(pocetnoResenje, cene)
        cij = nove_cene_nebaznih_promenljivih(pocetnoResenje, cene, u, v)
    
        if(provera(cij)):
            najnegativnije = najmanja_nebazna_cena(cij)
            petlja = pronadjiPetlju([p for p,v in pocetnoResenje], najnegativnije, [najnegativnije])
            return optimalno_najmanje_cene(novoResenje(pocetnoResenje, petlja), cene)
        return pocetnoResenje
    
    def odrediTrosak_najmanje_cene(cene, resenje):
        suma = 0
        cene = [[20,11,15,13],
                [17,14,12,13],
                [15,12,18,18]]
        for i, red in enumerate(cene):
            for j, cena in enumerate(red):
                suma += cena*resenje[i][j]
        return suma
    
    def prikazi_optimalno_n_c(ponuda, potraznja,pocetno_res_nc, cene):
        u_nc,v_nc = potencijali(pocetno_res_nc, cene)
        tezinski_koeficijenti_nc = nove_cene_nebaznih_promenljivih(pocetno_res_nc, cene, u_nc, v_nc)
        nc_nebaznih_nc = nove_cene_nebaznih_promenljivih(pocetno_res_nc, cene, u_nc, v_nc)
        end_of_algorihm_nc = provera(nc_nebaznih_nc)
        najnegativnije_nc = najmanja_nebazna_cena(nc_nebaznih_nc)
        petlja_nc = pronadjiPetlju([p for p,v in pocetno_res_nc], najnegativnije_nc, [najnegativnije_nc])
        novo_nc = novoResenje(pocetno_res_nc, petlja_nc)
        penali = []
        transport, cena = transportniProblem_najmanje_cene(ponuda, potraznja, cene, penali)
       
        str2 = str(transport)
        lab10 =Label(ws, text= str2,font=('helvetica', 10))
        canvas.create_window(250, 670, window=lab10)
        
        str3 = str(cena)
        lab11 =Label(ws, text= "Cena transporta: " + str3,font=('helvetica', 10))
        canvas.create_window(250, 740, window=lab11)
        
    
    penali0 = []
    cene0 = konverzija()
    ponuda, potraznja, cene = balansiranje_problema(ponuda0,potraznja0,cene0, penali0)
    print(cene)
    pocetnoResenjeA = najmanja_cena(ponuda, potraznja,cene)
    print(pocetnoResenjeA)
    
    button4 = Button(ws, text='Računaj', command=lambda: najmanja_cena(ponuda, potraznja, konverzija()), bg='brown', fg='white', font=('helvetica', 9, 'bold'))
    canvas.create_window(250, 550, window=button4)
    
    cene1 = [[20,11,15,13],
                [17,14,12,13],
                [15,12,18,18]]
    
    button11 = Button(ws, text='Optimizuj', command=lambda: prikazi_optimalno_n_c(ponuda, potraznja,pocetnoResenjeA, cene1), bg='brown', fg='white', font=('helvetica', 9, 'bold'))
    canvas.create_window(250, 630, window=button11)
    
    



def New_Window1():
    ws = Toplevel()
    ws.title("Metod severozapadnog ugla")
    
    canvas =Canvas(ws, width = 500, height = 800,  relief = 'raised')
    canvas.pack()
    
    lab1 =Label(ws, text='Metod severozapadnog ugla')
    lab1.config(font=('helvetica', 14))
    canvas.create_window(250, 25, window=lab1)
    
    lab2 =Label(ws, text='Unesite potražnje:')
    lab2.config(font=('helvetica', 10))
    canvas.create_window(250, 100, window=lab2)
    
    entry1 =Entry (ws) 
    canvas.create_window(250, 140, window=entry1)
    
    potraznja0 = []
    def potraznja_f ():
        a1 = entry1.get()
        potraznja0.append(a1)
        for i in range(0, len(potraznja0)):
            potraznja0[i] = int(potraznja0[i])
        
          
    button = Button(ws, text='Dodaj', command=potraznja_f, bg='brown', fg='white', font=('helvetica', 9, 'bold'))
    canvas.create_window(250, 180, window=button)
    
    
    lab3 =Label(ws, text='Unesite ponude:')
    lab3.config(font=('helvetica', 10))
    canvas.create_window(250, 220, window=lab3)
    
    entry2 =Entry (ws) 
    canvas.create_window(250, 260, window=entry2)
    
    ponuda0 = []
    def ponuda_f ():
        a2 = entry2.get()
        ponuda0.append(a2)
        for i in range(0, len(ponuda0)):
            ponuda0[i] = int(ponuda0[i])
    
    button2 = Button(ws, text='Dodaj', command=ponuda_f, bg='brown', fg='white', font=('helvetica', 9, 'bold'))
    canvas.create_window(250, 300, window=button2)
    
    text_var = []
    entries = []
    matrix = []
    #potraznja = []
    potraznja_string = []
    
    
    # callback function to get your StringVars
    def get_mat(rows,cols):
        for i in range(rows):
            matrix.append([])
            for j in range(cols):
                matrix[i].append(text_var[i][j].get())
        
        
        print(matrix)
        #matrix = [[int(num) for num in sub] for sub in matrix]
        
        return matrix 
    
    def konverzija():
        cene = [[int(num) for num in sub] for sub in matrix]
        print("funkcija")
        print(cene)
        return cene
    
    
    def prikazi_matricu(r,c):
        Label(ws, text="Unesite cene :", font=('arial', 10)).place(x=200, y=365)
    
        x2 = 0
        y2 = 0
        rows, cols = (r,c)
        for i in range(rows):
            # append an empty list to your two arrays
            # so you can append to those later
            text_var.append([])
            entries.append([])
            for j in range(cols):
                # append your StringVar and Entry
                text_var[i].append(StringVar())
                entries[i].append(Entry(ws, textvariable=text_var[i][j],width=3))
                entries[i][j].place(x=200 + x2, y=410 + y2)
                x2 += 30
    
            y2 += 30
            x2 = 0
        
#    button6= Button(ws,text="Prikazi matricu", bg='bisque3', width=15, command= lambda: prikazi_matricu(len(ponuda), len(potraznja)))
#    button6.place(x=190,y=330)
    button06 = Button(ws, text='Prikaži matricu', command= lambda: prikazi_matricu(len(ponuda), len(potraznja)), bg='brown', fg='white', font=('helvetica', 9, 'bold'))
    canvas.create_window(250, 340, window=button06)
            
            
#    button3= Button(ws,text="Potvrdi2", bg='bisque3', width=15, command=lambda : get_mat(len(ponuda), len(potraznja)))
#    button3.place(x=110,y=520)
    button03 = Button(ws, text='Unesi cene', command=lambda : get_mat(len(ponuda), len(potraznja)), bg='brown', fg='white', font=('helvetica', 9, 'bold'))
    canvas.create_window(420, 430, window=button03)
 
    
#    button5= Button(ws,text="Potvrdi3", bg='bisque3', width=15, command=konverzija)
#    button5.place(x=300,y=520)
    button05 = Button(ws, text='Potvrdi', command=konverzija, bg='brown', fg='white', font=('helvetica', 9, 'bold'))
    canvas.create_window(420, 470, window=button05)
    
    
    pocetnoResenje_S = []
    def severozapadni_ugao(ponuda, potraznja):
        i = 0
        j = 0
        #pocetnoResenje_SZ = []
        ponudaCopy = ponuda.copy()
        potraznjaCopy = potraznja.copy()
        while len(pocetnoResenje_S) < len(ponuda) + len(potraznja) - 1:
            pon = ponudaCopy[i]
            potr = potraznjaCopy[j]
            v = min(pon, potr)
            ponudaCopy[i] -= v
            potraznjaCopy[j] -= v
            pocetnoResenje_S.append(((i, j), v))
            if ponudaCopy[i] == 0 and i < len(ponuda) - 1:
                i += 1
            elif potraznjaCopy[j] == 0 and j < len(potraznja) - 1: 
                j += 1
                
        pocetnoResenje_S_copy = pocetnoResenje_S.copy()
        str1 = str(pocetnoResenje_S_copy).strip('[]')
        lab3 =Label(ws, text= 'Promenljive su sledeće ',font=('helvetica', 10))
        canvas.create_window(250, 570, window=lab3)
        lab4 =Label(ws, text= str1,font=('helvetica', 10))
        canvas.create_window(250, 590, window=lab4)
        print(pocetnoResenje_S)
        return pocetnoResenje_S
    

    
    def transportniProblem(ponuda, potraznja, cene, penali):
        balansiranaPonuda, balansiranaPotraznja, balansiraneCene = (balansiranje_problema(ponuda, potraznja, cene, penali))
        pocetnoResenje = severozapadni_ugao(balansiranaPonuda, balansiranaPotraznja)
        optimalnoResenje = optimalno(pocetnoResenje, balansiraneCene)
    
        matrica = np.zeros((len(balansiraneCene), len(balansiraneCene[0])))
        for (i, j), v in optimalnoResenje:
            matrica[i][j] = v
        trosakTransporta = odrediTrosak(balansiraneCene, matrica)
        return matrica, trosakTransporta
    
    def optimalno(pocetnoResenje, cene):
        u,v = potencijali(pocetnoResenje, cene)
        cij = nove_cene_nebaznih_promenljivih(pocetnoResenje, cene, u, v)
    
        if(provera(cij)):
            najnegativnije = najmanja_nebazna_cena(cij)
            petlja = pronadjiPetlju([p for p,v in pocetnoResenje], najnegativnije, [najnegativnije])
            return optimalno(novoResenje(pocetnoResenje, petlja), cene)
        return pocetnoResenje
    
    def odrediTrosak(cene, resenje):
        suma = 0
        for i, red in enumerate(cene):
            for j, cena in enumerate(red):
                suma += cena*resenje[i][j]
        return suma
    
    def prikazi_optimalno(ponuda, potraznja,pocetnoResenje_SZ, cene):
        print(cene)
        u,v = potencijali(pocetnoResenje_SZ, cene)
        print(u)
        print(v)
        tezinski_koeficijenti = nove_cene_nebaznih_promenljivih(pocetnoResenje_SZ, cene, u, v)
        nc_nebaznih = nove_cene_nebaznih_promenljivih(pocetnoResenje_SZ, cene, u, v)
        end_of_algorihm = provera(nc_nebaznih)
        najnegativnije = najmanja_nebazna_cena(nc_nebaznih)
        petlja = pronadjiPetlju([p for p,v in pocetnoResenje_SZ], najnegativnije, [najnegativnije])
        novo = novoResenje(pocetnoResenje_SZ, petlja)
        penali = []
        transport, cena = transportniProblem(ponuda, potraznja, cene, penali)
        
        str2 = str(transport)
        lab10 =Label(ws, text= str2,font=('helvetica', 10))
        canvas.create_window(250, 690, window=lab10)
        
        str3 = str(cena)
        lab11 =Label(ws, text= "Cena transporta: " + str3,font=('helvetica', 10))
        canvas.create_window(250, 750, window=lab11)
    

    cene0 = konverzija()    
#    cene0 = [[10,12,0],
#                 [8,4,3],
#                 [6,9,4],
#                 [7,8,5]]
    penali0 = []

    
    ponuda, potraznja, cene = balansiranje_problema(ponuda0,potraznja0,cene0, penali0)
    print(cene)
    pocetnoResenje_SZ = severozapadni_ugao(ponuda, potraznja)
    print(pocetnoResenje_SZ)
    print(cene)
    
    button3 = Button(ws, text='Računaj', command=lambda: severozapadni_ugao(ponuda, potraznja), bg='brown', fg='white', font=('helvetica', 9, 'bold'))
    canvas.create_window(250, 540, window=button3) 
    

    button04 = Button(ws, text='Optimizuj', command=lambda: prikazi_optimalno(ponuda, potraznja,pocetnoResenje_SZ, konverzija()), bg='brown', fg='white', font=('helvetica', 9, 'bold'))
    canvas.create_window(250, 630, window=button04)
    
    
    

def New_Window2():
    ws = Toplevel()
    ws.title("Vogelov metod")

    canvas =Canvas(ws, width = 500, height = 800,  relief = 'raised')
    canvas.pack()
    
    lab1 =Label(ws, text='Vogelov metod')
    lab1.config(font=('helvetica', 14))
    canvas.create_window(250, 40, window=lab1)
    
        #Ova f-ja racuna razlike po vrstama
    def razlika_po_redovima(cene, potraznja):
        cene1 = np.transpose(cene)  
        cene1 = [sorted(i) for i in cene1] 
       # print(cene)
        red_razlika = [] 
        for i in range(len(cene1)):
                if(potraznja[i] != 0): 
                    red_razlika.append(cene1[i][1] - cene1[i][0])
                else:
                    red_razlika.append(0)
        return red_razlika
      #  print(red_razlika)
    
    #Ova f-ja racuna razlike po kolonama
    def razlika_po_kolonama(cene, ponuda):
    
        cene = [sorted(i) for i in cene]
        razlika_kolona=[]
        for i in range(len(cene)):
                if(ponuda[i] != 0):
                    razlika_kolona.append(cene[i][1]-cene[i][0])
                else:
                    razlika_kolona.append(0)
        return razlika_kolona
    
    
    #Ovde nam se trazi najveca razlika
    def indeks_sa_najvecom_razlikom(razlika_kolona, red_razlika):
        max_red = max(red_razlika)
        #print("Najveci u redu" + " " + str(max_red))
        max_kolona = max(razlika_kolona)
        red = False
        
        if(max_red > max_kolona):
            index_max = red_razlika.index(max_red) 
            red=True
           
        else:
            index_max=razlika_kolona.index(max_kolona)
        #print("INdex max" + " " + str(index_max))        
        return index_max,red
        
    #Ovde pronalazi najmanji element kroz koji ce pokusati da provuce najvecu cenu 
    def Uzmi_minimum_u_matrici_red(index_max, ponuda, potraznja, red, cene):
        minimalni_index = 0
        
        if(red == True):
            l = []
            for ind1 in range(0, len(cene)):
               l.append(cene[ind1][index_max]) #uzima sve elemente iz kolone/reda
            lm = min(l)
            print("Lista l" + " " + str(l))  
    
            minimalni_index = l.index(min(l))
            return minimalni_index,index_max
        else:
            print(cene[index_max])
            lm=min(cene[index_max])
            l=cene[index_max]
            minimalni_index = l.index(lm)
            print("Lista l" + " " + str(l))  
           
            return index_max, minimalni_index
        
        
    
    
    def nadjiPocetnoResenje(pocetnoResenje_VOGEL, cene, ponuda, potraznja, i, j, red):
       
        if(ponuda[i] > potraznja[j]):
            pocetnoResenje_VOGEL[i][j]=potraznja[j]
            ponuda[i]=ponuda[i]-potraznja[j]
            potraznja[j]=0
            for ind in range(0,len(ponuda)):
              cene[ind][j]=99999
    
        else:
            pocetnoResenje_VOGEL[i][j]=ponuda[i]
            potraznja[j]=potraznja[j]-ponuda[i]
            ponuda[i]=0
            for ind in range(0, len(ponuda)):
                cene[i][ind] = 99 
        return pocetnoResenje_VOGEL,cene
    
#    cene = [[20,11,15,13],
#            [17,14,12,13],
#            [15,12,18,18]]
    
    potraznja = []
    def potraznja_f ():
        a1 = entry1.get()
        potraznja.append(a1)
        for i in range(0, len(potraznja)):
            potraznja[i] = int(potraznja[i])
    
    lab2 =Label(ws, text='Unesite potražnje:')
    lab2.config(font=('helvetica', 10))
    canvas.create_window(250, 100, window=lab2)
    
    entry1 =Entry (ws) 
    canvas.create_window(250, 140, window=entry1)
          
    button = Button(ws, text='Dodaj potražnju', command=potraznja_f, bg='brown', fg='white', font=('helvetica', 9, 'bold'))
    canvas.create_window(250, 180, window=button)
    
    lab3 =Label(ws, text='Unesite ponude:')
    lab3.config(font=('helvetica', 10))
    canvas.create_window(250, 220, window=lab3)
    
    entry2 =Entry (ws) 
    canvas.create_window(250, 260, window=entry2)
    
    ponuda = []
    def ponuda_f ():
        a2 = entry2.get()
        ponuda.append(a2)
        for i in range(0, len(ponuda)):
            ponuda[i] = int(ponuda[i])
        
        
    
    button2 = Button(ws, text='Dodaj ponudu', command=ponuda_f, bg='brown', fg='white', font=('helvetica', 9, 'bold'))
    canvas.create_window(250, 300, window=button2)
    
#   ponuda = [2,6,7]
#    potraznja = [3,3,4,5]
    pocetnoResenje_VOGEL=np.zeros((3,4))
    

    
    text_var = []
    entries = []
    matrix = []
    #potraznja = []
    potraznja_string = []
    
    
    # callback function to get your StringVars
    def get_mat(rows,cols):
        for i in range(rows):
            matrix.append([])
            for j in range(cols):
                matrix[i].append(text_var[i][j].get())
        
        
        print(matrix)
        #matrix = [[int(num) for num in sub] for sub in matrix]
        
        return matrix 
    
    def konverzija():
        cene = [[int(num) for num in sub] for sub in matrix]
        print("funkcija")
        print(cene)
        return cene
    
    
    def prikazi_matricu(r,c):
        Label(ws, text="Unesite cene :", font=('arial', 10)).place(x=200, y=365)
    
        x2 = 0
        y2 = 0
        rows, cols = (r,c)
        for i in range(rows):
            # append an empty list to your two arrays
            # so you can append to those later
            text_var.append([])
            entries.append([])
            for j in range(cols):
                # append your StringVar and Entry
                text_var[i].append(StringVar())
                entries[i].append(Entry(ws, textvariable=text_var[i][j],width=3))
                entries[i][j].place(x=200 + x2, y=410 + y2)
                x2 += 30
    
            y2 += 30
            x2 = 0
        
#    button6= Button(ws,text="Prikazi matricu", bg='bisque3', width=15, command= lambda: prikazi_matricu(len(ponuda), len(potraznja)))
#    button6.place(x=190,y=330)
    button06 = Button(ws, text='Prikaži matricu', command= lambda: prikazi_matricu(len(ponuda), len(potraznja)), bg='brown', fg='white', font=('helvetica', 9, 'bold'))
    canvas.create_window(250, 340, window=button06)
            
            
#    button3= Button(ws,text="Potvrdi2", bg='bisque3', width=15, command=lambda : get_mat(len(ponuda), len(potraznja)))
#    button3.place(x=110,y=520)
    button03 = Button(ws, text='Unesi cene', command=lambda : get_mat(len(ponuda), len(potraznja)), bg='brown', fg='white', font=('helvetica', 9, 'bold'))
    canvas.create_window(420, 430, window=button03)
 
    
#    button5= Button(ws,text="Potvrdi3", bg='bisque3', width=15, command=konverzija)
#    button5.place(x=300,y=520)
    button05 = Button(ws, text='Potvrdi', command=konverzija, bg='brown', fg='white', font=('helvetica', 9, 'bold'))
    canvas.create_window(420, 470, window=button05)
    
    cene = konverzija()
    button4 = Button(ws, text='Računaj', command=lambda: racunaj(pocetnoResenje_VOGEL, konverzija(), ponuda, potraznja), bg='brown', fg='white', font=('helvetica', 9, 'bold'))
    canvas.create_window(250, 550, window=button4)
    
    def racunaj(pocetnoResenje_VOGEL, cene, ponuda, potraznja):
        #U ovom delu je ograniceno koliko metoda treba da se izvrsava, tj dokle god je potraznja
        #razlicita od 0
        while(max(potraznja)!=0):
        
            potraznjaA = [i for i, v in enumerate(potraznja) if v == 0]
            print(potraznjaA)
            if len(potraznjaA)==len(potraznja)-1:
                # Ovdje je dio kada ostane ne kraju jedna kolona
                potraznja0 = [i for i, element in enumerate(potraznja) if element != 0]
        
                red=True
                index_max=potraznja0[0]
                for k in range(len(ponuda)):
                    pocetnoResenje_VOGEL[k][index_max]+=ponuda[k]
                potraznja[index_max]=0
        
        
            else:
                razlika_kolona = razlika_po_kolonama(cene,ponuda)
                red_razlika = razlika_po_redovima(cene,potraznja)
                index_max,red = indeks_sa_najvecom_razlikom(razlika_kolona,red_razlika)
                i,j=Uzmi_minimum_u_matrici_red(index_max,ponuda,potraznja,red, cene)
        
                pocetnoResenje_VOGEL,cene=nadjiPocetnoResenje(pocetnoResenje_VOGEL,cene,ponuda,potraznja,i,j,red)
          
            
        print(pocetnoResenje_VOGEL)
        str2 = str(pocetnoResenje_VOGEL)
        lab10 =Label(ws, text= str2,font=('helvetica', 10))
        canvas.create_window(250, 600, window=lab10)
        
        p=[]
        for i, red in enumerate(pocetnoResenje_VOGEL):
                for j, vrednost in enumerate(red):
                    if(vrednost!=0):
                        p.append(((i,j),vrednost))
                        
        
        pocetnoResenje_SZ = [(x, int(y)) for x, y in p]


ws = Tk()
ws.title("Transportni problem")

canvas =Canvas(ws, width = 600, height = 400,  relief = 'raised')
canvas.pack()

lab =Label(ws, text='Transportni problem')
lab.config(font=('helvetica', 20))
canvas.create_window(310, 30, window=lab)


lab1 =Label(ws, text='Metod severozapadnog ugla')
lab1.config(font=('helvetica', 14))
canvas.create_window(310, 100, window=lab1)
#button5= Button(ws,text="Metod severozapadnog ugla", bg='bisque3', width=30, command=lambda: New_Window1())
#button5.place(x=200,y=130)

button5 = Button(ws, text='Metod severozapadnog ugla', command=lambda: New_Window1(), bg='brown', fg='white', font=('helvetica', 10, 'bold'))
canvas.create_window(310, 150, window=button5)

lab2 =Label(ws, text='Metod najmanjih cena')
lab2.config(font=('helvetica', 14))
canvas.create_window(310, 200, window=lab2)
#button6= Button(ws,text="Metod najmanjih cena", bg='bisque3', width=30, command=lambda: New_Window())
#button6.place(x=200,y=230)
button6 = Button(ws, text='    Metod najmanjih cena     ', command=lambda: New_Window(), bg='brown', fg='white', font=('helvetica', 10, 'bold'))
canvas.create_window(310, 250, window=button6)


lab3 =Label(ws, text='Vogelov metod')
lab3.config(font=('helvetica', 14))
canvas.create_window(310, 300, window=lab3)
#button10= Button(ws,text="Vogelov metod", bg='bisque3', width=30, command=lambda: New_Window2())
#button10.place(x=200,y=330)
button10 = Button(ws, text='          Vogelov metod          ', command=lambda: New_Window2(), bg='brown', fg='white', font=('helvetica', 10, 'bold'))
canvas.create_window(310, 350, window=button10)



ws.mainloop()