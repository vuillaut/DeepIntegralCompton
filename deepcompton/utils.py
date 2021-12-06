import numpy as np


#UTILITAIRES

def cottheta(E1, E2): #calcul de la cotangente de l'angle Compton

    Me = 511. #masse electron (keV)

    costheta = 1. - Me*E1/(E2*(E1+E2))

    test = costheta == 1.

    costheta = costheta - test

    cot = (costheta/np.sqrt(1.-costheta**2))*(1.-test)

    return cot

def colatitudeaxe(A, B): #;calcul de la colatitude de la direction de l'axe AB

    XAB = B[0] - A[0]
    NormeABcarre = (B[1]-A[1])**2+(B[2]-A[2])**2+(B[0]-A[0])**2
    NormeAB = np.sqrt(NormeABcarre)

    if NormeAB == 0.: #;sécurité si A = B
        colat = 0.
    else:
        colat = np.arccos(XAB/NormeAB)

    return colat

def longitudeaxe(A, B): #;calcul de la longitude de la direction de l'axe AB

    YAB = B[1] - A[1]
    ZAB = B[2] - A[2]

    if (YAB == 0.)*(ZAB == 0.): #;securite si A et B sont sur l'axe X
        phi = 0.
    elif ZAB >= 0:
        phi = np.arccos(YAB/np.sqrt(YAB**2 + ZAB**2))
    else:
      phi = 2.*np.pi - np.arccos(YAB/np.sqrt(YAB**2 + ZAB**2))

    return phi

def calculcolat(A): #;calcul de la colatitude d'un point

    B = np.array([0., 0., 0.])

    theta = colatitudeaxe(B, A)

    return theta


def calculcolattab(Ax, Ay, Az): #calcul de la colatitude d'un tableau de points

    XAB = Ax
    NormeABcarre = (Ay)**2+(Az)**2+(Ax)**2
    NormeAB = np.sqrt(NormeABcarre)

    testnormeAB = NormeAB == 0.

    NormeAB = NormeAB + testnormeAB

    colat = np.arccos(XAB/NormeAB)

    colat = colat*(1.-testnormeAB)

    return colat

def calcullongit(A): #;calcul de la longitude d'un point

    B = np.array([0., 0., 0.])

    phi = longitudeaxe(B, A)

    return phi


def calcullongittab(Ax, Ay, Az): #calcul la longitude d'un tableau de points

    YAB = Ay
    ZAB = Az

    test = (YAB == 0.)*(ZAB == 0.)

    YAB = test + YAB

    testz = (ZAB >= 0.)

    phi = np.arccos(YAB/np.sqrt(YAB**2 + ZAB**2))*testz + (2*np.pi - np.arccos(YAB/np.sqrt(YAB**2 + ZAB**2)))*(1.-testz)

    phi = phi*(1.-test)

    return phi

def rotationtheta(S, theta): #; rotation d'un angle theta autour de l'axe z

    xs = S[0]
    ys = S[1]
    zs = S[2]

    x = np.cos(theta)*xs - np.sin(theta)*ys
    y = np.sin(theta)*xs + np.cos(theta)*ys
    z = zs

    A = np.array([x,y,z])

    return A

def rotationphi(S, phi): #; rotation d'un angle phi autour de l'axe x

    xs = S[0]
    ys = S[1]
    zs = S[2]

    y = np.cos(phi)*ys - np.sin(phi)*zs
    z = np.sin(phi)*ys + np.cos(phi)*zs
    x = xs

    A = np.array([x,y,z])

    return A

def colatitudeaxetab(Ax, Ay, Az, Bx, By, Bz): #;calcul de la colatitude de la direction de l'axe AB

    XAB = Bx - Ax
    NormeABcarre = (By-Ay)**2+(Bz-Az)**2+(Bx-Ax)**2
    NormeAB = np.sqrt(NormeABcarre)

    test = NormeAB == 0.

    NormeAB = NormeAB + test

    colat = np.arccos(XAB/NormeAB)*(1.-test)

    return colat

def longitudeaxetab(Ax, Ay, Az, Bx, By, Bz): #;calcul de la longitude de la direction de l'axe AB

    YAB = By - Ay
    ZAB = Bz - Az

    test1 = (YAB == 0)*(ZAB == 0)

    YAB = test1 + YAB

    test2 = ZAB >= 0

    phi = (np.arccos(YAB/np.sqrt(YAB**2 + ZAB**2)))*test2+(2*np.pi - np.arccos(YAB/np.sqrt(YAB**2 + ZAB**2)))*(1.-test2)

    phi = phi*(1.-test1)

    return phi

#RECONSTRUCTION DES CONES

def coordconexr2(r, xa, ya, za, ct, st, cp, sp, cotheta, alpha): #;ct = cos(theta), st = sin(theta), cp = cos(phi), sp = sin (phi)

    ca = np.cos(alpha)

    rho = np.abs(r*np.sin(np.arctan(1./cotheta)))

    x = rho*(ct*cotheta-st*ca) + xa

    return x

def coordconeyr2(r, xa, ya, za, ct, st, cp, sp, cotheta, alpha): #;ct = cos(theta), st = sin(theta), cp = cos(phi), sp = sin (phi)

    ca = np.cos(alpha)
    sa = np.sin(alpha)

    rho = abs(r*np.sin(np.arctan(1./cotheta)))

    y = rho*(cp*(st*cotheta+ct*ca)-sp*sa) + ya

    return y

def coordconezr2(r, xa, ya, za, ct, st, cp, sp, cotheta, alpha): #;ct = cos(theta), st = sin(theta), cp = cos(phi), sp = sin (phi)

    ca = np.cos(alpha)
    sa = np.sin(alpha)

    rho = abs(r*np.sin(np.arctan(1./cotheta)))

    z = rho*(sp*(st*cotheta+ct*ca)+cp*sa) + za

    return z

def coordconexr(r , xa, ya, za, ct, st, cp, sp, cotheta, alpha): #;ct = cos(theta), st = sin(theta), cp = cos(phi), sp = sin (phi)

    ca = np.cos(alpha)
    sa = np.sin(alpha)

    kx = ct*cotheta-st*ca

    ky = cp*(st*cotheta+ct*ca)-sp*sa

    kz = sp*(st*cotheta+ct*ca)+cp*sa

    a = kx**2 + ky**2 + kz**2

    b = 2*(kx*xa + ky*ya + kz*za)

    c = xa**2 + ya**2 + za**2 - r**2

    delta = b**2 - 4*a*c

    rho = (-b + np.sqrt(delta))/(2.*a)

    x = rho*(ct*cotheta-st*ca) + xa

    return x

def coordconeyr(r, xa, ya, za, ct, st, cp, sp, cotheta, alpha): #;ct = cos(theta), st = sin(theta), cp = cos(phi), sp = sin (phi)

    ca = np.cos(alpha)
    sa = np.sin(alpha)

    kx = ct*cotheta-st*ca

    ky = cp*(st*cotheta+ct*ca)-sp*sa

    kz = sp*(st*cotheta+ct*ca)+cp*sa

    a = kx**2 + ky**2 + kz**2

    b = 2*(kx*xa + ky*ya + kz*za)

    c = xa**2 + ya**2 + za**2 - r**2

    delta = b**2 - 4*a*c

    rho = (-b + np.sqrt(delta))/(2.*a)

    y = rho*(cp*(st*cotheta+ct*ca)-sp*sa) + ya

    return y

def coordconezr(r, xa, ya, za, ct, st, cp, sp, cotheta, alpha): #;ct = cos(theta), st = sin(theta), cp = cos(phi), sp = sin (phi)

    ca = np.cos(alpha)
    sa = np.sin(alpha)

    kx = ct*cotheta-st*ca

    ky = cp*(st*cotheta+ct*ca)-sp*sa

    kz = sp*(st*cotheta+ct*ca)+cp*sa

    a = kx**2 + ky**2 + kz**2

    b = 2*(kx*xa + ky*ya + kz*za)

    c = xa**2 + ya**2 + za**2 - r**2

    delta = b**2 - 4*a*c

    rho = (-b + np.sqrt(delta))/(2.*a)

    z = rho*(sp*(st*cotheta+ct*ca)+cp*sa) + za

    return z

def colatconer(r, xa, ya, za, theta, phi, cotheta, precision): #;calcul de la colatitude de tous les points du cône

    colat = np.zeros((int(precision)))

    alpha = np.arange(int(precision))*2.*np.pi/precision

    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)

    Ax = coordconexr(r, xa, ya, za, ct, st, cp, sp, cotheta, alpha)
    Ay = coordconeyr(r, xa, ya, za, ct, st, cp, sp, cotheta, alpha)
    Az = coordconezr(r, xa, ya, za, ct, st, cp, sp, cotheta, alpha)

    colatitude = calculcolattab(Ax, Ay, Az)

    colat = colatitude*180./np.pi

    return colat

def longitconer(r, xa, ya, za, theta, phi, cotheta, precision): #;calcul de la longitude de tous les points du cône

    longit = np.zeros((int(precision)))

    alpha = np.arange(int(precision))*2.*np.pi/precision

    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)

    Ax = coordconexr(r, xa, ya, za, ct, st, cp, sp, cotheta, alpha)
    Ay = coordconeyr(r, xa, ya, za, ct, st, cp, sp, cotheta, alpha)
    Az = coordconezr(r, xa, ya, za, ct, st, cp, sp, cotheta, alpha)

    longitude = calcullongittab(Ax, Ay, Az)

    longit = longitude*180./np.pi

    return longit