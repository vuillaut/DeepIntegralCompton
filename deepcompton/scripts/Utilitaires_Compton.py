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


# ______ Ajout Alexandre

Ee = 511
z_isgri = 0 # position plan ISGRI -> 0
z_picsit = -8.68  # position plan PICSIT -> distance entre les deux plans


def make_cone(x, z_isgri=z_isgri, z_picsit=z_picsit, Ee=Ee):
    """
    Single cone reconstruction
    :param x: `numpy.ndarray`
        (x, y, z)
    :param z_isgri: float
    :param z_picsit: float
    :param Ee: float
    :return:
        theta, phi, cotheta
    """
    x1cur = z_isgri
    x2cur = z_picsit

    energ1cur = x[0]
    energ2cur = x[1]

    y1cur = x[7]
    z1cur = -x[6]

    y2cur = x[9]
    z2cur = -x[8]

    E0 = energ1cur + energ2cur

    Ec = E0 / (1 + 2 * E0 / Ee)

    # why this condition ??
    if (energ2cur >= Ec) and (energ1cur <= E0 - Ec):
        cotheta = cottheta(energ1cur, E0 - energ1cur)

        A = np.array([x1cur, y1cur, z1cur])
        B = np.array([x2cur, y2cur, z2cur])

        theta =colatitudeaxe(B, A)
        phi = longitudeaxe(B, A)

        return np.array([theta, phi, cotheta])
    else:
        return None


def make_cone_density(theta_source, phi_source, z_isgri, z_picsit, precision=5000., density_precision=2., r=1e14,
                      max_cones=2000000, lon_max=360., lat_max=90., progress=True, datadir=None):
    if datadir is None:
        name = "./save_Compton/theta_" + str(theta_source) + "_phi_" + str(phi_source) + ".npy"
    else:
        name = "{}/theta_".format(datadir) + str(theta_source) + "_phi_" + str(phi_source) + ".npy"

    X = np.load(name).astype(np.float64)
    N = X.shape[0]
    # if empty data return None
    if N == 0:
        return None

    # density grid
    density = np.zeros((int(lon_max / density_precision), int(lat_max / density_precision)))

    # cone counter
    ncones = 0

    # for each row in the data create a cone
    #if progress:
        #progress_msg = "Loading cones, theta:{}, phi:{}".format(theta_source, phi_source)
        #printProgressBar(0, N, prefix=progress_msg, suffix='Complete', length=50)
    for i in range(N):
        # while cone count is not reached
        if ncones < max_cones:
            cone = make_cone(X[i, :], z_isgri, z_picsit)
            if cone is not None:
                [theta, phi, cotheta] = cone
                x1cur = z_isgri
                x2cur = z_picsit

                y1cur = X[i, 7]
                z1cur = -X[i, 6]
                y2cur = X[i, 9]
                z2cur = -X[i, 8]

                colat = colatconer(r, x1cur, y1cur, z1cur, theta, phi, cotheta, precision)
                longit = longitconer(r, x1cur, y1cur, z1cur, theta, phi, cotheta, precision)

                hemisphere = (colat < 90)
                longit = longit[hemisphere]
                colat = colat[hemisphere]
                d = np.zeros((int(lon_max / density_precision), int(lat_max / density_precision)))

                l = ((abs(longit) % lon_max) / density_precision).astype(int)
                c = ((abs(colat) % lat_max) / density_precision).astype(int)

                d[l, c] = 1.

                density += d

                ncones += 1
        #if progress:
            #printProgressBar(i + 1, N, prefix=progress_msg, suffix='Complete', length=50)
    return density