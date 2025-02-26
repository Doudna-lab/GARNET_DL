import numpy as np
from numpy.linalg import norm,det,eig
import random
import math

#===============================================================================
def dist(p1, p2):
    if (isinstance(p1, (tuple,list))):
        p1 = np.array(p1)
    if (isinstance(p2, (tuple,list))):
        p2 = np.array(p2)
    return norm(p1-p2)

#===============================================================================
def angle(p1, p2, p3):
    if (isinstance(p1, (tuple,list))):
        p1 = np.array(p1)
    if (isinstance(p2, (tuple,list))):
        p2 = np.array(p2)
    if (isinstance(p3, (tuple,list))):
        p3 = np.array(p3)
    v1=p1-p2
    v2=p3-p2
    return np.arccos(np.dot(v1,v2)/norm(v1)/norm(v2))

#===============================================================================
def dihe(p1, p2, p3, p4):
    if (isinstance(p1, (tuple,list))):
        p1 = np.array(p1)
    if (isinstance(p2, (tuple,list))):
        p2 = np.array(p2)
    if (isinstance(p3, (tuple,list))):
        p3 = np.array(p3)
    if (isinstance(p4, (tuple,list))):
        p4 = np.array(p4)
    v1=p1-p2
    v2=p4-p3
    rv=p3-p2
    n1=np.cross(rv,v1)
    n2=np.cross(rv,v2)
    theta=np.arccos(np.dot(n1,n2)/(norm(n1)*norm(n2)))
    s=np.dot(np.cross(n1,n2),rv)
    if s<0:
        theta=2*np.pi-theta
    return theta

#===============================================================================
def translate(A, p0, p1):
    if (isinstance(p0, (tuple,list))):
        p0 = np.array(p0)
    if (isinstance(p1, (tuple,list))):
        p1 = np.array(p1)

    delta=p1-p0;

    return A+np.ones((A.shape[0],1))*delta

#===============================================================================
def rotate(A, p0, p1, theta):
    """
    theta: rotation angle (in rad)
    """
    if (isinstance(p0, (tuple,list))):
        p0 = np.array(p0)
    if (isinstance(p1, (tuple,list))):
        p1 = np.array(p1)

    # translate A to origin
    B = translate(A, p0, (0.,0.,0.))

    rv = p1 - p0
    rv /= norm(rv)
    x,y,z = rv
    R = np.zeros((3,3),'d')

    # setup rotation matrix
    R[0,0] = 1+(1-np.cos(theta))*(x*x-1)
    R[1,0] = -z*np.sin(theta)+(1-np.cos(theta))*x*y
    R[2,0] = y*np.sin(theta)+(1-np.cos(theta))*x*z
    R[0,1] = z*np.sin(theta)+(1-np.cos(theta))*x*y
    R[1,1] = 1+(1-np.cos(theta))*(y*y-1)
    R[2,1] = -x*np.sin(theta)+(1-np.cos(theta))*y*z
    R[0,2] = -y*np.sin(theta)+(1-np.cos(theta))*x*z
    R[1,2] = x*np.sin(theta)+(1-np.cos(theta))*y*z
    R[2,2] = 1+(1-np.cos(theta))*(z*z-1)

    # rotate A and translate it back
    return translate(np.dot(B,R), (0.,0.,0.), p0)

#===============================================================================
def reorient(A, pnt0, pnt1, ref0, ref1):
    if (isinstance(pnt0, (tuple,list))):
        pnt0 = np.array(pnt0)
    if (isinstance(pnt1, (tuple,list))):
        pnt1 = np.array(pnt1)
    if (isinstance(ref0, (tuple,list))):
        ref0 = np.array(ref0)
    if (isinstance(ref1, (tuple,list))):
        ref1 = np.array(ref1)

    v1 = pnt1 - pnt0
    v2 = ref1 - ref0

    B = translate(A, pnt0, ref0)
    p0 = ref0
    p1 = p0 + np.cross(v1, v2)
    # cross vector always come up with vector of angle [0, pi)
    theta = np.arccos(np.dot(v1,v2)/(norm(v1)*norm(v2)))

    return rotate(B, p0, p1, theta)

#===============================================================================
def lsqfit(A, B):
    """
    Fit A onto B;
    Return rot matrix and rmsd; return None on failure
    A (input): pre-centered
    B (input): pre-centered
    rot (output): a 3X3 matrix
    rmsd (output): rmsd
    """
    U = np.dot(B.T, A)
    d = det(U)

    # either of atom sets lies in one plane or one line
    if (d==0.0):
        return None,None;

    Omega = np.zeros((6, 6))
    Omega[0:3, 3:6] = U
    Omega[3:6, 0:3] = U.T

    w,v = eig(Omega)
    perm = np.argsort(w)  # to sort in descending order, use argsort(-w)
    v = v[:, perm]

    if d > 0.0:
        rot = v[3:6, 5].reshape(-1,1) * v[0:3, 5] + \
              v[3:6, 4].reshape(-1,1) * v[0:3, 4] + \
              v[3:6, 3].reshape(-1,1) * v[0:3, 3]
    else:
        rot = v[3:6, 5].reshape(-1,1) * v[0:3, 5] + \
              v[3:6, 4].reshape(-1,1) * v[0:3, 4] - \
              v[3:6, 3].reshape(-1,1) * v[0:3, 3]

    rot *= 2.0
    A = np.dot(A,rot) - B
    rmsd = np.sqrt(np.mean(np.sum(A**2, axis=1)))

    return rot,rmsd


######################################################################
#              Random generator on (sphere surface)                  #
# author: Manuel Metz                                                #
# organization: Argelander-Institut fuer Astronomie, University Bonn #
# contact: mmetz @ astro.uni-bonn.de                                 #
# license: GPL2                                                      #
# version: 0.1                                                       #
# data: 2008-04-15                                                   #
#                                                                    #
# Sphere Point Picking algorithm, see                                #
#   http://mathworld.wolfram.com/SpherePointPicking.html             #
#                                                                    #
######################################################################

def rndSphere():
    """Generate a random point on a unit sphere, equally spaced on the
    surface of the sphere. Returns cartesian coordinates of a vector.
    """
    sph = [0,0,0]
    
    sph[2] = random.uniform(-1.0,1.0)
    z2     = math.sqrt(1.0 - sph[2]*sph[2])
    phi    = (2. * math.pi) * random.random()
    sph[0] = z2 * math.cos(phi)
    sph[1] = z2 * math.sin(phi)
    
    return sph


def xrndSphere(n):
    """Generator, create n random points on a unit sphere. This is
    very useful for looping:
    
        for r in xrndSphere(10):
            print(r)
    """
    for i in range(n):
        yield rndSphere()


def arndSphere(N):
    """Generate N random points on a unit sphere, equally spaced on the
    surface of the sphere, and return them as columns of an array.
    """
    sph = np.empty( (N,3), np.float64 )
    
    sph[:,2] = np.random.uniform(-1.0,1.0,N) # z-coordinates
    z2 = np.sqrt(1.0 - sph[:,2]**2)
    phi = (2.0 * math.pi) * np.random.random( N )
    sph[:,0] = z2 * np.cos(phi) # x 
    sph[:,1] = z2 * np.sin(phi) # y
    
    return sph
