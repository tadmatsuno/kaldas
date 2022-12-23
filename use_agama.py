### This program is to calculate stellar kinematics using agama for a large 
### data set. It is primarily written by T. Matsuno (tadafumi.mn@gmail.com).

import psutil

from astropy import table
import astropy.units as u
from astropy.coordinates import SkyCoord,CartesianDifferential,Galactocentric

import numpy as np
import time

#orbit
import agama

class SolarPosition:
  def __init__(self):
    self.x = -8.127  
    self.y =  0.
    self.z =  0.025
    self.vx =  11.1
    self.vy =  247.97
    self.vz =  7.25  
    self.reference = \
      {'x':  'Gravity collaboration et al. 2018, A&A, 615, L15 ',
       'y':  'By definition',
       'z':  'Juric et al. 2008, ApJ, 673, 864 ',
       'vx': 'Schonrich et al. 2010, MNRAS, 403, 1829 ',
       'vy': 'Reid & Brunthaler, 2004, ApJ, 616, 872  ',
       'vz': 'Schonrich et al. 2010, MNRAS, 403, 1829 '}

  def printsolarpos(self):
     print('x ={0:8.3f} kpc {1:s}'.format(self.x,\
                                   self.reference['x']))
     print('y ={0:8.3f} kpc {1:s}'.format(self.y,\
                                   self.reference['y']))
     print('z ={0:8.3f} kpc {1:s}'.format(self.z,\
                                   self.reference['z']))
     print('vx={0:8.3f} km/s {1:s}'.format(self.vx,\
                                   self.reference['vx']))
     print('vy={0:8.3f} km/s {1:s}'.format(self.vy,\
                                   self.reference['vy']))
     print('vz={0:8.3f} km/s {1:s}'.format(self.vz,\
                                   self.reference['vz']))

def cartesian_2_cylindrical(x,y,z,vx,vy,vz):
  '''
    Output (R, phi, vR, vTHETA, vPHI)
      R^2: distance from the center in x-y plane
         x^2+y^2
      phi: phase in x-y plane
         arctan(y,x)
      vR : velocity in radial direction in x-y plane
         (x*vx + y*vy)/R 
      vTHETA: velocity in theta direction
         (R*vz - z*vR)/r
      vPHI: velocity in phi direction
         (x*vy - y*vx)/R

      where r = sqrt(x^2+y^2+z^2)
  '''
  R =np.sqrt(x**2 + y**2)
  r =np.sqrt(x**2 + y**2 + z**2)
  phi=np.arctan2(y,x)

  vR     = (x*vx + y*vy)/R
  vTHETA = (R*vz - z*vR)/r
  vPHI   = (x*vy - y*vx)/R
  return R, phi, vR, vTHETA, vPHI

def Observed2NormGCFrame(ra,dec,dist,pmra,pmdec,rv,gc_frame,units):
  '''
    Input
      ra, dec   : in degree
      dist      : kpc
      pmra,pmdec: in mas/year
      rv        : km/s
      gc_frame  : astropy.coordinates.Galactocentric object
      units     : dictionary that contains units for L, V, T, M
    Output 
      normalized (x,y,z,vx,vy,vz)
  '''
  _R0 = units['L']
  _V0 = units['V']
  t1 = time.time()
  gccoord = (SkyCoord(ra*u.degree,\
                      dec*u.degree,\
                      dist*u.kpc,\
                      pm_ra_cosdec=pmra*u.mas/u.year,\
                      pm_dec=pmdec*u.mas/u.year,\
                      radial_velocity=rv*u.km/u.s)).transform_to(gc_frame)
  x = (gccoord.x/u.kpc).decompose().value/_R0
  y = (gccoord.y/u.kpc).decompose().value/_R0
  z = (gccoord.z/u.kpc).decompose().value/_R0
  vx = ((gccoord.v_x)/(u.km/u.s)).decompose().value/_V0 
  vy = ((gccoord.v_y)/(u.km/u.s)).decompose().value/_V0
  vz = ((gccoord.v_z)/(u.km/u.s)).decompose().value/_V0 
  print('  {0:10.4f} sec for coordinate transformation'.format(time.time()-t1))
  return x,y,z,vx,vy,vz

def SamplePMraPMdecDist(parallax,pmra,pmdec,\
                        parallax_error,pmra_error,pmdec_error,\
                        parallax_pmra_corr=0.0,\
                        parallax_pmdec_corr=0.0,\
                        pmra_pmdec_corr=0.0,\
                        rscale=None,size=10000):
  '''
    Sampling distance and proper motions 

    Input
      input parameters assume gaia units
      rscale: in kpc 
              For the prior, exp(-d/rscale)*d^2 / (2.0*rscale^3)
              Bailer-Jones+2018, ApJ, 156, 58
              If None, no prior will be used (simple inverse of the parallax).
    Output
      distance, pmra, pmdec
                  
  '''
  # Construct mean vector and covariance matrix
  sigma = np.array([parallax_error,pmra_error,pmdec_error])\
                  .repeat(3).reshape(3,3)
  corr_coeff = np.array(\
    [[1.0                 , parallax_pmra_corr , parallax_pmdec_corr ],
     [parallax_pmra_corr  , 1.0                , pmra_pmdec_corr     ],
     [parallax_pmdec_corr , pmra_pmdec_corr    , 1.0                 ]])
  covariance = corr_coeff * sigma * sigma.T
  mean  = np.array([parallax,pmra,pmdec])

  # Prior function
  if rscale is None:
    fprior = lambda d:np.where(d>0,1.0,0.0)
  elif rscale < 0:
    fprior = lambda d:1.0
  else:
    fprior = lambda d:np.where(d>0,\
                      np.exp(-d/rscale)*d**2.0/(2.0*rscale**3.0),\
                      0.0)

  # Repeat until sufficient number is obtained
  nsample = 0
  while (nsample < size):
    sampletmp = np.random.multivariate_normal(\
              mean=mean,\
              cov=covariance,\
              size=size)
    p_prior = fprior(1.0/sampletmp[:,0])
    p_prior = p_prior / np.max(p_prior) # Absolute value is not important
    # the probability that the sample is adopted is proportional to p_prior
    # p_prior takes between 0 and 1
    sampletmp_adopted = sampletmp[(np.random.rand(size) < p_prior)]
    if nsample == 0:
      samples = sampletmp_adopted.copy()
    else:
      samples = np.vstack([samples,sampletmp_adopted])
    nsample = len(samples)
  dist_out,pmra_out,pmdec_out = 1.0/samples[0:size,0],\
                                samples[0:size,1],samples[0:size,2]
  return dist_out,pmra_out,pmdec_out

def SamplePMraPMdec(pmra,pmdec,\
                      pmra_error,pmdec_error,\
                      pmra_pmdec_corr=0.0,\
                      size=10000):
  '''
    Sampling proper motions 

    Input
      input parameters assume gaia units
    Output
      pmra, pmdec
                  
  '''
  # Construct mean vector and covariance matrix
  sigma = np.array([pmra_error,pmdec_error])\
                  .repeat(2).reshape(2,2)
  corr_coeff = np.array(\
     [[1.0                , pmra_pmdec_corr     ],
      [pmra_pmdec_corr    , 1.0                 ]])
  covariance = corr_coeff * sigma * sigma.T
  mean  = np.array([pmra,pmdec])


  samples = np.random.multivariate_normal(\
              mean=mean,\
              cov=covariance,\
              size=size)
  pmra_out,pmdec_out = samples[0:size,0],samples[0:size,1]
  return pmra_out,pmdec_out

def get_Lcirc(potential,E):
  rcirc = potential.Rcirc(E=E)
  position = np.column_stack((rcirc, rcirc*0, rcirc*0)) 
  Lcirc = rcirc*np.sqrt(-rcirc*potential.force(position)[:,0])
  return Lcirc

def get_output_values(ra,dec,dist,pmra,pmdec,rv,\
                      gc_frame,units,potential,actf,unique_id=None,\
                      nTcirc_integ=10,np_per_orbit=1000,\
                      out_orbits_npyfile=None,\
                      maxTintegrate = 13.8,\
                      angles=True,integrate=True):
  '''
    Input
      ra, dec   : in degree
      dist      : kpc
      pmra,pmdec: in mas/year
      rv        : km/s
      gc_frame  : astropy.coordinates.Galactocentric object
      units     : dictionary that contains units for L, V, T, M
      potential : potential
      actf      : action finder
    (optional)
      unique_id : unique id
      nTcirc_integ: duration of the integration = Tcirc * nTcirc_integ
      np_per_orbit: total number of points to save orbits = 
                      np_per_orbit * nTcirc_integ
      out_orbits_npyfile: file to save orbits in npy format
                          uid,x,y,z,vx,vy,vz
      maxTintegrate: maximum integration time in Gyr
                     default 13.8 Gyr
    Output
      output_dict
        x, y, z, vx, vy, vz, vr, vperp, vphi, jr, jz, jphi, E, circ
        (if integrate is True) rmin, rmax, zmax, ecc, R2min, R2max
        (if angles is True) Or, Oz, Ophi
  ''' 
  nobj = len(np.atleast_1d(ra)) 
  if unique_id is None:
    unique_id = np.arange(0,nobj)

  avail_mem = psutil.virtual_memory().available
  mem_max = avail_mem*0.9
  ## Define large arrays to check if the available memory is sufficient
  out_dict = {'x':     np.array([np.nan]).repeat(nobj),
              'y':     np.array([np.nan]).repeat(nobj),
              'z':     np.array([np.nan]).repeat(nobj),
              'vx':    np.array([np.nan]).repeat(nobj),
              'vy':    np.array([np.nan]).repeat(nobj),
              'vz':    np.array([np.nan]).repeat(nobj),
              'vr':    np.array([np.nan]).repeat(nobj),
              'vperp': np.array([np.nan]).repeat(nobj),
              'vphi':  np.array([np.nan]).repeat(nobj),
              'jr':    np.array([np.nan]).repeat(nobj),
              'jz':    np.array([np.nan]).repeat(nobj),
              'jphi':  np.array([np.nan]).repeat(nobj),
              'E':     np.array([np.nan]).repeat(nobj),
              'circ':  np.array([np.nan]).repeat(nobj)}

  positions = np.array([np.nan]).repeat(6*nobj).reshape(nobj,6)

  if integrate:
    out_dict['rmin']   = np.array([np.nan]).repeat(nobj)
    out_dict['rmax']   = np.array([np.nan]).repeat(nobj)
    out_dict['zmax']   = np.array([np.nan]).repeat(nobj)
    out_dict['ecc']    = np.array([np.nan]).repeat(nobj)
    out_dict['R2min']  = np.array([np.nan]).repeat(nobj)
    out_dict['R2max']  = np.array([np.nan]).repeat(nobj)
#    orbits_integ = np.array([np.nan]).repeat(6*nobj*nTcirc_integ*np_per_orbit)
#    time_integ  = np.array([np.nan]).repeat(nobj*nTcirc_integ*np_per_orbit)
#    orbits_tmp  = np.array([np.nan]).repeat(int(7*nobj*nTcirc_integ*\
#                                                np_per_orbit*1.5))
    # 1.5 is a factor to ensure that the memory is sufficient
    assert mem_max > (8*nobj*6 + 8*nobj*nTcirc_integ*np_per_orbit*7*2),\
      MemoryError(f'Available{mem_max*1.0e-9:.2f} Required{8*nobj*nTcirc_integ*np_per_orbit*7*2*1.0e-9:.2f}')
#   del orbits_tmp # Not needed now 
#    del time_integ # Not needed now 
#    del orbits_integ # Not needed now 
    
  if angles:
    out_dict['Or']   = np.array([np.nan]).repeat(nobj)
    out_dict['Oz']   = np.array([np.nan]).repeat(nobj)
    out_dict['Ophi']   = np.array([np.nan]).repeat(nobj)
 
  # Get normalized cartesian coordinates 
  positions[:,0],positions[:,1],positions[:,2],\
    positions[:,3],positions[:,4],positions[:,5]=\
    Observed2NormGCFrame(ra,dec,dist,pmra,pmdec,rv,gc_frame,units)  

  # If integrate is true, calculate rmin,rmax,zmax,ecc,r2min,r2max
  if integrate:
    t1 = time.time()
    # orbit integration
    ntraj = nTcirc_integ*np_per_orbit

    # integration time
    integ_time = np.minimum(potential.Tcirc(positions)*nTcirc_integ,\
                            maxTintegrate/units['T'])
    integ_time = np.where(np.isnan(integ_time),maxTintegrate/units['T'],\
                          integ_time) # nan for unbound objects
    orbits_tmp = agama.orbit(ic = positions,\
                         potential = potential,\
                         time = integ_time,\
                         trajsize = ntraj)
 
    # orbit integration
    print('  {0:10.4f} sec for integration'.format(time.time()-t1))
    t1 = time.time()
    # get orbit and time sequence from the output
    orbits_integ = np.array([_tmp1[1] for _tmp1 in orbits_tmp])
    time_integ   = np.array([_tmp1[0] for _tmp1 in orbits_tmp])
   
    del orbits_tmp
    # save orbit to .npy file if filename is provided 
    if not out_orbits_npyfile is None:
      orbits_tmp = np.hstack([\
                   unique_id.repeat(ntraj).reshape(nobj*ntraj,1),\
                   time_integ.reshape(nobj*ntraj,1),\
                   orbits_integ.reshape(nobj*ntraj,6)])
      np.save(out_orbits_npyfile,orbits_tmp)
      del orbits_tmp
    
    # get and store rmin etc.
#   print(orbits_integ)
#   print(time_integ.shape)
    rr = np.sqrt(np.sum(orbits_integ[:,:,0:3]**2.0,axis=2))
    out_dict['rmin'] = np.min(rr,axis=1)
    out_dict['rmax'] = np.max(rr,axis=1)
    out_dict['zmax'] = np.max(np.abs(orbits_integ[:,:,2]),axis=1)
    out_dict['ecc'] = (out_dict['rmax'] - out_dict['rmin'])/\
                       (out_dict['rmax'] + out_dict['rmin'])
    rr = np.sqrt(np.sum(orbits_integ[:,:,0:2]**2.0,axis=2))
    out_dict['R2min'] = np.min(rr,axis=1)
    out_dict['R2max'] = np.max(rr,axis=1)
    del rr
  
    del orbits_integ
    del time_integ
    print('  {0:10.4f} sec for post processing integration'.format(time.time()-t1))

  t1 = time.time()
  # action calculation
  if angles:
    (out_dict['jr'],out_dict['jz'],out_dict['jphi']),(_,_,_),\
      (out_dict['Or'],out_dict['Oz'],out_dict['Ophi']) = \
      np.array([_arr.T for _arr in actf(positions,angles=True)])
  else:
    out_dict['jr'],out_dict['jz'],out_dict['jphi'] = actf(positions).T
  print('  {0:10.4f} sec for action calculation'.format(time.time()-t1))

  # energy
  out_dict['E'] = potential.potential(positions[:,0:3])+\
                  0.5*np.sum(positions[:,3:6]**2,axis=1)
  out_dict['circ'] = out_dict['jphi']/get_Lcirc(potential,out_dict['E'])

  # positions
  out_dict['x'],out_dict['y'],out_dict['z'],\
    out_dict['vx'],out_dict['vy'],out_dict['vz'] = positions.T
  # vr,vperp,vphi
  _,_,out_dict['vr'],_,out_dict['vphi'] = \
    cartesian_2_cylindrical(*positions.T)
  out_dict['vperp'] = np.sqrt(out_dict['vr']**2.0+out_dict['vz']**2.0)

  return out_dict


def get_errorestimatesMC(ra,dec,plxdist,pmra,pmdec,rv,nmc,\
                               gc_frame,units,potential,actf,unique_id=None,\
                               ra_error = None, dec_error = None,\
                               plxdist_error = None, pmra_error = None,\
                               pmdec_error = None, rv_error = None,\
                               plx_pmra_corr = None, plx_pmdec_corr = None,\
                               pmra_pmdec_corr = None,\
                               rscale = None,isplxdist='plx',\
                               nTcirc_integ=10,np_per_orbit=1000,\
                               maxTintegrate = 13.8,\
                               angles=True,integrate=True, \
                               outputdir_mcsample = None):
  '''
    Output (example)
      output_dict = {'jr': {'mean'  : mean,
                            'median': median,
                            'std'   : std, 
                            'q02'   :  2 percentile (2 sigma),
                            'q16'   : 16 percentile (1 sigma),
                            'q25'   : 25 percentile,
                            'q75'   : 75 percentile,
                            'q84'   : 84 percentile (1 sigma),
                            'q98'   : 98 percentile (2 sigma)},
                      'jz': {....} }
  '''                      
 
  avail_mem = psutil.virtual_memory().available
  nobj = len(np.atleast_1d(ra)) 
  if unique_id is None:
    unique_id = np.arange(0,nobj)

  # Set 0 for None
  def set0forNone(param,nlen):
    if param is None:
      return np.zeros(nlen)
    else:
      return param
  ra_error        = set0forNone(ra_error       ,nobj)
  dec_error       = set0forNone(dec_error      ,nobj)
  plxdist_error   = set0forNone(plxdist_error  ,nobj)
  pmra_error      = set0forNone(pmra_error     ,nobj)
  pmdec_error     = set0forNone(pmdec_error    ,nobj)
  rv_error        = set0forNone(rv_error       ,nobj)
  plx_pmra_corr   = set0forNone(plx_pmra_corr  ,nobj)
  plx_pmdec_corr  = set0forNone(plx_pmdec_corr ,nobj)
  pmra_pmdec_corr = set0forNone(pmra_pmdec_corr,nobj)
  if rscale is None:
    rscale = np.array([None]*nobj)

  # check if sufficient memory is available
  ra_sample   = np.array(ra).repeat(nmc)
  dec_sample  = np.array(dec).repeat(nmc)
  dist_sample = np.array(plxdist).repeat(nmc) 
  pmra_sample = np.array(pmra).repeat(nmc)
  pmdec_sample = np.array(pmdec).repeat(nmc)
  rv_sample = np.array(rv).repeat(nmc)
  results = np.zeros(nobj*nmc*30)  

  # Sample dist,pmra,pmdec depending on the sample 
  t1 = time.time()
  if isplxdist == 'plx':
    dist_sample,pmra_sample,pmdec_sample = \
      np.hstack([SamplePMraPMdecDist(*xx[:-1],rscale=xx[-1],size=nmc) for xx \
        in zip(plxdist,pmra,pmdec,plxdist_error,pmra_error,pmdec_error,\
               plx_pmra_corr,plx_pmdec_corr,pmra_pmdec_corr,rscale)])
  elif isplxdist == 'dist':
    pmra_sample,pmdec_sample = \
      np.hstack([SamplePMraPMdec(*xx,size=nmc) for xx \
        in zip(pmra,pmdec,pmra_error,pmdec_error,pmra_pmdec_corr)])
    dist_sample = plxdist.repeat(nmc) + \
                    plxdist_error.repeat(nmc)*np.random.randn(nmc*nobj)
  ra_sample  = ra.repeat(nmc) + \
                 ra_error.repeat(nmc)*np.random.randn(nmc*nobj)
  dec_sample = dec.repeat(nmc) + \
                 dec_error.repeat(nmc)*np.random.randn(nmc*nobj)
  rv_sample  = rv.repeat(nmc) + \
                 rv_error.repeat(nmc)*np.random.randn(nmc*nobj)
  print('  {0:10.4f} sec for producing samples'.format(time.time()-t1))

  isplit = 1 # number of steps (memory is limited) 
  isdone = False
  ntotal = nobj
  istart = 0 # the first index of the step
  istep = 1  # step counter
  output_dict = {} # store outputs
  # Get values
  while not isdone: 
    try:
      iendobj = np.minimum(istart+ntotal//isplit,ntotal)
      iend = iendobj*nmc
      resultstmp = get_output_values(
              ra_sample[istart*nmc:iend],\
              dec_sample[istart*nmc:iend],\
              dist_sample[istart*nmc:iend],\
              pmra_sample[istart*nmc:iend],\
              pmdec_sample[istart*nmc:iend],\
              rv_sample[istart*nmc:iend],\
              gc_frame,\
              units,\
              potential,\
              actf,\
              nTcirc_integ = nTcirc_integ,
              np_per_orbit = np_per_orbit,
              out_orbits_npyfile = None,\
              maxTintegrate = maxTintegrate,\
              angles= angles,\
              integrate = integrate)
      # Store results
      for key,value in resultstmp.items():
        _valuetmp = value.reshape(iendobj-istart,nmc)

        if outputdir_mcsample is not None:
          np.save(outputdir_mcsample + f'/{key}{istep:d}.npy', value)
        if istep == 1: # For the first iteration
          output_dict[key] = {'mean'   : np.nanmean(_valuetmp,axis=1),
                           'median' : np.nanmedian(_valuetmp,axis=1),
                           'std'    : np.nanstd(_valuetmp,axis=1),
                           'q02'    : np.nanpercentile(_valuetmp, 2.275,axis=1),
                           'q16'    : np.nanpercentile(_valuetmp,15.865,axis=1),
                           'q25'    : np.nanpercentile(_valuetmp,25.000,axis=1),
                           'q75'    : np.nanpercentile(_valuetmp,75.000,axis=1),
                           'q84'    : np.nanpercentile(_valuetmp,84.135,axis=1),
                           'q98'    : np.nanpercentile(_valuetmp,97.725,axis=1)}
        else:
          output_dict[key]['mean'] = \
            np.append(output_dict[key]['mean'],np.nanmean(_valuetmp,axis=1))
          output_dict[key]['median'] = \
            np.append(output_dict[key]['median'],np.nanmedian(_valuetmp,axis=1))
          output_dict[key]['std'] = \
            np.append(output_dict[key]['std'],\
            np.nanstd(_valuetmp,axis=1))
          output_dict[key]['q02'] = \
            np.append(output_dict[key]['q02'],\
            np.nanpercentile(_valuetmp, 2.275,axis=1))
          output_dict[key]['q16'] = \
            np.append(output_dict[key]['q16'],\
            np.nanpercentile(_valuetmp,15.865,axis=1))
          output_dict[key]['q25'] = \
            np.append(output_dict[key]['q25'],\
            np.nanpercentile(_valuetmp,25.000,axis=1))
          output_dict[key]['q75'] = \
            np.append(output_dict[key]['q75'],\
            np.nanpercentile(_valuetmp,75.000,axis=1))
          output_dict[key]['q84'] = \
            np.append(output_dict[key]['q84'],\
            np.nanpercentile(_valuetmp,84.135,axis=1))
          output_dict[key]['q98'] = \
            np.append(output_dict[key]['q98'],\
                      np.nanpercentile(_valuetmp,97.725,axis=1))
      istep = istep+1
      istart = iendobj
      if (istart == ntotal):
        isdone = True

    # Catch memory error
    except MemoryError:
      if adjust:
        print('Memory error has been detected')
        print('Step size increased {0:d} -> {1:d}'.format(isplit,isplit*2))
        isplit = isplit*2
      else: 
        raise MemoryError
  return output_dict     

class GetKinematicsAll:
   def __init__(self,inputfile,output_asobserved,output_mcsummary,\
                distance_source='plx', nmc=1000, getecc=True,\
                angles=True, solarposition=None,\
                plx_0pt = -0.029,\
                pot = None,_R0=1.0,_V0=1.0,_M0=1.0,\
                maxTintegrate = 13.8,\
                nTcirc_integ = 20,\
                np_per_orbit = 1000,\
                dist_scale = 1.0,
                output_fmt = '.3f',\
                outputdir_trajectory=None, outputdir_mcsample=None):
     # Store input parameters 
     self.infile = inputfile
     self.outasobserved = output_asobserved
     self.outmcsummary  = output_mcsummary
     if not distance_source in ['plx','BJ','dist']:
       raise ValueError('distance_source must be one of [plx,BJ,dist]')
     self.distance_source = distance_source
     self.plx_0pt = plx_0pt 
     self.nmc = int(nmc)
     if (outputdir_trajectory is not None) or getecc:
       self.integrate = True 
     else:
       self.integrate = False
     self.angles = angles
     self.maxTintegrate = maxTintegrate
     self.nTcirc_integ = nTcirc_integ
     self.np_per_orbit = np_per_orbit
     self.output_fmt   = output_fmt
     self.label = {'unique_id'     :'source_id'     ,
          'ra'      :'ra'      ,'dec'      :'dec'     ,'parallax':'parallax',
          'pmra'    :'pmra'    ,'pmdec'    :'pmdec'   ,'rv'      :'rv'     ,
          'parallax_error':'parallax_error','pmra_error'   :'pmra_error'    ,
          'pmdec_error'   :'pmdec_error'   ,'rv_error'     :'err_rv'     ,
          'parallax_pmra_corr'             :'parallax_pmra_corr'            ,
          'parallax_pmdec_corr'            :'parallax_pmdec_corr'           ,
          'pmra_pmdec_corr'                :'pmra_pmdec_corr'               ,
          'r_est'   :'r_est'  ,'r_len'     :'r_len',
          'dist'    :'dist'   ,'dist_error':'dist_error'}
     self.outputdir_trajectory = outputdir_trajectory
     self.outputdir_mcsample   = outputdir_mcsample

     # Drop unnecessary labels
     if self.distance_source == 'BJ': 
       droplabel = ['dist','dist_error']       
     elif self.distance_source == 'plx': 
       droplabel = ['dist','dist_error','r_est','r_len']       
     elif self.distance_source == 'dist': 
       self.dist_scale = dist_scale
       droplabel = ['r_est','r_len','parallax','parallax_error',\
                    'parallax_pmra_corr','parallax_pmdec_corr']
     [self.label.pop(dl) for dl in droplabel]

     # Set units
     _T0 = (_R0*u.kpc/(_V0*u.km/u.s)/u.Gyr).decompose()
     self.units = {'L': _R0,\
                   'V': _V0,\
                   'M': _M0,\
                   'T': _T0}
     agama.setUnits( mass=_M0, length=_R0, velocity=_V0)
     
     # Set coordinates using astropy
     if solarposition is None:
       solarposition = SolarPosition()
     v_sun = CartesianDifferential(np.array([\
                                     solarposition.vx,\
                                     solarposition.vy,\
                                     solarposition.vz\
                                             ])*u.km/u.s)
     self.gc_frame = Galactocentric(
       galcen_distance = np.abs(solarposition.x)*u.kpc,\
       galcen_v_sun=v_sun,\
       z_sun=solarposition.z*u.kpc)

     # Create ActionFinder and potential
     if pot is None:
       self.potential = agama.Potential(agama.__file__[:agama.__file__.rfind('/')]+'/data/McMillan17.ini')
     else:
       self.potential = pot
     print('Agama setup')
     self.actf= agama.ActionFinder(self.potential, interp=False) 

   def printunit(self):
     print('length   :{0:.3e} kpc'.format(self.units['L']))
     print('velocity :{0:.3e} km/s'.format(self.units['V']))
     print('Time     :{0:.3e} Gyr'.format(self.units['T']))
     print('Mass     :{0:.3e} Msun'.format(self.units['M']))

   def changeunit(self,_R0,_V0,_M0=1.0):
     _T0 = (_R0*u.kpc/(_V0*u.km/u.s)/u.Gyr).decompose()
     self.units['L'] = _R0
     self.units['V'] = _V0
     self.units['M'] = _M0
     self.units['T'] = _T0
     agama.setUnits( mass=_M0, length=_R0, velocity=_V0)

   def read_clean_data(self,fileformat='fits'):
     ## Read
     if fileformat == 'hdf5':
       import vaex
       data = vaex.open(self.infile).to_astropy_table()
     else:
       data = table.Table.read(self.infile,format=fileformat)
     ## Check if columns exist
     for l1,l2 in self.label.items():
       if not l2 in data.columns:
         print('{0:s} (as {1:s}) is not in the columns'.format(l1,l2))
         if l2 in \
             ['parallax_pmra_corr','parallax_pmdec_corr','pmra_pmdec_corr']:
           print('Correlation is set to zero')
           data[l2] = 0.0
     ## Correct for the zero point     
     if self.distance_source in ['BJ','plx']:
       data[self.label['parallax']] = data[self.label['parallax']] - \
                                      self.plx_0pt
     ## Filter the data
     for l1,l2 in self.label.items():
       if l1 == 'unique_id': 
         continue
       ndatain = len(data)
       data = data[np.isfinite(np.array(data[l2]))]
       if (ndatain != len(data)):
         print('{0:s} (as {1:s}) must be finite\n'.format(l1,l2)+\
             '{0:d} points are excluded'.format(ndatain-len(data)))
     # Specific to each distance source
     if self.distance_source == 'BJ':
       ndatain = len(data)
       poe = np.array(data[self.label['parallax']]/\
                      data[self.label['parallax_error']])
       # it takes too long to sample if parallax_over_error < -2
       data = data[poe > -2]
       if len(data) != ndatain:
         print('parallax must be finite and greater than -2*e_plx'+\
               ' to use BJ distance\n'+\
               '{0:d} points are excluded'.format(ndatain-len(data)))
       # In BJ, r_est and r_len are given in pc
       data[self.label['r_est']] = 1.0e-3*data[self.label['r_est']] 
       data[self.label['r_len']] = 1.0e-3*data[self.label['r_len']] 

     elif self.distance_source == 'plx':
       ndatain = len(data)
       poe = np.array(data[self.label['parallax']]/\
                      data[self.label['parallax_error']])
       data = data[poe > 0.0]
       # parallax needs to be positive
       if len(data) != ndatain:
         print('parallax must be finite and greater than zero'+\
               ' to use the inverse of parallax\n'+\
               '{0:d} points are excluded'.format(ndatain-len(data)))

     elif self.distance_source == 'dist':
       ndatain = len(data)
       data = data[np.array(data[self.label['dist']]) > 0.0]
       print(f'distance will be multiplied by {self.dist_scale}')
       data[self.label['dist']] = self.dist_scale*data[self.label['dist']]
       data[self.label['dist_error']] = self.dist_scale*data[self.label['dist_error']]
       # distance needs to be positive
       if len(data) != ndatain:
         print('distance must be finite and greater than zero'+\
               ' to use distance\n'+\
               '{0:d} points are excluded'.format(ndatain-len(data)))

     ## Store data
     self.inputdata = data

   def get_as_observed(self,isplit=1,adjust=True):

     ## Check if data are already read and stored
     if not hasattr(self,'inputdata'):
        raise AttributeError('Do read_clean_data first')
    
#     isplit = 1 # number of steps (memory is limited) 
     isdone = False
     ntotal = len(self.inputdata)
     istart = 0 # the first index of the step
     istep = 1  # step counter
     outputkeys = None

     # Start calculation
     while (not isdone):

       # if orbits are to be saved
       if self.outputdir_trajectory is None:
         outnpy = None
       else:
         outnpy = (self.outputdir_trajectory + '/{0:8d}.npy'.format(istep)).\
                  replace(' ','')

       try: 
         iend = np.minimum(istart+ntotal//isplit,ntotal) # the last index
         print('trying {0:8d} step {1:12d} -- {2:12d} ({3:12d})'.format(\
               istep,istart,iend,ntotal))
         ## data used for this step
         step_input = self.inputdata[istart : iend]

         ## get distance depending on the method
         if self.distance_source == 'plx':
           dist = 1.0/step_input[self.label['parallax']]
         elif self.distance_source == 'BJ':
           dist = step_input[self.label['r_est']]
         elif self.distance_source == 'dist':
           dist = step_input[self.label['dist']]
         
         ## get outputs for this step 
         results = get_output_values(
                     np.array(step_input[self.label['ra']]),\
                     np.array(step_input[self.label['dec']]),\
                     np.array(dist),\
                     np.array(step_input[self.label['pmra']]),\
                     np.array(step_input[self.label['pmdec']]),\
                     np.array(step_input[self.label['rv']]),\
                     self.gc_frame,\
                     self.units,\
                     self.potential,\
                     self.actf,\
                     unique_id=np.array(step_input[self.label['unique_id']]),\
                     nTcirc_integ = self.nTcirc_integ,
                     np_per_orbit = self.np_per_orbit,
                     out_orbits_npyfile = outnpy,\
                     maxTintegrate = self.maxTintegrate,\
                     angles= self.angles,\
                     integrate = self.integrate)
         if outputkeys is None: # For the first iteration. write header
           outputkeys = results.keys()
           fout = open(self.outasobserved,'w')
           fout.write('#unique_id ra dec')
           _ = [fout.write(' {0:s}'.format(key)) for key in outputkeys]
           fout.write('\n')
           
           # Format for the output
           output_line_fmt = '{:} {:12.6f} {:+12.6f}'+\
                             (' {:'+self.output_fmt+'}')*len(outputkeys)+\
                             '\n'
         else: # Just open for istep > 1
           fout = open(self.outasobserved,'a')
         t1 = time.time() 
         # Write results for this round
         for oneobj_out in zip(np.array(step_input[self.label['unique_id']]),\
                               np.array(step_input[self.label['ra']]),\
                               np.array(step_input[self.label['dec']]),\
                               *[results[key] for key in outputkeys]):
           fout.write(output_line_fmt.format(*oneobj_out))
         fout.close()
         print('{0:10.4f} sec for writing files'.format(time.time()-t1))

         # if orbits are saved, make a note to the header 
         if not outnpy is None:
           fnpy_head = open(self.outputdir_trajectory+'/head.dat','w')
           fnpy_head.write('{0:8d} {1:12d} {2:12d} {3:100s}\n'.format(\
                            istep,istart,iend,outnpy.split('/')[-1])) 
           fnpy_head.close()
         
         # Update counters
         istep = istep+1
         istart = iend
         if (istart == ntotal):
           isdone = True

       # Catch memory error
       except MemoryError:
         if adjust:
           print('Memory error has been detected')
           print('Step size increased {0:d} -> {1:d}'.format(isplit,isplit*2))
           isplit = isplit*2
         else: 
           raise MemoryError
     return  
         
   def get_MC_summary(self,isplit=1,adjust=True):

     ## Check if data are already read and stored
     if not hasattr(self,'inputdata'):
        raise AttributeError('Do read_clean_data first')
   
     nmc = self.nmc 
#     isplit = 1 # number of steps (memory is limited) 
     isdone = False
     ntotal = len(self.inputdata)
     istart = 0 # the first index of the step
     istep = 1  # step counter
     outputkeys = None

     # Start calculation
     while (not isdone):

       # if orbits are to be saved
#      if self.outputdir_mcsample is None:
#        outmc = None
#      else:
#        outmc = (self.outputdir_mcsample + '/{0:8d}.npy'.format(istep)).\
#                 replace(' ','')

       try: 
         iend = np.minimum(istart+ntotal//isplit,ntotal) # the last index
         print('trying {0:8d} step {1:12d} -- {2:12d} ({3:12d})'.format(\
               istep,istart,iend,ntotal))
         ## data used for this step
         step_input = self.inputdata[istart : iend]

         ## parallax/distance input
         if self.distance_source == 'plx':
           plxdist = step_input[self.label['parallax']]
           plxdist_error = step_input[self.label['parallax_error']]
           plx_pmra_corr = step_input[self.label['parallax_pmra_corr']]
           plx_pmdec_corr = step_input[self.label['parallax_pmdec_corr']]
           pmra_pmdec_corr = step_input[self.label['pmra_pmdec_corr']]
           rscale  = None
           isplxdist = 'plx'
         elif self.distance_source == 'BJ':
           plxdist = step_input[self.label['parallax']]
           plxdist_error = step_input[self.label['parallax_error']]
           plx_pmra_corr = step_input[self.label['parallax_pmra_corr']]
           plx_pmdec_corr = step_input[self.label['parallax_pmdec_corr']]
           pmra_pmdec_corr = step_input[self.label['pmra_pmdec_corr']]
           rscale  = step_input[self.label['r_len']]
           isplxdist = 'plx'
         elif self.distance_source == 'dist':
           plxdist = step_input[self.label['dist']]
           plxdist_error = step_input[self.label['dist_error']]
           plx_pmra_corr = None
           plx_pmdec_corr = None
           pmra_pmdec_corr = step_input[self.label['pmra_pmdec_corr']]
           #pmra_pmdec_corr = None
           rscale  = None
           isplxdist = 'dist'
         
         ## get outputs for this step 
         results = get_errorestimatesMC(
           np.array(step_input[self.label['ra']]),\
           np.array(step_input[self.label['dec']]),\
           np.array(plxdist),\
           np.array(step_input[self.label['pmra']]),\
           np.array(step_input[self.label['pmdec']]),\
           np.array(step_input[self.label['rv']]),\
           self.nmc,\
           self.gc_frame,\
           self.units,\
           self.potential,\
           self.actf,\
           unique_id=np.array(step_input[self.label['unique_id']]),\
           ra_error = None,#np.array(step_input[self.label['ra_error']]),\
           dec_error = None,#np.array(step_input[self.label['dec_error']]),\
           plxdist_error = np.array(plxdist_error),\
           pmra_error = np.array(step_input[self.label['pmra_error']]),\
           pmdec_error = np.array(step_input[self.label['pmdec_error']]),\
           rv_error = np.array(step_input[self.label['rv_error']]),\
           plx_pmra_corr = plx_pmra_corr,\
           plx_pmdec_corr = plx_pmdec_corr,\
           pmra_pmdec_corr = pmra_pmdec_corr,\
           rscale = rscale,isplxdist= isplxdist,\
           nTcirc_integ = self.nTcirc_integ,
           np_per_orbit = self.np_per_orbit,
           maxTintegrate = self.maxTintegrate,\
           angles= self.angles,\
           integrate = self.integrate,
           outputdir_mcsample = self.outputdir_mcsample)

         if outputkeys is None: # For the first iteration. write header
           outputkeys = results.keys()
           fout = open(self.outmcsummary,'w')
           fout.write('#unique_id ra dec')
           _ = [fout.write(' {0:s} {0:s}_std {0:s}_med '.format(key) +\
                           ' {0:s}_02 {0:s}_16 {0:s}_25 '.format(key) +\
                           ' {0:s}_75 {0:s}_84 {0:s}_98 '.format(key) \
                           ) for key in outputkeys]
           statslabel = ['mean','std','median','q02',\
                         'q16','q25','q75','q84','q98']
           fout.write('\n')
           
           # Format for the output
           output_line_fmt = '{:} {:12.6f} {:+12.6f}'+\
                             (' {:'+self.output_fmt+'}')*9*len(outputkeys)+\
                             '\n'
         else: # Just open for istep > 1
           fout = open(self.outmcsummary,'a')
      
         # Write results for this round
         for oneobj_out in zip(np.array(step_input[self.label['unique_id']]),\
                               np.array(step_input[self.label['ra']]),\
                               np.array(step_input[self.label['dec']]),\
                               *[results[key][sl] for key in outputkeys \
                                                  for sl in statslabel]):
           fout.write(output_line_fmt.format(*oneobj_out))
         fout.close()
         
         # Update counters
         istep = istep+1
         istart = iend
         if (istart == ntotal):
           isdone = True

       # Catch memory error
       except MemoryError:
         if adjust:
           print('Memory error has been detected')
           print('Step size increased {0:d} -> {1:d}'.format(isplit,isplit*2))
           isplit = isplit*2
         else:
           raise MemoryError
