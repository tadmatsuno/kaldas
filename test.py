# Test code
import use_agama as u_agama
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('Agg')
def testSolarPosition():
  print('#### Testing Solar Position class ####')
  print('Default solar position')
  sunp = u_agama.SolarPosition()
  sunp.printsolarpos()

  print('\nChange x to McMillan et al. 2017')
  sunp.x = -8.2
  sunp.reference['x'] = 'McMillan et al. 2017'
  sunp.printsolarpos()

  print('#### Done ####\n\n')

def testGetKinematicsAll():

  print('#### Test plx input ####')
  gka1 = u_agama.GetKinematicsAll('test/testdata.fits','test/test_asobserved.dat',\
                       'test/test_MCout.dat',
                       distance_source='plx',nmc=1000, getecc=True,\
                       angles=True, solarposition=None,\
                       plx_0pt = -0.029,\
                       pot = None,_R0=1.0,_V0=1.0,_M0=1.0,\
                       outputdir_trajectory='test_orbits/',\
                       outputdir_mcsample=None)
  gka1.label['rv'] = 'radial_velocity'  
  gka1.label['rv_error'] = 'radial_velocity_error'  
  gka1.printunit()
  gka1.read_clean_data()
  gka1.get_as_observed()
  gka1.nmc = 100
  gka1.get_MC_summary()
  print('#### Done ####\n\n')

  print('#### Test sampling for plx input ####')
  u_agama.SamplePMraPMdecDist(\
    gka1.inputdata[0][gka1.label['parallax']],\
    gka1.inputdata[0][gka1.label['pmra']],\
    gka1.inputdata[0][gka1.label['pmdec']],\
    gka1.inputdata[0][gka1.label['parallax_error']],\
    gka1.inputdata[0][gka1.label['pmra_error']],\
    gka1.inputdata[0][gka1.label['pmdec_error']],\
    gka1.inputdata[0][gka1.label['parallax_pmra_corr']],\
    gka1.inputdata[0][gka1.label['parallax_pmdec_corr']],\
    gka1.inputdata[0][gka1.label['pmra_pmdec_corr']])
  print('#### Done ####\n\n')

  print('#### Test dist input ####')
  gka1 = u_agama.GetKinematicsAll('test/testdata.fits','test/test_asobserved.dat',\
                       'test/test_MCout.dat',
                       distance_source='dist',nmc=1000, getecc=True,\
                       angles=True, solarposition=None,\
                       plx_0pt = -0.029,\
                       pot = None,_R0=1.0,_V0=1.0,_M0=1.0,\
                       outputdir_trajectory=None, outputdir_mcsample=None)
  gka1.label['rv'] = 'radial_velocity'  
  gka1.label['rv_error'] = 'radial_velocity_error'  
  gka1.label['dist_error'] = 'dist_err'
  gka1.printunit()
  gka1.read_clean_data()
  gka1.get_as_observed()
  print('#### Done ####\n\n')

  print('#### Test BJ input ####')
  gka2 = u_agama.GetKinematicsAll('test/testdata.fits','test/test_asobserved.dat',\
                       'test/test_MCout.dat',
                       distance_source='BJ',nmc=1000, getecc=True,\
                       angles=True, solarposition=None,\
                       plx_0pt = -0.029,\
                       pot = None,_R0=1.0,_V0=1.0,_M0=1.0,\
                       outputdir_trajectory=None, outputdir_mcsample=None)
  gka2.label['rv'] = 'radial_velocity'  
  gka2.label['rv_error'] = 'radial_velocity_error'  
  gka2.read_clean_data()
  gka1.get_as_observed()
  print('#### Done ####\n\n')

  print('#### Test sampling for BJ input ####')
  u_agama.SamplePMraPMdecDist(\
    gka2.inputdata[0][gka2.label['parallax']],\
    gka2.inputdata[0][gka2.label['pmra']],\
    gka2.inputdata[0][gka2.label['pmdec']],\
    gka2.inputdata[0][gka2.label['parallax_error']],\
    gka2.inputdata[0][gka2.label['pmra_error']],\
    gka2.inputdata[0][gka2.label['pmdec_error']],\
    gka2.inputdata[0][gka2.label['parallax_pmra_corr']],\
    gka2.inputdata[0][gka2.label['parallax_pmdec_corr']],\
    gka2.inputdata[0][gka2.label['pmra_pmdec_corr']],\
    rscale=gka2.inputdata[0][gka2.label['r_len']])
  print('#### Done ####\n\n')

  print('#### Test coordinates transform ####')
  sunp = u_agama.SolarPosition()
  x,y,z,vx,vy,vz = u_agama.Observed2NormGCFrame(13.0,52.0,0.0,13.0,-132.0,0.0,\
                                            gka2.gc_frame,gka2.units)
  print('x: {0:.3f} (true {1:.3f})'.format(x,sunp.x))
  print('y: {0:.3f} (true {1:.3f})'.format(y,sunp.y))
  print('z: {0:.3f} (true {1:.3f})'.format(z,sunp.z))
  print('vx: {0:.3f} (true {1:.3f})'.format(vx,sunp.vx))
  print('vy: {0:.3f} (true {1:.3f})'.format(vy,sunp.vy))
  print('vz: {0:.3f} (true {1:.3f})'.format(vz,sunp.vz))
  print('#### Done ####\n\n')

  print('#### Test calculations ####')
  print('compare these with solar values')
  sun = [13.0,52.0,0.0,13.0,-132.0,0.0]
  g6412 = [205.0103750,-0.0385417,0.266,-228.840,-80.070,442.51]
  outdict= u_agama.get_output_values(*np.array([sun,g6412]).T,\
    gka2.gc_frame,gka2.units,gka2.potential,gka2.actf,unique_id=None,\
    nTcirc_integ=20,np_per_orbit=1000,\
    out_orbits_npyfile='test/test.npy',\
    angles=True,integrate=True)
  for key,value in outdict.items() :
    print('{0:s} {1:.3f}'.format(key,value[0]))
  print('These are for G64-12')
  for key,value in outdict.items() :
    print('{0:s} {1:.3f}'.format(key,value[1]))

  testorbit = np.load('test/test.npy')
  def plotorbit(orbits,figname):
    fig,axs = plt.subplots(1,2,figsize=(10,5))
    axs = np.atleast_1d(axs).ravel()
    ax = axs[0]
    cm = ax.scatter(orbits[:,2],orbits[:,3],c=orbits[:,1],s=0.2)
    fig.colorbar(cm,ax=ax,label='Time')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
  
    ax = axs[1]
    ax.scatter(np.sqrt(orbits[:,2]**2.0+orbits[:,3]**2.0),\
                    orbits[:,4],c=orbits[:,1],s=0.2)
    ax.set_xlabel('R')
    ax.set_ylabel('Z')
    fig.tight_layout()
    fig.savefig(figname)
  plotorbit(testorbit[testorbit[:,0]==0.0],'test/testorbit_Sun.png')
  plotorbit(testorbit[testorbit[:,0]==1.0],'test/testorbit_G6412.png')
  print('#### Done ####\n\n')


if __name__ == '__main__':
  testSolarPosition()
  testGetKinematicsAll()
