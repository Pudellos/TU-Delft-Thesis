import pydicom
import matplotlib.pyplot as plt
import numpy as np
import sympy as smp
from alpha_blending import linear_regression_dualE
from alpha_blending import linear_regression_multiE
import cv2
from scipy import odr


def alpha_blending_dualE(fps,inserts_densities,inserts_densities_err):
    """
    Return parameter alpha for Möhler et al. (2017) method for DECT
    
    Parameters
    ----------
    fps : ndarray 
        array of file paths/names, shape: (2,), dtype: str
        
    inserts_densities : ndarray
        array of float phantom inserts densities
    
    inserts_densities_err : ndarray
        array of float phantom inserts densities uncertainties
        
    Returns
    -------
    tmp : ndarray
        tmp is an output of linear_regression_dualE() function from alpha_blending.py script
        tmp[0] are calculated values of alpha blending parameter
        tmp[1] are associated errors 
    """
    d=DICOM_Images_calibration(fps)
    sigms=np.array(d.sigmas(inserts_densities,inserts_densities_err))
    coords=mapa(fps)
    sigma_vals,sigma_vals_err=[],[]
    for k in sigms:
        for i in coords:
            for j in i:
                sigma_vals.append(k[0][j[0],j[1]])
                sigma_vals_err.append(k[1][j[0],j[1]])  
    
    sigma_vals=np.array(sigma_vals)
    sigma_vals=np.reshape(sigma_vals,(len(fps),int(len(sigma_vals)/len(fps))))
    sigma_vals_err=np.array(sigma_vals_err)
    sigma_vals_err=np.reshape(sigma_vals_err,(len(fps),int(len(sigma_vals_err)/len(fps))))

    tmp=linear_regression_dualE(sigma_vals[0],sigma_vals[1], sigma_vals_err[0], sigma_vals_err[1])
    return(tmp[0],tmp[1])

def alpha_blending_multiE(fps,inserts_densities,inserts_densities_err):
    """
    Return parameters alpha for Möhler et al. (2017) method for PCCT
    
    Parameters
    ----------
    fps : ndarray 
        array of file paths/names, shape: (2,), dtype: str
        
    inserts_densities : ndarray (1d)
        array of float phantom inserts densities
    
    inserts_densities_err : ndarray (1d)
        array of float phantom inserts densities uncertainties
        
    Returns
    -------
    tmp : ndarray
        tmp is an output of linear_regression_multiE() function from alpha_blending.py script
        tmp[0] are calculated values of alpha blending parameter
        tmp[1] are associated errors 
    """
    d=DICOM_Images_calibration(fps)
    sigms=np.array(d.sigmas(inserts_densities,inserts_densities_err))
    coords=mapa(fps)
    sigma_vals,sigma_vals_err=[],[]
    for k in sigms:
        for i in coords:
            for j in i:
                sigma_vals.append(k[0][j[0],j[1]])
                sigma_vals_err.append(k[1][j[0],j[1]])  
    
    sigma_vals=np.array(sigma_vals)
    sigma_vals=np.reshape(sigma_vals,(len(fps),int(len(sigma_vals)/len(fps))))
    sigma_vals_err=np.array(sigma_vals_err)
    sigma_vals_err=np.reshape(sigma_vals_err,(len(fps),int(len(sigma_vals_err)/len(fps))))
    params=linear_regression_multiE(sigma_vals,sigma_vals_err)
    return(params[0],params[1])


def linear_regression_polynomial(x,y,x_err,y_err,deg,plot=False):
    """
    Orthogonal distance regression method for fitting a polynomial to data
    
    Parameters
    ----------
    x : ndarray (1d)
        x-axis values
    y : ndarray (1d)
        y-axis values
    x_err : ndarray (1d)
        errors associated with x
    y_err : ndarray (1d)
        errors associated with y
    deg : int
        degree of the polynomial
    plot : bool
        if True a plot of x vs y and fitted polynomial is shown
        
    Returns
    -------
    out.beta : ndarray
        coefficients of the fitted polynomial
    out.sd_beta : ndarray
        uncertainties associated with the values of the coefficients of the fitted polynomial
    """
    poly_model = odr.polynomial(deg)
    data = odr.RealData(x, y, sx=x_err, sy=y_err)
    odr_obj = odr.ODR(data, poly_model)
    out = odr_obj.run()
    if plot:
        fig, ax = plt.subplots()
        poly = np.poly1d(out.beta[::-1])
        poly_y = poly(x)
        plt.plot(x,y,'bo',markersize=4)
        ax.errorbar(x, y, xerr=x_err, yerr=y_err,color='black', linestyle='None', marker='x',label='input data')
        ax.plot(x, poly_y,'ro',label="polynomial ODR")
#         out.pprint()
    return(out.beta,out.sd_beta)

# def propagation_of_errors(expression,i,j,i_err,j_err):
#     '''expression must be in x,y terms
#     i,j,i_err,j_err = arrays'''
#     x,y=smp.symbols('x y')
#     dx=[float(smp.diff(expression,x).subs([(x,k),(y,l)])) for k,l in zip(i,j)]
#     dy=[float(smp.diff(expression,y).subs([(x,k),(y,l)])) for k,l in zip(i,j)] 
#     return(np.sqrt((dx*i_err)**2+(dy*j_err)**2))


def rho_rel(rho_array):
    """
    Converts electron densitites to relative to water electron densities
    
    Parameters
    ----------
    rho_array : ndarray
        values of electron densities
        
    Returns
    -------
    rho_array : ndarray
        values of electron densities relative to water
    """
    return(rho_array/3.34e23)

def rho_rel_err(rho_array):
    """
    Converts errors in electron densitites to errors relative to water electron density
    
    Parameters
    ----------
    rho_array : ndarray
        values of uncertainties in electron densities
        
    Returns
    -------
    rho_array : ndarray
        values of uncertainties in electron densities relative to water
    """
    return(rho_array/1e5)



################### ASSUMPTION: error on water density and other phantom densities in order O(1e10)



def electron_density_map(fn,inserts_rho=np.empty(0),error_map=False,plot=False):
    """
    Creates ndarray (image data) for an electron density map of the phantom, values of electron densities of inserts can be assigned.
    
    Parameters
    ----------
    inserts_rho : ndarray
        An array of electron densities of the pahtnom inserts in configuration:
                                            1
                                        2       3
                                        4   5   6
                                    7   8   9   10   11
                                       12   13  14
                                       15       16
                                            17
       (if None returns empty map of the phantom with all inserts being water)
                                            
    error_map : bool
        if True inserts_rho should be an array of electron densities uncertainties of the phantom inserts
    plot : bool
        if True an image of electron density map is shown
        
    Returns
    -------
    pixels : ndarray (image data)
        image data of electron density map of the phantom
    """
    
    #read image dimensions:
    pixels=np.full(np.shape(pydicom.dcmread(fn).pixel_array),10.0)
    
    #specify where inserts are:
    centres_inserts=[(int(512/2-1), int(58)),(int(117), int(117)),(int(512-117), int(115)),(int(181), int(179)),\
                     (int(512/2), int(150+2)),(int(512-183), int(179)),(int(57), int(512/2)),(int(152), int(512/2)),\
                     (int(512/2), int(512/2-2)),(int(358),int(512/2-2)),(int(453), int(512/2-2)),(int(184), int(512-184)),\
                     (int(512/2+3), int(360-1)),(int(512-182), int(512-184)),(int(117), int(512-117)),\
                     (int(512-115), int(512-117)),(int(512/2), int(440))]
    radii=[25]*4+[9]+[25]*3+[20]+[25]*8
    
    #specify electron densities:
    air=0.00001
    water=1.0
    inserts_init=1.0
    boundary=0.00001
    
    #create instert:
    for i in range(len(centres_inserts)):
        for x in range(int(centres_inserts[i][0] - radii[i]), int(centres_inserts[i][0] + radii[i])):
            for y in range(centres_inserts[i][1]- radii[i], centres_inserts[i][1] + radii[i]):
                if (x - centres_inserts[i][0])**2 + (y-centres_inserts[i][1])**2 <=radii[i]**2:pixels[y][x]=inserts_init
                
    centres_elipse=[((int(512/2+2)),(int(512/2)-7)),((int(512/2)),(int(512/2)+15))]
    axesLengths = [(285, 232),(278, 234)]
    angle_Start=[180,0]
    angle_End=[360,180]
    
#     create air:
    empty=[]
    for i in range(len(centres_elipse)):
        pixels = cv2.ellipse(pixels, centres_elipse[i], axesLengths[i],
               0, angle_Start[i], angle_End[i], (boundary,boundary,boundary), 1)   
    for row in range(len(pixels)):
        try:
            start=np.where(pixels[row]==boundary)[0][0]
            end=np.where(pixels[row]==boundary)[0][-1]
        except:
            empty.append(row)
            start=0
            end=-1
        if pixels[row][0]>=boundary:
            pixels[row][0:start]=air
            if row<=143 or row>=365:pixels[row][end:]=air    
    for i in range(len(pixels[0])):
            for j in empty:
                if j<=100 or j>=500:pixels[j][i]=air
            if i>=477:pixels[i]=air
                
    #assign inserts values
    if np.any(inserts_rho):
        if not error_map: 
            inserts_rho=rho_rel(inserts_rho)
            
        if error_map: 
            inserts_rho=rho_rel_err(inserts_rho)
        for i in range(len(centres_inserts)):
            for x in range(int(centres_inserts[i][0] - radii[i]), int(centres_inserts[i][0] + radii[i])):
                for y in range(centres_inserts[i][1]- radii[i], centres_inserts[i][1] + radii[i]):
                    if (x - centres_inserts[i][0])**2 + (y-centres_inserts[i][1])**2 <=radii[i]**2 and not error_map :pixels[y][x]=inserts_rho[i]
                    elif (x - centres_inserts[i][0])**2 + (y-centres_inserts[i][1])**2 <=radii[i]**2 and error_map :pixels[y][x]=inserts_rho[i]
    
#     #plot
    if plot:
        plt.figure(figsize=(10,10))
        plt.imshow(pixels,cmap=plt.cm.bone) 
    return(pixels)




def mapa(fn,inserts=np.empty(0),error_map=False,plot=True,return_array=True):
    """
    Return coordinates of all pixels corresponding to the phantom insters
    
    Parameters
    ----------
    fn : str
        file path of phantom image
    inserts : np.array
    
    """
    
    
    #specify where inserts are:
    centres_inserts=[(int(512/2-1), int(58)),(int(117), int(117)),(int(512-117), int(115)),(int(181), int(179)),\
                     (int(512/2), int(150+2)),(int(512-183), int(179)),(int(57), int(512/2)),(int(152), int(512/2)),\
                     (int(512/2), int(512/2-2)),(int(358),int(512/2-2)),(int(453), int(512/2-2)),(int(184), int(512-184)),\
                     (int(512/2+3), int(360-1)),(int(512-182), int(512-184)),(int(117), int(512-117)),\
                     (int(512-115), int(512-117)),(int(512/2), int(440))]
    radii=[25]*4+[9]+[25]*3+[20]+[25]*8
    coords=[[0,0]]*len(centres_inserts)
    for i in range(len(centres_inserts)):
        for x in range(int(centres_inserts[i][0] - radii[i]), int(centres_inserts[i][0] + radii[i])):
            for y in range(centres_inserts[i][1]- radii[i], centres_inserts[i][1] + radii[i]):
                if (x - centres_inserts[i][0])**2 + (y-centres_inserts[i][1])**2 <=radii[i]**2:
                    coords[i]=coords[i]+[[y,x]]
        coords[i]=coords[i][2:]
    return(coords)

def sigma_effective(sigmas,sigmas_err,p):
    """ 
    Calculates effective photon absorption cross sections per electron (sigmas) from provided separate SECT images
    
    Parameters
    ----------
    sigmas : ndarray (shape: (number of images to be combined, image_width, image_height))
        arrays of pixels of values equal to sigmas of each image
    sigmas_err : ndarray (shape: (number of images to be combined, image_width, image_height))
        arrays of pixels of values equal to uncertainties in sigmas of each image
    p : float or ndarray
        mixing parameter for deriving effective sigmas (ratios to use to mix values from different images)
        p is a float for DECT and ndarray for PCCT
        
    Returns
    -------
    out : ndarray (shape: (image_width, image_height)
        array (image data) of effective sigma values for the combined images
    out_err : ndarray (shape: (image_width, image_height)
        array (image data) of uncertainties in effective sigma values for the combined images
        
    """
    out=np.zeros(np.shape(sigmas[0]))
    out_err=np.zeros(np.shape(sigmas_err[0]))
    if type(p)==float:
        p=(p,1-p)
    for i in range(len(sigmas)):
        for j in range(len(sigmas[i])):
            for k in range(len(sigmas[i])):
                out[j][k]+=sigmas[i][j][k]*p[i]
                out_err[j][k]+=(sigmas_err[i][j][k]*p[i])**2
    return(out,np.sqrt(out_err))


def L_rel(beta,I,I_err):
    """
    Calculates relative to water stopping number (L) from Bethe formula
    
    Parameters
    ----------
    beta : float
        v/c for particle
    I : ndarray
        ionisation energies (eV)
    I_err : ndarray
        uncertainties associated with I (eV)
    
    Returns
    -------
    L/L_w : ndarray
        relative to water stopping number L
    L_err : ndarray
        relative to water uncertainty in stopping number L
    """
    m_ec2=0.511*10**6 #eV
    I_w=78 #eV (ICRU 2014 - Mohler Exp verifications)
    L_w=np.log((2*m_ec2*beta**2)/(1-beta**2))-beta**2-np.log(I_w)
    L=np.log((2*m_ec2*beta**2)/(1-beta**2))-beta**2-np.log(I)
    L_err=np.sqrt((((-1/I)/L_w)*I_err)**2)
    return(L/L_w,L_err)



class DICOM_Image():
    """ 
    Class for deriving quantities from a dicom image.
    
    Parameters
    ----------
    file_path : str
        CT image path/name
    """
        
    def __init__(self, file_path):
        self.fn = file_path
        self.ds = pydicom.dcmread(self.fn)
        self.pixels = self.ds.pixel_array*int(self.ds.RescaleSlope)+int(self.ds.RescaleIntercept)

    
    def sigma(self,inserts_rho_e,inserts_rho_e_err):
        """ 
        Retunrs photon absorption cross sections per electron (sigmas) of the calibrayion CT image (image of the phantom)
        
        Parameters
        ----------
        inserts_rho_e : ndarray
            array of electron densities of phantom inserts
        inserts_rho_e_err : ndarray
            array of uncertainties in electron densities of phantom inserts    
        
        Returns
        -------
        out : ndarray
            array of sigma values of pixels in the CT image
        err : ndarray
            array of uncertainties in sigma values of pixels in the CT image
        """
        #rhos are already made relative
        CT_numbers=self.pixels
        CT_number_err=self.CT_noise()
        if not np.any(inserts_rho_e_err):inserts_rho_e_err=np.zeros(len(inserts_rho_e))
        rho_e=self.rho_e_map(inserts_rho_e)
        rho_e_err=self.rho_e_map(inserts_rho_e_err,True)
        err=((1/(1000*rho_e))*(CT_number_err))**2+(((CT_numbers/1000+1)/rho_e**2)*(rho_e_err))**2
        out=(CT_numbers/1000+1)/rho_e
        return(out,np.sqrt(err))
    
    
    def profile(self,y,plot=True):
        """
        Plots a cross sectional view of the CT numbers of the image
        
        Parameters
        ----------
        y : int
            index of the row in the image for which the cross sectional view of CT numbers is to be plotted
        plot : bool
            if True a plot is shown
            
        Returns
        -------
        None
        """
        if plot==True:
            fig, (ax1,ax2) = plt.subplots(1,2,figsize=(18,6))
            fig.tight_layout()
            ax1.hlines(y, 0, len(self.pixels[0]), colors='green')
            ax1.set_xlabel('pixels')
            ax1.set_ylabel('pixels')
            ax1.imshow(self.pixels, cmap=plt.cm.bone) 
            ax2.plot(range(len(self.pixels)),self.pixels[y],'green')
            ax2.hlines(0, 0, len(self.pixels), colors='blue',label='water')
            ax2.set_xlabel('pixels')
            ax2.set_ylabel('pixel_value (HU)')
            plt.legend()
            plt.show()
        return()
    
    def CT_noise_per_line(self,line_index,plot=False,cut_off=False):
        """
        Calculates noise (standard deviation (std) of CT numbers) per row in the CT image
        
        Parameters
        ----------
        line_index : int
            index of the row for which the noise is to be caluclated
        plot: bool
            if True two plots are shown. One plot of CT numbers before correction and one plot of CT numebrs after noise correction. 
            Noise correction is done to exclude outliers associated with air around the phantom (outliers = values above 3*std of CT numbers ).
        cut_off : bool
            Manually setting threshold of outlier correction (HU of image pixels)
            
        """
        pixel_array=np.copy(self.pixels[line_index])
        if plot:
            fig, (ax1,ax2) = plt.subplots(1,2,figsize=(18,6))
            ax1.plot(range(len(pixel_array)),pixel_array)
            ax1.title.set_text('Before correction')
            ax1.set_ylabel('pixel_value (HU)')
            ax1.set_xlabel('pixels')   
            ax1.hlines(np.mean(pixel_array), 0, len(pixel_array), colors='black',label='water')
        if cut_off : pixel_array=np.delete(pixel_array,np.where(abs(pixel_array)>=cut_off))
        else : pixel_array=np.delete(pixel_array,np.where(abs(pixel_array)>=abs(np.mean(pixel_array))+3*abs(np.std(pixel_array))))
        if plot:
            ax2.plot(range(len(pixel_array)),pixel_array)
            ax2.title.set_text('After correction')
            ax2.set_xlabel('pixels')
            ax2.set_ylabel('pixel_value (HU)')
            ax2.hlines(np.mean(pixel_array), 0, len(pixel_array), colors='black',label='water')
            plt.show()
#         return(np.sqrt(np.std(pixel_array)))#here delete sqrt to hold
        return(np.std(pixel_array))#here delete sqrt to hold

    def CT_noise(self,data_points=10):
        """
        Calculates CT noise in the image based on deviations of measured HU of water in the phantom
        
        Parameters
        ----------
        data_points : int
            number of data points (water rows in the image) used for noise calculation after outliers are excluded from the set (values above 3 std of HU).
            Air is excluded from the calculation. Only rows in phantom image that are completly filled with water values are used to calculate CT noise.
            
        Returns
        -------
        average_noise : ndarray
            array (image data) full of average noise value of the image (assumption: noise is uniform across the entire image and so is the uncertainty associated with measured HU of the image)
        """
        average_noise=0 
        try:
            pixel_array=np.copy(self.pixels)
            stds=np.zeros(len(pixel_array))
            for i in range(len(pixel_array)):
                stds[i]=np.std(pixel_array[i])
            #find where air is and exclude it from noise calculation (only look inside phantom, not edges of image)
            start=np.where(stds>=3*np.std(pixel_array[0]))[0][0]
            end=np.where(stds>=3*np.std(pixel_array[0]))[0][-1]
    
            noise=np.zeros(len(pixel_array))
            for i in range(len(pixel_array)):
                noise[i]=self.CT_noise_per_line(i)
            noise=noise[start:end+1]
            tmp=np.zeros(data_points)
            for i in range(data_points):
                tmp[i]=noise[np.where(noise==min(noise))]
                noise=np.delete(noise,np.where(noise==min(noise)))
            average_noise=np.mean(tmp)
        except ValueError:
            print('try less data points for noise acquisition')
        return(np.full(np.shape(self.pixels),average_noise))

    def rho_e_map(self,inserts_densities,error_map=False,plot=False):
        """
        Deriving map of electron densities of the phantom image
        
        Parameters
        ----------
        inserts_densities : ndarray (1d)
            array of values of insert densities/uncertainties in inser densities (see help(electron_density_map()) for more details)
        error_map : bool
            if True inserts_densities should be an array of uncertainties in electron densities of phatom inserts, outputs map of uncertainties in electron densities of phatom inserts
            
        Returns
        -------
        ndarray (image data) of a map of electron densities/ uncertainties in electron densities of the phantom
        """
        return(electron_density_map(self.fn,inserts_densities,error_map,plot))
    
#     def L_map(self,L_values_inserts,error_map=False,plot=False):
#         return(L_map_function(self.fn,L_values_inserts,error_map,plot))


class DICOM_Images_calibration():
    """
    Class for deriving quantities from multiple CT images (eg. high, low energy scans in DECT or multiple images in PCCT).
    
    Parameters
    ----------
    file_paths : ndarray
        array of file names/paths of CT images used for calibration procedure
    """
    def __init__(self, file_paths):
        self.fns = file_paths
        self.dss = [pydicom.dcmread(i) for i in self.fns]
        self.pixels = [i.pixel_array*int(i.RescaleSlope)+int(i.RescaleIntercept) for i in self.dss]
        self.dcmi_objects=[DICOM_Image(i) for i in self.fns]
        
        
    def sigma_eff_func(self,inserts_densities,inserts_densities_err,params):
        """
        Returns array (image data) of effective photon absorption cross sections per electron (sigmas) of the CT images based on their CT numbers and electron densities of the phantom.
        
        Parameters
        ----------
        inserts_densities : ndarray (1d)
            array of electron densities of phantom inserts
        inserts_densities_err : ndarray (1d)
            array of uncertainties in electron densities of phantom inserts
        params : float or ndarray
            float for DECT, ndarray for PCCT
            mixing paramater for deriving effective sigma values from multiple images
            
        Returns
        -------
        sigma_eff : ndarray
            array (image data) of effective sigmas derived from the provided images
        sigma_eff_err : ndarray
            array (image data) of uncertainties in effective sigmas derived from the provided images    
        """
        rhos=[d.rho_e_map(inserts_densities,False) for d in self.dcmi_objects]
        rhos_errs=[d.rho_e_map(inserts_densities_err,True) for d in self.dcmi_objects]
        tmp=[d.sigma(inserts_densities,inserts_densities_err) for d in self.dcmi_objects]
        sigmas=tmp[0]
        sigmas_err=tmp[1]
        tmp=sigma_effective(sigmas,sigmas_err,params)
        sigma_eff=tmp[0]
        sigma_eff_err=tmp[1]
        sigma_eff[np.where(sigma_eff==0)]=1e-10
        sigma_eff_err[np.where(sigma_eff_err==0)]=1e-10
        return(sigma_eff,sigma_eff_err)
    
    def sigmas(self,inserts_densities,inserts_densities_err):
        """
        Calculates photon absorption cross sections per electron (sigmas) for images provided based on their CT numbers and electron densities of the phantom.
        
        Parameters
        ----------
        inserts_densities : ndarray (1d)
            array of electron densities of phantom inserts
        inserts_densities_err : ndarray (1d)
            array of uncertainties in electron densities of phantom inserts
        
        Returns
        -------
        tmp : ndarray (shape: (2, number of images provided for deriving sigmas, image width, image height)
            tmp[0] = simga values of the images
            tmp[1] = uncertainties in sigma values of the images
        """
        rhos=[d.rho_e_map(inserts_densities,False) for d in self.dcmi_objects]
        rhos_errs=[d.rho_e_map(inserts_densities_err,True) for d in self.dcmi_objects]
        tmp=[d.sigma(inserts_densities,inserts_densities_err) for d in self.dcmi_objects]
        return(tmp)
    
    def sigma_L_values_inserts(self,inserts_densities,inserts_densities_err,params,I_values,I_values_err,beta,plot=False):
        """
        Deriving photon absorption cross sections per electron (sigmas) and stopping numbers (L) of the phantom images.
        
        Parameters
        ----------
        inserts_densities : ndarray (1d)
            array of electron densities of phantom inserts
        inserts_densities_err : ndarray (1d)
            array of uncertainties in electron densities of phantom inserts
        params : float or ndarray
            float for DECT, ndarray for PCCT
            mixing paramater for deriving effective sigma values from multiple images    
        I : ndarray
            ionisation energies of phantom inserts (eV)
        I_err : ndarray
            uncertainties associated with I (eV)
        beta : float
            v/c for particle
        plot : bool
            if True a plot of effective sigmas vs Ls is shown along with their uncertainties
         
        Returns
        -------
        sigma_vals : ndarray
            array of effective sigma values
        sigma_vals_err : ndarray
            array of uncertainties in effective sigma values   
        L_vals : ndarray
            array of values of L
        L_vals_err : ndarray
            array of values of uncertainties of L
        """
        coords=mapa(self.fns[0])
        sigma=self.sigma_eff_func(inserts_densities,inserts_densities_err,params)
        sigma_vals,sigma_vals_err=[],[]
        for i in coords:
            for j in i:
                    sigma_vals.append(sigma[0][j[0],j[1]])
                    sigma_vals_err.append(sigma[1][j[0],j[1]]) 
        
        tmp=L_rel(beta,I_values,I_values_err)
        L_values=tmp[0]
        L_values_err=tmp[1]
        
        L=[[L_values[i]]*len(coords[i]) for i in range(len(coords))]
        L_err=[[L_values_err[i]]*len(coords[i]) for i in range(len(coords))]
        L_vals,L_vals_err=[],[]
        
        for i in range(len(L)):
            for j,k in zip(L[i],L_err[i]):
                L_vals.append(j)
                L_vals_err.append(k)  
                      
        if plot:
            plt.plot(sigma_vals,L_vals,'bo')
            plt.errorbar(sigma_vals,L_vals,L_vals_err,sigma_vals_err,marker='+',color='black',alpha=0.3)
            plt.xlabel('sigma')
            plt.ylabel('L')
            plt.show()
        return(sigma_vals,sigma_vals_err,L_vals,L_vals_err)
    
    def calibration_fit(self,inserts_densities,inserts_densities_err,params,I_values,I_values_err,beta,deg):
        """
        Calculates coefficients of polynomial fitted to sigma and L values of the phantom images
        
        Parameters
        ----------
        inserts_densities : ndarray (1d)
            array of electron densities of phantom inserts
        inserts_densities_err : ndarray (1d)
            array of uncertainties in electron densities of phantom inserts
        params : float or ndarray
            float for DECT, ndarray for PCCT
            mixing paramater for deriving effective sigma values from multiple images    
        I : ndarray
            ionisation energies of phantom inserts (eV)
        I_err : ndarray
            uncertainties associated with I (eV)
        beta : float
            v/c for particle
        deg : int
            degree of the fitted polynomial
            
        Returns
        -------
        fit : ndarray
            fit[0] = coefficients of the fitted polynomial
            fit[1] = uncertainties associated with the coefficients of the fitted polynomial
        """
        tmp=self.sigma_L_values_inserts(inserts_densities,inserts_densities_err,params,I_values,I_values_err,beta)
        sigma=tmp[0]
        sigma_err=tmp[1]
        L=tmp[2]
        L_err=tmp[3]
        fit=linear_regression_polynomial(sigma,L,sigma_err,L_err,deg)
        return(fit)

    def alpha_blending(self,inserts_densities,inserts_densities_err):
        """
        Performs alpha blending routine described by Möhler et al. (2017) to derive alpha paramaters for calculating electron densities of the CT images
        
        Paramaters
        ----------
        inserts_densities : ndarray (1d)
            array of electron densities of phantom inserts
        inserts_densities_err : ndarray (1d)
            array of uncertainties in electron densities of phantom inserts
            
        Returns
        -------
        out : ndarray
            out[0] = alpha values
            out[1] = uncertainties in alpha values
        """
        if len(self.fns)==2:
            return(alpha_blending_dualE(self.fns,inserts_densities,inserts_densities_err))
        else:
            return(alpha_blending_multiE(self.fns,inserts_densities,inserts_densities_err))
         
class DICOM_Images_POST_calibration():
    """
    Class for deriving quantities of CT images (post calibration)
    
    Parameters
    ----------
    images_fns : ndarray
        array of file names/paths of CT images of the patient (high and low energy scans for DECT, multiple images for PCCT)
    fit : ndarray
        output of calibration_fit() function of DICOM_Images class
    """
    def __init__(self, images_fns,fit):
        self.fns=images_fns
        self.dss = [pydicom.dcmread(i) for i in self.fns]
        self.fit=fit[0]
        self.fit_err=fit[1]
        self.instances=[DICOM_Image(i) for i in self.fns]
        self.CT_numbers = [i.pixel_array*int(i.RescaleSlope)+int(i.RescaleIntercept) for i in self.dss]
        self.CT_numbers_err=[o.CT_noise() for o in self.instances]

        
    def sigmas(self,rho_e,rho_e_err):
        """
        Calculates photon absorption cross sections per electron (sigmas) for images provided based on their CT numbers and electron densities of the image.
        
        Parameters
        ----------
        inserts_densities : ndarray (1d)
            array of electron densities of phantom inserts
        inserts_densities_err : ndarray (1d)
            array of uncertainties in electron densities of phantom inserts
        
        Returns
        -------
        sigmas : ndarray (shape(number of images, image width, image height))
            simga values of the images
        sigmas_err : ndarray (shape(number of images, image width, image height))
            uncertainties in simga values of the images
        """
        sigmas=[(i/1000+1)/rho_e for i in self.CT_numbers]
        sigmas_err=[((1/(1000*rho_e))*(self.CT_numbers_err[i]))**2+(((self.CT_numbers[i]/1000+1)/rho_e**2)*(rho_e_err))**2 for i in range(len(self.fns))]
        return(sigmas,np.sqrt(sigmas_err))
        
        
    def sigma_eff(self,rho,rho_err,p):
        """
        Returns effective photon absorption cross sections per electron (sigmas) of the images.
       
        Parameters
        ----------
        rho : ndarray 
            array (image data) of electron densities of the image
        rho_err : ndarray 
            array (image data) of uncertainties in electron densities of the image
        p : float or ndarray
            float for DECT, ndarray for PCCT
            mixing paramater for deriving effective sigma values from multiple images
            
        Returns
        -------
        out : ndarray
            out[0] = array (image data) of effective sigmas derived from the provided images
            out[1] = array (image data) of uncertainties in effective sigmas derived from the provided images 
        """
        tmp=self.sigmas(rho,rho_err)
        sigms=tmp[0]
        sigms_err=tmp[1]
        return(sigma_effective(sigms,sigms_err,p))
    
    
    def CT_to_L(self,rho_e,rho_e_err,p):
        """
        Derives stopping numbers (L) of the image based on its CT numbers and electron densities
        
        Parameters
        ----------
        rho_e : ndarray
            array (image data) of electron densities of the CT image
        rho_e_err : ndarray
            array (image data) of uncertainties in electron densities of the CT image
        p : float or ndarray
            float for DECT, ndarray for PCCT
            mixing paramater for deriving effective sigma values from multiple images
            
        Returns
        -------
        L : ndarray 
            array (image data) of L values of the image
        L_err : ndarray 
            array (image data) of uncertainties in L values of the image
            
        """
        tmp=self.sigma_eff(rho_e,rho_e_err,p)
        sigmas_values=tmp[0]
        sigmas_values_err=tmp[1]
        L=np.poly1d(self.fit[::-1])(sigmas_values)

        L_prime=np.poly1d(self.fit[::-1]).deriv(1)(sigmas_values)
        L_err=(L_prime*sigmas_values_err)**2
        
        k=[]
        for i in range(len(self.fit)):
            k.append(i)
        
        for i in range(np.shape(sigmas_values)[0]):
            for j in range(np.shape(sigmas_values)[1]):
                for k in range(len(self.fit)):
                    L_err[i][j]+=(sigmas_values[i][j]**k*self.fit_err[k])**2
                    
        return(L,np.sqrt(L_err))

class SPR():
    """
    Class for deriving stopping power relative to water of patient scans
    
    Parameters
    ----------
    file_paths : ndarray
        array of file names/paths of CT images of the patient (high and low energy scans for DECT, multiple images for PCCT)
    fit : ndarray
        output of calibration_fit() function of DICOM_Images class
    
    alpha : ndarray
        output of alpha_blending() function of DICOM_Images class, calculated for scans of calibration phantom
    param_sigma_effective : float or ndarray
            float for DECT, ndarray for PCCT
            mixing paramater for deriving effective sigma values from multiple images
    """
    def __init__(self,file_paths,fit,alpha,param_sigma_effective):
        self.p=param_sigma_effective
        self.alpha=alpha
        self.fit=fit
        self.fps=file_paths
        self.rho_object=Alpha_Blending(self.fps,self.alpha)
        self.rho=self.rho_object.rho_map()
        self.L_object=DICOM_Images_POST_calibration(self.fps,self.fit)
        self.L=self.L_object.CT_to_L(self.rho[0],self.rho[1],self.p)
        
    def spr_map(self):
        val=self.L[0]*self.rho[0]
        err=(self.rho[0]*self.L[1])**2+(self.L[0]*self.rho[1])**2
        return(val,np.sqrt(err))

        
class Alpha_Blending():
    """
    Class for performing alpha blending routine (Möhler et al. (2017) on CT images and deriving electron densities map of the image vased on its CT numbers map and calibration parameters.
    
    Parameters
    ----------
    file_paths : ndarray
        array of file names/paths for high and low energy scans (DECT) or multiple energy bins images (PCCT)
    alpha_blending_paramaters : ndarray
        output of alpha_blending() function of DICOM_Images class, calculated for scans of calibration phantom
    """
    def __init__(self, file_paths,alpha_blending_paramaters):
        self.fns = file_paths
        self.dss = [pydicom.dcmread(i) for i in self.fns]
        self.pixels = [i.pixel_array*int(i.RescaleSlope)+int(i.RescaleIntercept) for i in self.dss]
        self.alpha = alpha_blending_paramaters
        self.dcmi_objects=[DICOM_Image(i) for i in self.fns]
        self.noise=[d.CT_noise() for d in self.dcmi_objects]
        
    def rho_map(self):
        """
        Derives image data (array) of the electron densities of the CT image based on its CT numbers and calibartion paramaters
        
        Paramaters
        ----------
        None
        
        Returns
        -------
        val : ndarray
            array (image data) of electron densities of the image
        err : ndarray
            array (image data) of uncertainties in electron densities of the image
        """
        if len(self.fns)==2:
            val = self.alpha[0]*self.pixels[0]+(1-self.alpha[0])*self.pixels[1]
            err = (self.alpha[0]*self.noise[0])**2+((1-self.alpha[0])*(self.noise[1]))**2+((self.pixels[0]-self.pixels[1])*(self.alpha[1]))**2
            val[np.where(val==0)]=1
            return(val,np.sqrt(err))
        else:
            val=np.array([(sum(self.alpha[0][i]*self.pixels[i] for i in range(len(self.fns))))])
            err1=np.array([(sum(self.alpha[1][i]**2*self.pixels[i]**2 for i in range(len(self.fns))))])
            err2=np.array([(sum(self.alpha[0][i]**2*self.noise[i]**2 for i in range(len(self.fns))))])
            err=err1+err2
            val[0][np.where(val[0]==0)]=1
            return(val[0],np.sqrt(err)[0])