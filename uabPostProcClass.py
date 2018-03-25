"""
% Python version of spClass_confMapToPolygonStructure_v2
% Version: 2
% Date: 03/20/2018
% Author: Jordan Malof
% Email:  jmmalo03@gmail.com
%
%%%%%%%%%  DOC DETAILS:  SEE MATLAB IMPLEMENTATION 		%%%%%%%%%%%%%%%%%
%%%%%%%%%  EXAMPLE CODE: SEE IN THE MAIN FUNCTION BELOW %%%%%%%%%%%%%%%%%
%
% NOTE: Polygons are in form of XY
%"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from skimage import measure
from scipy.spatial import KDTree
import cv2


class spClass_confMapToPolygonStructure_v2:
	version = 1
	#------------ panel params ------------
	minRegion = 10 # any detected regions must be at least this large
	maxThreshold = 140
	minThreshold = 20
	#------------ polygon params ------------
	epsilon = 2
	linkingRadius = 55
	#------------ commercial params ------------	
	commercialAreaThreshold = 1500;  # Any panel above this threshold (but below maxRegion threshold) is labeled as commercial
	commercialPanelDensityThreshold = 0.2;  # Any panel that is within a region with panel density above this threshold is labeled as commercial
	commercialNeighborhoodRadius = 50;

	def __init__(self):
		self.objectStructure = pd.DataFrame(
			columns=['iLocation','jLocation','pixelList','confidence','area','maxIntensity','isCommercial'])
		self.objectStructure.pixelList = self.objectStructure.pixelList.astype(object)

	def dropStructures(self):
		self.objectStructure = self.objectStructure.iloc[0:0]
		self.__init__()

	def getConfigs(self):
		return "minR{:d}-maxT{:d}-minT{:d}-ComA{:d}-DenT{:f}-Rad{:d}".format(
			self.minRegion,self.maxThreshold,self.minThreshold,self.commercialAreaThreshold,
			self.commercialPanelDensityThreshold,self.commercialNeighborhoodRadius)


	"""  MAP THE POLYGONS ONTO AN IMAGE FOR DISPLAY """
	def polygonStructureToImage(self,confidenceImage): #polygon cords in form of xy
		H,W = confidenceImage.shape
		polygonImage = np.zeros((H,W))
		for poly in self.objectStructure.polygon: 
			img = Image.new('L', (W,H), 0)
			ImageDraw.Draw(img).polygon(poly.ravel().tolist(), outline=1, fill=1)
			polygonImage += np.array(img,dtype=bool)
		return polygonImage


	"""  CREATE REGIONS FROM CONFIDENCE MAPS """
	def confidenceImageToObjectStructure(self,confidenceImage):
		imThresh = confidenceImage>=self.minThreshold
		imLabel = measure.label(imThresh)
		regProps = measure.regionprops(imLabel, confidenceImage)
		for rp in regProps:
			if rp.area>=self.minRegion and rp.max_intensity>=self.maxThreshold:
				temp = [*[int(c) for c in rp.centroid],rp.coords,rp.mean_intensity,rp.area,rp.max_intensity,0]
				self.objectStructure = self.objectStructure.append(
					dict(zip(['iLocation','jLocation','pixelList','confidence','area','maxIntensity','isCommercial'],temp)),ignore_index=True)


	def addCommercialLabelToObjectStructure(self,confidenceImage,return_sum=False):
		if not self.objectStructure.empty:
			""" IDENTIFY USING CONNECTED COMPONENT SIZE """ 
			objAreas = self.objectStructure['area']
			isCommercialSize = objAreas>=self.commercialAreaThreshold

			""" IDENTIFY USING PANEL PIXEL DENSITY """ 
			neighborhoodFilter = np.ones([2*self.commercialNeighborhoodRadius+1]*2)
			dummyImage = np.zeros(confidenceImage.shape)
			pixelList = np.vstack(np.array(self.objectStructure['pixelList'])).transpose()
			dummyImage[pixelList[0],pixelList[1]] = 1

			panelNeighborhoodCount = np.real(np.fft.ifft2(np.fft.fft2(dummyImage)*np.fft.fft2(neighborhoodFilter,dummyImage.shape))) # imfilt
			countThreshold = int(self.commercialPanelDensityThreshold*np.prod(neighborhoodFilter.shape))
			commercialPanelMap = panelNeighborhoodCount>countThreshold
			commercialCenters = np.vstack(np.nonzero(commercialPanelMap)).transpose()

			if commercialCenters.any():
				panelCenters = np.vstack((self.objectStructure['iLocation'],self.objectStructure['jLocation'])).transpose()
				Mdl = KDTree(commercialCenters)
				# Search for panels with neighborhood radius of a commercial center
				out = Mdl.query_ball_point(panelCenters,self.commercialNeighborhoodRadius)
				isCommercialDensity = [bool(i) for i in out]
			else:
				isCommercialDensity = np.zeros(objAreas.shape,dtype=bool)

			self.objectStructure.isCommercial = isCommercialSize & isCommercialDensity

			if return_sum:
				non_commercial = self.objectStructure.loc[self.objectStructure['isCommercial'] is False]
				


	def addPolygonToObjectStructure(self,predIm):
		polygons = [list() for _ in range(self.objectStructure.shape[0])]
		for r,reg in self.objectStructure.iterrows():
			dummyImage = np.zeros(predIm.shape)
			pixl = reg['pixelList'].transpose()
			dummyImage[pixl[0],pixl[1]] = 1
			_,contours,_=cv2.findContours(dummyImage.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
			polygons[r] = measure.approximate_polygon(np.squeeze(contours),self.epsilon) # Douglas Peucker
		self.objectStructure['polygon']=pd.Series(polygons,index=self.objectStructure.index).astype(object)



	""" LINK EACH HOUSE WITH ONE (OR MORE) DETECTED PANELS """
	def linkHousesToObjects(self,housePixelCoordinates,houseIdList):
		raise ValueError('To be translated')

#             %'objectStructure' must contain the centroid of each solar
#             % array object detected (given by iLocation and jLocation)
#             %
#             %'housePixelCoordinates' must contain the i,j pixel locations
#             % of each house in the image
#             %
#             %'houseIdList' must contain the identification number for the 
#             % houses.  
#             %
#             %This function assigns one, or no, houses to each solar panel.
#             % The houseId is a new field in 'objectStructure' 
        
#             %Build a k-d tree of the panel locations
#             ptsDet = [[objectStructure.iLocation]',[objectStructure.jLocation]'];
#             Mdl = KDTreeSearcher(housePixelCoordinates);
            
#             %Do a search of all panels that are within a radius 
#             % of each house.  
#             out = Mdl.rangesearch(ptsDet,self.linkingRadius);  %pretty sure it puts closest panel first
            
#             %Map index to house ID
#             for iOut = 1:length(out)
#                 if ~isempty(out{iOut})
#                     %Only take first match (closest house to the panel
#                     objectStructure(iOut).houseId = houseIdList(out{iOut}(1));
#                 else
#                     %No house-match found
#                     objectStructure(iOut).houseId = -1;  
#                 end
#             end
            
#         end


# example 
if __name__=="__main__":
	i_name = "/home/jordan/Documents/data/solar/uab_datasets/Results/Prediction/UnetCropCV_(FixedRes)CTFinetune+nlayer9_PS(572, 572)_BS5_EP100_LR1e-05_DS50_DR0.1_SFN32/fold0/testing/Groton/185695_ne_CONFPROC_x10x1300x40x140_14252564.jpg"
	conf_im = np.asarray(Image.open(i_name)).astype(np.uint8)

	# Instantiate the class
	ppObj = spClass_confMapToPolygonStructure_v2()

	# Map tp objects
	ppObj.confidenceImageToObjectStructure(conf_im)

	# Assign commercial labels to panels
	ppObj.addCommercialLabelToObjectStructure(conf_im)

	# APPROXIMATE EACH OBJECT WITH POLYGON
	ppObj.addPolygonToObjectStructure(conf_im)
	# print(ppObj.objectStructure)
	
	# FOR TESTING, PLOT THE POLYGONS ON AN IMAGE
	polygonImage = ppObj.polygonStructureToImage(conf_im)
	plt.figure()
	plt.imshow(polygonImage)
	plt.colorbar()
	plt.show()

	# # Link OBJECTS to houses (Note that this can be done before you create polygons)
	# # This assumes you have loaded in the house IDs and locations
	# linkedList = ppObj.linkHousesToObjects(objectStructure,housePixelLocations,houseIDs);