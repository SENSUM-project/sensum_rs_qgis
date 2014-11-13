import numpy as np
import sys,os
import gdal
import shutil
import tempfile
from itertools import chain
from sensum.preprocess import clip_rectangular
from sensum.conversion import read_image,read_image_parameters,world2pixel,write_image

def main():
    input_rasters = map(str, arg.input_rasters)
    output_path = str(arg.output_path)
    merge(output_path,*input_rasters)

def args():
    parser = argparse.ArgumentParser(description='Calculate Height')
    parser.add_argument("output_path", help="????")
    parser.add_argument("-i", "--input_rasters", nargs="+", help="????")
    args = parser.parse_args()
    return args

def merge(output_path,first_path,second_path,*others):
    others = list(others)
    others.append(second_path)
    raster_arrays = RasterGdalExtent(first_path,*others)
    raster_arrays.write(output_path)

def change_projection(first_path,*others):
    raster_ref = RasterArray(first_path)
    rasters = [RasterArray(path,projection=raster_ref._parser_projection) for path in others]

def executeGdal(command):
    if os.name != "posix":
        import ctypes
        SEM_NOGPFAULTERRORBOX = 0x0002 # From MSDN
        ctypes.windll.kernel32.SetErrorMode(SEM_NOGPFAULTERRORBOX);
        bit = ("64" if os.path.isdir("C:/OSGeo4W64") else "")
        prefix = "C:/Python27/python.exe C:/OSGeo4W{}/bin/".format(bit)
        command = prefix + command
    os.system(command)

class RasterArray(object):

    def __init__(self,input_raster,data_type=np.uint16,band_selection=0,projection=None):
        self.input_raster = input_raster
        self.band_list = read_image(input_raster, data_type, band_selection)
        self.rows, self.cols, self.nbands, self.geo_transform, self.projection = read_image_parameters(input_raster)
        if projection and self._parser_projection != projection:
            self.__change_projection(projection)
        self.resolution = abs(self.geo_transform[1])

    @property
    def extent_coordinates(self):
        origin_x = self.geo_transform[0]
        origin_y = self.geo_transform[3]
        end_x = origin_x+self.cols*self.resolution
        end_y = origin_y-self.rows*self.resolution
        return (origin_x,origin_y,end_x,end_y)

    @property
    def _parser_projection(self):
        crs = self.projection.split("\"")
        return "{}:{}".format(crs[-4],crs[-2])

    @property
    def mean(self):
        mean = list()
        for b in range(self.nbands):
            mean.append(np.mean(self.band_list[b]))
        return mean

    @property
    def std(self):
        std = list()
        for b in range(self.nbands):
            std.append(self.band_list[b]-self.mean[b])
        return std

    def pixel_from_coordinate(self,coordinates,band=0):
        try:
            pixel_x, pixel_y = world2pixel(self.geo_transform, coordinates[0], coordinates[1])
            return self.band_list[band][pixel_y][pixel_x]
        except:
            return 0

    def write(self,path=None):
        if path is None:
            path = self.input_raster
        write_image(self.band_list,np.uint16,0,path,self.rows,self.cols,self.geo_transform,self.projection)

    def __change_projection(self,projection_ref):
        tmpfile = tempfile.NamedTemporaryFile().name
        os.system("gdalwarp -overwrite -t_srs {} {} {}".format(projection_ref,self.input_raster,tmpfile))
        shutil.move(tmpfile, self.input_raster)
        self.rows, self.cols, self.nbands, self.geo_transform, self.projection = read_image_parameters(self.input_raster)

class RasterArraysExtent(RasterArray):

    def __init__(self,first_path,*others):
        self.__raster_ref = RasterArray(first_path)
        self.__rasters = [RasterArray(path,projection=self.__raster_ref._parser_projection) for path in others]
        self.__rasters.append(RasterArray(first_path))
        self.__x_coordinates = list(chain.from_iterable([(raster.extent_coordinates[0],raster.extent_coordinates[2]) for raster in self.__rasters]))
        self.__y_coordinates = list(chain.from_iterable([(raster.extent_coordinates[1],raster.extent_coordinates[3]) for raster in self.__rasters]))
        self.__coordinate_x_init = min(self.__x_coordinates)
        self.__coordinate_y_init = max(self.__y_coordinates)
        self.__coordinate_x_final = max(self.__x_coordinates)
        self.__coordinate_y_final = min(self.__y_coordinates)
        self.resolution = self.__raster_ref.resolution
        self.__range_x = np.arange(self.__coordinate_x_init, self.__coordinate_x_final, self.resolution)
        self.__range_y = np.arange(self.__coordinate_y_init, self.__coordinate_y_final, -self.resolution)
        self.geo_transform = [self.__coordinate_x_init, self.__raster_ref.geo_transform[1], self.__raster_ref.geo_transform[2], self.__coordinate_y_init, self.__raster_ref.geo_transform[4], self.__raster_ref.geo_transform[5]]
        self.projection = self.__raster_ref.projection
        self.cols = len(self.__range_x)
        self.rows = len(self.__range_y)
        self.band_list = self.__band_list()

    def __iter__(self):
        return iter(self.__rasters)

    def __getitem__(self,index):
        return self.__rasters[index]

    def _std_fix(self):
        for i in range(1,len(self.__rasters)):
            output_band = list()
            for b in range(self.__rasters[0].nbands):
                ref_mean = self.__rasters[0].mean[b]
                output_band.append(ref_mean + self.__rasters[i].std[b])
            self.__rasters[i].band_list = output_band

    def __band_list(self,std_fix=True):
        if std_fix:
            self._std_fix()
        output_band = np.zeros((self.rows,self.cols),dtype=np.uint16)
        for i,coordinate_x in enumerate(self.__range_x):
            sys.stdout.write("\r{}/{}".format(i+1,self.cols))
            sys.stdout.flush()
            for coordinate_y in self.__range_y:
                for raster_array in self.__rasters:
                    pixel = raster_array.pixel_from_coordinate((coordinate_x,coordinate_y))
                    if pixel:
                        pixel_x, pixel_y = world2pixel(self.geo_transform, coordinate_x, coordinate_y)
                        output_band[pixel_y][pixel_x] = pixel
                        break
        return [output_band]

    @property
    def extent_coordinates(self):
        return (self.__coordinate_x_init,self.__coordinate_y_init,self.__coordinate_x_final,self.__coordinate_y_final)

class RasterGdalExtent(RasterArraysExtent):

    GDAL_MERGE_CMD =  "gdal_merge.py"

    def __init__(self,first_path,*others):
        self.__raster_ref = RasterArray(first_path)
        self.__rasters = [RasterArray(path,projection=self.__raster_ref._parser_projection) for path in others]
        self.__rasters.append(RasterArray(first_path))
        self.band_list = self.__band_list()
        self.resolution = self.__raster_ref.resolution
        self.rows, self.cols, self.nbands, self.geo_transform, self.projection = read_image_parameters(self.tmpfile)

    def __band_list(self,std_fix=True):
        raster_arrays_str = " ".join([raster.input_raster for raster in self.__rasters])
        self.tmpfile = tempfile.NamedTemporaryFile().name
        cmd = RasterGdalExtent.GDAL_MERGE_CMD+str(" -init \"0 0 255\" -n 0 -o {} ".format(self.tmpfile))+raster_arrays_str
        executeGdal(cmd)
        output_band = read_image(self.tmpfile,0,0)
        return output_band

    @property
    def extent_coordinates(self):
        return (self.geo_transform[0],self.geo_transform[3],self.geo_transform[0]+self.cols,self.geo_transform[3]+self.rows)

    def write(self,path=None):
        shutil.move(self.tmpfile,path)

if __name__ == "__main__":
    main()