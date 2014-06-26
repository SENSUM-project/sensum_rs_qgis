import config
import gdal
import ogr
import os
import psycopg2

class Postgis(object): #conn is the dataset
    def __init__(self,user,dbname,host='localhost',password=None, port=5432):
        self.user, self.password, self.dbname, self.host, self.port = user, password, dbname, host, port
        self.conn = None
    def getConnection(self):
        if self.password:
            self.conn = ogr.Open("PG: host='"+self.host+"' dbname='"+self.dbname+"' user='"+self.user+"' port="+str(self.port)+" password='"+self.password+"'")
        else:
            self.conn = ogr.Open("PG: host='"+self.host+"' dbname='"+self.dbname+"' user='"+self.user+"' port="+str(self.port))
        return self.conn
    def getRaster(self,table): #work in memory without saving
        if self.password:
            self.conn = gdal.Open("PG: host='"+self.host+"' dbname='"+self.dbname+"' user='"+self.user+"' port="+str(self.port)+" table='"+table+"'"" password='"+self.password+"'")
        else:
            self.conn = gdal.Open("PG: host='"+self.host+"' dbname='"+self.dbname+"' user='"+self.user+"' port="+str(self.port)+" table='"+table+"'")
        return self.conn
    def getLayer(self,table):
        if self.conn:
            self.layer = self.conn.ExecuteSQL("select geom from "+table)
        else:
            self.getConnection()
            self.layer = self.conn.ExecuteSQL("select geom from "+table)
        return self.layer
    def execQuery(self,query):
        self.layer = self.conn.ExecuteSQL(query)
    def getGeometry(self):
        while feature:
            feature = layer.GetNextFeature()
            geometry = feature.GetGeometryRef()
            print geometry
    def img2db(self,path,table):
        os.system("raster2pgsql -I -C -M "+path+" -F "+table+" > tmp.sql")
        if self.password:
            if os.name == 'posix':
                os.system("export PGPASSWORD='"+self.password+"'; psql -h "+self.host+" -d "+self.dbname+" -U "+self.user+" -p "+str(self.port)+" -f tmp.sql")
            else:
                os.system("set PGPASSWORD='"+self.password+"'")
                os.system("psql -h "+self.host+" -d "+self.dbname+" -U "+self.user+" -p "+str(self.port)+" -f tmp.sql")
        else:
            os.system("psql -h "+self.host+" -d "+self.dbname+" -U "+self.user+" -p "+str(self.port)+" -f tmp.sql")
        os.system("rm tmp.sql")
    def shp2db(self,path,table):
        os.system("shp2pgsql "+path+" "+table+" > tmp.sql")
        if self.password:
            if os.name == 'posix':
                os.system("export PGPASSWORD='"+self.password+"'; psql -h "+self.host+" -d "+self.dbname+" -U "+self.user+" -p "+str(self.port)+" -f tmp.sql")
            else:
                os.system("set PGPASSWORD='"+self.password+"'")
                os.system("psql -h "+self.host+" -d "+self.dbname+" -U "+self.user+" -p "+str(self.port)+" -f tmp.sql")
        else:
            os.system("psql -h "+self.host+" -d "+self.dbname+" -U "+self.user+" -p "+str(self.port)+" -f tmp.sql")
        os.system("rm tmp.sql")
    def db2img(self,path,table=None):
        if table: #getRaster method need to be just called
            self.getRaster(table)
            driver = gdal.GetDriverByName("GTiff")
        else:
            driver = gdal.GetDriverByName("GTiff")
        driver.CreateCopy(path, self.conn, 0)
    def db2shp(self,path,table):
        if self.password:
            os.system("pgsql2shp -f "+path+" -h "+self.host+" -u "+self.user+" -p "+str(self.port)+" -P "+self.password+" "+self.dbname+" "+table)
        else:
            os.system("pgsql2shp -f "+path+" -h "+self.host+" -u "+self.user+" -p "+str(self.port)+" "+self.dbname+" "+table)

class SensumDB(object):
    #query generator
    def __init__(self,user,dbname,host='localhost',password=None, port=None):
        self.con = psycopg2.connect(database=dbname, user=user, host=host, password=password)
        self.cur = self.con.cursor()
    def objectMain(self):
        self.objectMain = ObjectMain(self)
        return self.objectMain
    def objectMainDetail(self):
        self.objectMainDetail = ObjectMainDetail(self)
        return self.objectMainDetail
    def runQuery(self,query):
        self.cur.execute(query)
        self.con.commit()
    def runQueryMany(self,query,array):
        self.cur.executemany(query, array)
        self.con.commit()
    def fetchQuery(self,query):
        self.cur.execute(query)
        result = self.cur.fetchall()
        return result

class Child(object):
    #builder method 
    def __init__(self,parent):
        self.parent = parent

class ObjectMain(Child):
    #common queries
    def add(self,survey_gid,description,source,resolution):
        query = "INSERT INTO object.main (survey_gid,description,source,resolution) VALUES (\'"+str(survey_gid)+"\',\'"+str(description)+"\',\'"+str(source)+"\',\'"+str(resolution)+"\')"
        self.parent.runQuery(query)
    def addArray(self,array):
        query = "INSERT INTO object.main (survey_gid,description,source,resolution) VALUES (%s, %s, %s, %s)"
        self.parent.runQueryMany(query,array)
    def get(self):
        query = "SELECT * FROM object.main"
        rows = self.parent.fetchQuery(query)
        return rows
    def remove(self,gid):
        query = "DELETE FROM object.main WHERE gid = " + str(gid)
        self.parent.runQuery(query)
    def update(self,survey_gid,description,source,resolution,gid):
        query = "UPDATE object.main SET survey_gid = "+str(survey_gid)+" , description = \'"+str(description)+"\' , source = \'"+str(source)+"\' , resolution = "+str(resolution)+" WHERE gid = " + str(gid)
        self.parent.runQuery(query)

class ObjectMainDetail(Child):
    #common query -> table ObjectMainDetail
    def add(self,object_id,resolution2_id,resolution3_id,attribute_type_code,attribute_value,attribute_numeric_1, attribute_numeric_2,attribute_text_1):
        query = "INSERT INTO object.main_detail (object_id,resolution2_id,resolution3_id,attribute_numeric_1, attribute_numeric_2,attribute_text_1) VALUES (\'"+str(object_id)+"\',\'"+str(resolution2_id)+"\',\'"+str(resolution3_id)+"\',\'"+str(attribute_numeric_1)+"\',\'"+str(attribute_numeric_2)+"\',\'"+str(attribute_text_1)+"\')"
        self.parent.runQuery(query)

if __name__ == '__main__':

    ####################################
    #### SensumDB CALLING EXAMPLES #####
    ####################################

    sensum = SensumDB("postgres", "Izmir", password="postgres")
    obj = sensum.objectMain()
    objdetail = sensum.objectMainDetail()

    # Add single Object
    obj.add(0,"Maserati","Granturismo",1000)
    objdetail.add(107,0,0,"Maserati","Granturismo",1000,1000,"Fiat",)

    # Add Object as Array
    objs = (
        (1, 'Audi', 'A3' , 52642),
        (2, 'Mercedes', 'CLASS A', 57127),
        (3, 'Skoda', 'Fabia', 9000),
        (4, 'Citroen', 'C3', 21000),
        (5, 'Volkswagen', 'Golf', 21600),
        (6, 'Fiat', 'Panda' , 52642),
        (7, 'Alfa Romeo', 'Spider', 57127),
    )
    obj.addArray(objs)

    # Get all Object
    rows = obj.get()
    for row in rows:
        print row

    # Remove Object
    obj.remove(106)

    # Update Object
    obj.update(0, 'Red Bull', 'RB10', 52642, 313)

    del obj
    del objdetail
    del sensum

    ###################################
    #### Postgis CALLING EXAMPLES #####
    ###################################
   
    dbname = 'Izmir'
    host = 'localhost'
    user = 'postgres'
    password = 'postgres'

    # Upload shape file to db
    watershed = Postgis(user,dbname)
    watershed.shp2db("wetransfer-749d73/watershed_005.shp", 'example1')
    del watershed

    # Get shape layer from db
    watershed = Postgis('postgres','Izmir')
    watershed.getConnection()
    print watershed.getLayer('example1').GetFeatureCount()
    del watershed

    # Save a copy of shape file from db
    watershed = Postgis('postgres', 'Izmir')
    watershed.db2shp('example.shp','example1')
    del watershed

    # Upload raster file to db
    pansharp = Postgis(user,dbname)
    pansharp.img2db("wetransfer-749d73/pansharp.TIF","pansharp")
    del pansharp

    # Get raster with Gdal driver from db
    pansharp = Postgis(user,dbname)
    inputimg = pansharp.getRaster("pansharp")
    print inputimg.RasterCount
    del pansharp

    # Save a copy of raster from db
    pansharp = Postgis(user,dbname)
    pansharp.db2img("example.tif","pansharp")
    del pansharp