# version 1.0 : 


import psycopg2
import socket
import sys


# Defining a class
class CT22PosGreSQL:
    def __init__(self):

        #Access to PostGreSQL
        self.postgres_connect = None
        self.cursor = None

    # Opening PostGreSQL
    def open_PostGres(self):

      try:
         self.postgres_connect = psycopg2.connect(user = "context22",
                                  # password = "AIM2020",
                                  # host = "127.0.0.1",
                                  port = "5432",
                                  database = "context22"
                                  )


      except (Exception, psycopg2.Error) as error :
         print("Error while connecting to PostgreSQL", error)
         print ("Hello")


      self.cursor = self.postgres_connect.cursor()
      print ( self.postgres_connect.get_dsn_parameters(),"\n")
      self.cursor.execute("SELECT version();")
      record = self.cursor.fetchone()
      print("You are connected to - ", record,"\n")

    def write_PosGres(self,ListLikeSQL):
        print (" In write_PosGres" ) 
        SQL_value = self.tbcSQL
        # postgres_currentSQLs_query = """ INSERT INTO currentSQLs (hash,SQL,frequency) VALUES (md5(%s),%s,%s);"""
        postgres_currentSQLs_query = """ INSERT INTO currentSQLs (hash,SQL,status,frequency) VALUES (md5(%s),%s,%s,%s);"""
        # print ("Length of hashSim " , len(hashSim) , " -- " , hashSim )
        self.cursor.execute(postgres_currentSQLs_query, (SQL_value, SQL_value, 'Pending' , 300 ,))
        try :
           self.postgres_connect.commit()
        except :
           pass
        postgres_similarSQLs_query = """ INSERT INTO similarSQLs (hash,hashSim,status,score) VALUES (md5(%s),%s,%s,%s);"""
        for sqlSim in ListLikeSQL:
            self.cursor.execute(postgres_similarSQLs_query, (SQL_value, sqlSim[0], 'Pending', sqlSim[2] ,))
            try :
               self.postgres_connect.commit()
            except:
               pass

    def close_PosGres(self):
        self.postgres_connect.close()

    def main_process(self):
        self.posGresPrep()
        self.close_PosGres()


