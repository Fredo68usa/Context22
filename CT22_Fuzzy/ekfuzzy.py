# version 1.1 : automatic addition of SQLs being tested


from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import psycopg2
import socket
import sys


# Defining a class
class EKFuzzy:
    def __init__(self):
        self.minScore = 30

        #Access to PostGreSQL
        self.postgres_connect = None
        self.cursor = None
        if len(sys.argv) < 2 :
           self.tbcSQL = None
        else : 
           self.tbcSQL = sys.argv[1].upper()



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


    def posGresPrep(self):
        self.open_PostGres()
        self.cursor = self.postgres_connect.cursor()
        print ( self.postgres_connect.get_dsn_parameters(),"\n")
        self.cursor.execute("SELECT version();")
        record = self.cursor.fetchone()
        print("You are connected to - ", record,"\n")

        postgres_currentSQLs_query = """ SELECT * FROM currentsqls"""
        self.cursor.execute(postgres_currentSQLs_query)
        self.sql_records = self.cursor.fetchall()

    def checkFuzzy(self):
      ListLikeSQL = []
      for row in self.sql_records:
          # print (row[0], "  ", row[1])
          refSQL = row[1].upper()
          # print (fuzz.ratio(self.tbcSQL, row[1]), "--" ,fuzz.partial_ratio(self.tbcSQL, row[1]))
          print ('Tested against ' , row[1] , "Standard Fuzzy score :" ,fuzz.ratio(self.tbcSQL, refSQL), "-- Partial Fuzzy score :" ,fuzz.partial_ratio(self.tbcSQL, refSQL))
          current_score = fuzz.partial_ratio(self.tbcSQL, refSQL)
          if current_score >= self.minScore :
              likeSQL = []
              likeSQL.append(row[0])
              likeSQL.append(row[1])
              likeSQL.append(current_score)
              ListLikeSQL.append(likeSQL)
              # print (' Current like SQL statement : ' ,likeSQL)


      print ("To Be Recorded")
      self.write_PosGres(ListLikeSQL)



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
        self.checkFuzzy()
        self.close_PosGres()


